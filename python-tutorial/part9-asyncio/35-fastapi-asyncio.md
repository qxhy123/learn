# 第35章：FastAPI 与 asyncio 实战

> FastAPI 不只是一个 Web 框架，它是一个建立在 `asyncio` 之上的**异步服务运行时**。理解 FastAPI 与 asyncio 的协作方式，你才能写出既高效又可靠的推理服务、API 网关和流式处理系统。

> **前置阅读**：本章假设你已完成第25-30章（asyncio 基础、同步原语、与同步代码集成）的学习。

---

## 学习目标

完成本章学习后，你将能够：

1. 理解 FastAPI 的 `async def` 路由如何与 `asyncio` 事件循环协作
2. 正确使用 `Depends`、`lifespan`、`BackgroundTasks` 等核心机制
3. 用 `asyncio.Semaphore`、`asyncio.timeout`、`asyncio.Event` 为 FastAPI 服务加入限流、超时和就绪屏障
4. 实现 WebSocket 流式推理接口
5. 避免在异步路由中阻塞事件循环的常见错误
6. 构建一个生产级异步推理网关

---

## 35.1 为什么选择 FastAPI

FastAPI 是目前 Python 生态中与 `asyncio` 结合最紧密的 Web 框架之一。

### 与深度学习服务的契合点

| 特性 | 对 ML 服务的价值 |
|------|----------------|
| 原生 `async def` 路由 | 并发处理多个推理请求，不阻塞 |
| Pydantic 数据验证 | 自动验证输入特征、返回结构化响应 |
| 自动 OpenAPI 文档 | 前端 / 客户端开箱即用 |
| `lifespan` 生命周期 | 模型预加载、资源清理 |
| WebSocket 支持 | 流式推理、实时交互 |
| 与 `asyncio` 工具无缝集成 | Semaphore、timeout、Event 直接可用 |

### FastAPI 在 asyncio 架构中的位置

```text
HTTP 请求
    │
    ▼
Uvicorn（ASGI 服务器，基于 asyncio）
    │
    ▼
FastAPI（路由分发、中间件、依赖注入）
    │
    ▼
async def 路由函数
    │
    ├─ await asyncio 原语（Semaphore、timeout、Event ...）
    ├─ await 异步 I/O（数据库、Redis、HTTP 客户端 ...）
    └─ asyncio.to_thread()（同步模型推理）
```

FastAPI 本身是 ASGI 应用，运行在 `uvicorn` 启动的事件循环上。  
路由函数里的所有 `asyncio` 知识——Task、Semaphore、timeout、Event——都可以直接使用。

---

## 35.2 FastAPI 核心概念

### 35.2.1 最小应用

```python
from fastapi import FastAPI

app = FastAPI(title="推理服务", version="1.0.0")

@app.get("/health")
async def health():
    return {"status": "ok"}
```

启动：
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

访问 `http://localhost:8000/docs` 可以看到自动生成的交互式 API 文档。

### 35.2.2 路径参数与查询参数

```python
from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI()

@app.get("/models/{model_name}/predict")
async def predict(
    model_name: str,                                    # 路径参数
    top_k: int = Query(default=1, ge=1, le=10),        # 查询参数，带验证
    threshold: Optional[float] = Query(default=None),  # 可选参数
):
    return {"model": model_name, "top_k": top_k, "threshold": threshold}
```

### 35.2.3 Pydantic 请求体与响应模型

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from typing import List

app = FastAPI()


class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_length=1, description="输入特征向量")
    top_k: int = Field(default=1, ge=1, le=10)

    @field_validator("features")
    @classmethod
    def check_no_nan(cls, v: List[float]) -> List[float]:
        import math
        if any(math.isnan(x) for x in v):
            raise ValueError("features 中不能包含 NaN")
        return v


class Prediction(BaseModel):
    class_id: int
    probability: float
    label: str


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    model_name: str
    inference_ms: float


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    # 实际推理逻辑见后续章节
    ...
```

Pydantic 会在请求进入路由之前自动完成：
- 类型转换（`"1.0"` → `1.0`）
- 字段验证（`ge=1` 意味着必须 ≥ 1）
- 自定义 validator（`check_no_nan`）

### 35.2.4 错误响应

```python
from fastapi import FastAPI, HTTPException, status

app = FastAPI()


@app.post("/predict")
async def predict(request: PredictRequest) -> PredictResponse:
    if len(request.features) != 784:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"特征维度错误：期望 784，实际 {len(request.features)}",
        )
    ...
```

也可以注册全局异常处理器：

```python
from fastapi import Request
from fastapi.responses import JSONResponse


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "invalid_input", "detail": str(exc)},
    )
```

---

## 35.3 `async def` 路由与 asyncio 协同

### sync vs async 路由的本质区别

FastAPI 同时支持 `def` 和 `async def` 路由，但它们的运行方式完全不同：

| 路由类型 | 运行方式 | 适合场景 |
|---------|---------|---------|
| `async def` | 在事件循环线程上运行 | 异步 I/O：HTTP 客户端、异步数据库 |
| `def` | 在独立线程池中运行 | 同步阻塞操作：同步 ORM、同步推理库 |

一个关键原则：

> 在 `async def` 路由里调用阻塞同步函数（如 `model.forward()`）会卡死整个事件循环。

### 在 `async def` 路由中运行同步推理

正确做法是把同步推理放进线程池：

```python
import asyncio
from fastapi import FastAPI
from model import ModelInference   # 同步推理封装

app = FastAPI()
inference_engine = ModelInference("weights/model.pt")


@app.post("/predict")
async def predict(request: PredictRequest) -> PredictResponse:
    # ❌ 错误：直接调用同步函数，会阻塞事件循环
    # result = inference_engine.predict(request.features)

    # ✅ 正确：放进线程池，不阻塞事件循环
    result = await asyncio.to_thread(
        inference_engine.predict,
        request.features,
        request.top_k,
    )
    return PredictResponse(predictions=result, ...)
```

### 在路由中使用 asyncio 并发

需要并发抓取多个上游特征时：

```python
import asyncio
import httpx


@app.post("/aggregate")
async def aggregate(user_id: int) -> dict:
    async with httpx.AsyncClient() as client:
        # 并发抓取三个特征服务
        profile, risk, rec = await asyncio.gather(
            client.get(f"http://profile-svc/user/{user_id}"),
            client.get(f"http://risk-svc/user/{user_id}"),
            client.get(f"http://rec-svc/user/{user_id}"),
        )

    return {
        "profile": profile.json(),
        "risk": risk.json(),
        "recommendation": rec.json(),
    }
```

---

## 35.4 依赖注入（Depends）

`Depends` 是 FastAPI 最强大的特性之一，它让你可以把"共享资源"从路由函数里分离出来，统一管理和复用。

### 35.4.1 基础依赖

```python
from fastapi import FastAPI, Depends, Header, HTTPException

app = FastAPI()


def get_api_key(x_api_key: str = Header(...)) -> str:
    """从请求头中提取并验证 API Key"""
    if x_api_key != "secret-key":
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


@app.post("/predict")
async def predict(
    request: PredictRequest,
    api_key: str = Depends(get_api_key),   # 注入依赖
) -> PredictResponse:
    ...
```

### 35.4.2 共享异步资源（模型实例、数据库连接）

```python
from fastapi import FastAPI, Depends
from model import ModelInference

app = FastAPI()

# 模块级全局实例（通过 lifespan 在启动时初始化，见 35.5）
_inference_engine: ModelInference | None = None


def get_inference_engine() -> ModelInference:
    """依赖函数：返回全局模型实例"""
    if _inference_engine is None:
        raise RuntimeError("inference engine not initialized")
    return _inference_engine


@app.post("/predict")
async def predict(
    request: PredictRequest,
    engine: ModelInference = Depends(get_inference_engine),
) -> PredictResponse:
    result = await asyncio.to_thread(engine.predict, request.features, request.top_k)
    return PredictResponse(predictions=result, ...)
```

### 35.4.3 异步依赖（async Depends）

依赖函数也可以是 `async def`：

```python
import asyncio
import redis.asyncio as aioredis

_redis_client: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """返回全局 Redis 连接"""
    if _redis_client is None:
        raise RuntimeError("Redis not initialized")
    return _redis_client


@app.post("/cached_predict")
async def cached_predict(
    request: PredictRequest,
    redis: aioredis.Redis = Depends(get_redis),
    engine: ModelInference = Depends(get_inference_engine),
) -> PredictResponse:
    cache_key = f"pred:{hash(tuple(request.features))}"

    cached = await redis.get(cache_key)
    if cached:
        return PredictResponse.model_validate_json(cached)

    result = await asyncio.to_thread(engine.predict, request.features, request.top_k)
    response = PredictResponse(predictions=result, ...)
    await redis.setex(cache_key, 300, response.model_dump_json())
    return response
```

### 35.4.4 依赖的生命周期管理（`yield` 依赖）

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator


async def get_db_session() -> AsyncGenerator:
    """每个请求创建一个数据库 session，请求结束后关闭"""
    session = await create_async_session()
    try:
        yield session
    finally:
        await session.close()


@app.post("/log_prediction")
async def log_prediction(
    request: PredictRequest,
    session = Depends(get_db_session),   # 请求级 session
) -> dict:
    await session.execute(...)
    return {"logged": True}
```

---

## 35.5 `lifespan` 与应用生命周期

`lifespan` 是 FastAPI 管理应用级资源的标准方式——在服务启动时初始化，在关闭时清理。

### 基础模式

```python
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from model import ModelInference
import redis.asyncio as aioredis

# 应用级状态
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动 → 运行 → 关闭的完整生命周期"""
    print("启动：加载模型...")
    _state["engine"] = ModelInference("weights/model.pt")

    print("启动：连接 Redis...")
    _state["redis"] = await aioredis.from_url("redis://localhost:6379")

    print("服务就绪")
    yield   # 服务运行期间，在这里阻塞

    # 关闭清理
    print("关闭：释放 Redis 连接")
    await _state["redis"].aclose()
    print("关闭完成")


app = FastAPI(lifespan=lifespan)
```

### 结合 `asyncio.Event` 的就绪屏障

当初始化耗时较长（如加载大模型）时，可以用 `Event` 让健康检查接口感知就绪状态：

```python
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

_ready = asyncio.Event()
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 异步加载（例如从远程下载权重）
    _state["engine"] = await load_model_async("s3://bucket/model.pt")
    _ready.set()   # 通知"服务已就绪"
    yield
    _ready.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    if not _ready.is_set():
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}


@app.post("/predict")
async def predict(request: PredictRequest) -> PredictResponse:
    # 如果服务还没就绪，等待（通常只在启动最初几秒会触发）
    await asyncio.wait_for(_ready.wait(), timeout=10.0)
    engine = _state["engine"]
    ...
```

---

## 35.6 后台任务

FastAPI 提供了 `BackgroundTasks`，与 `asyncio.create_task()` 有相似之处但用途不同。

### `BackgroundTasks`

```python
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()


async def log_to_db(request_id: str, result: dict) -> None:
    """请求完成后异步写入日志，不影响响应时间"""
    await asyncio.sleep(0)   # 示例：实际是写数据库
    print(f"logged {request_id}: {result}")


@app.post("/predict")
async def predict(
    request: PredictRequest,
    background_tasks: BackgroundTasks,
) -> PredictResponse:
    result = await asyncio.to_thread(_engine.predict, request.features)
    response = PredictResponse(predictions=result, ...)

    # 响应已发送给客户端后，再执行后台任务
    background_tasks.add_task(log_to_db, request_id="abc", result=result)

    return response
```

### `BackgroundTasks` vs `asyncio.create_task()`

| | `BackgroundTasks` | `asyncio.create_task()` |
|--|--|--|
| 执行时机 | 响应发出后 | 立即并发 |
| 生命周期 | 绑定到请求 | 独立于请求 |
| 错误处理 | 失败静默（需自行捕获） | 可通过 Task 回调处理 |
| 适合场景 | 请求后日志、通知 | 后台长期运行的任务 |

```python
# asyncio.create_task 适合"触发后不管"的长期后台任务
@app.on_event("startup")   # 或在 lifespan 里
async def start_background_worker():
    asyncio.create_task(periodic_cache_flush(), name="cache-flush")


async def periodic_cache_flush():
    while True:
        await asyncio.sleep(60)
        await flush_cache()
```

---

## 35.7 中间件

中间件可以在请求/响应经过路由之前/之后执行逻辑。

### 请求计时与请求 ID 注入

```python
import time
import uuid
from fastapi import FastAPI, Request, Response

app = FastAPI()


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next) -> Response:
    # 请求进入时
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    t0 = time.perf_counter()

    # 调用下游路由
    response = await call_next(request)

    # 响应返回时
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response
```

### 在路由中访问中间件注入的状态

```python
from fastapi import Request


@app.post("/predict")
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    request_id = request.state.request_id
    print(f"[{request_id}] predict called")
    ...
```

### CORS

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myapp.example.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## 35.8 WebSocket 与流式推理

WebSocket 允许服务端主动向客户端推送数据，非常适合**流式文本生成**和**实时推理**。

### 35.8.1 基础 WebSocket 路由

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()


@app.websocket("/ws/stream")
async def stream_inference(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 接收客户端请求
            data = await websocket.receive_json()
            features = data.get("features", [])

            # 模拟流式输出（例如 LLM token by token）
            for token in await generate_tokens(features):
                await websocket.send_json({"token": token, "done": False})
                await asyncio.sleep(0)   # 让出执行权，保持响应性

            await websocket.send_json({"token": "", "done": True})

    except WebSocketDisconnect:
        print("客户端断开连接")
```

### 35.8.2 管理多个 WebSocket 连接

```python
from typing import Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

# 活跃连接集合（生产环境应加锁保护）
_active_connections: Set[WebSocket] = set()


@app.websocket("/ws/broadcast")
async def broadcast_endpoint(websocket: WebSocket):
    await websocket.accept()
    _active_connections.add(websocket)
    try:
        while True:
            msg = await websocket.receive_text()
            # 广播给所有连接
            for conn in list(_active_connections):
                try:
                    await conn.send_text(f"broadcast: {msg}")
                except Exception:
                    _active_connections.discard(conn)
    except WebSocketDisconnect:
        _active_connections.discard(websocket)
```

### 35.8.3 流式 HTTP 响应（SSE）

对于只需要服务端推送的场景，Server-Sent Events（SSE）是比 WebSocket 更轻量的选择：

```python
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()


async def token_generator(features: list):
    """模拟 LLM 逐 token 生成"""
    tokens = ["Hello", " world", "!", " How", " are", " you", "?"]
    for token in tokens:
        await asyncio.sleep(0.1)   # 模拟生成延迟
        yield f"data: {token}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/stream")
async def stream_predict(request: PredictRequest) -> StreamingResponse:
    return StreamingResponse(
        token_generator(request.features),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
```

---

## 35.9 限流、超时与背压（asyncio 知识整合）

这是把 Part 9 asyncio 知识与 FastAPI 结合最直接的地方。

### 35.9.1 用 `asyncio.Semaphore` 限制并发推理数

```python
import asyncio
from fastapi import FastAPI, HTTPException

app = FastAPI()

# 最多同时 8 个推理请求（防止显存 OOM）
_inference_sem = asyncio.Semaphore(8)


@app.post("/predict")
async def predict(request: PredictRequest) -> PredictResponse:
    # 尝试获取信号量，如果满了就等待
    acquired = await asyncio.wait_for(
        _inference_sem.acquire(),
        timeout=5.0,   # 超过 5 秒等不到就报 503
    )
    try:
        result = await asyncio.to_thread(_engine.predict, request.features)
        return PredictResponse(predictions=result, ...)
    finally:
        _inference_sem.release()
```

或者用异步上下文管理器写法：

```python
@app.post("/predict")
async def predict(request: PredictRequest) -> PredictResponse:
    try:
        async with asyncio.timeout(5.0):
            async with _inference_sem:
                result = await asyncio.to_thread(_engine.predict, request.features)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Service overloaded")
    return PredictResponse(predictions=result, ...)
```

### 35.9.2 请求级超时

```python
@app.post("/predict")
async def predict(request: PredictRequest) -> PredictResponse:
    try:
        async with asyncio.timeout(2.0):   # 整个推理流程 2 秒内必须完成
            async with _inference_sem:
                result = await asyncio.to_thread(
                    _engine.predict, request.features, request.top_k
                )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Inference timeout after 2.0s",
        )
    return PredictResponse(predictions=result, ...)
```

### 35.9.3 限流（令牌桶）

对于 QPS 限制，可以用 asyncio 实现简单的令牌桶：

```python
import asyncio
import time


class TokenBucket:
    """简单令牌桶限流器"""

    def __init__(self, rate: float, capacity: int):
        self._rate = rate         # 每秒补充的令牌数
        self._capacity = capacity # 桶的最大容量
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, timeout: float = 1.0) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._capacity,
                    self._tokens + elapsed * self._rate,
                )
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
            # 令牌不足，等待
            if asyncio.get_running_loop().time() >= deadline:
                return False
            await asyncio.sleep(1.0 / self._rate)


_rate_limiter = TokenBucket(rate=100, capacity=200)  # 100 QPS


@app.post("/predict")
async def predict(request: PredictRequest) -> PredictResponse:
    if not await _rate_limiter.acquire(timeout=0.1):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    ...
```

---

## 35.10 实战：完整异步推理网关

把前面所有概念整合成一个生产级示例。

### 目录结构

```
inference_gateway/
├── main.py          # FastAPI 应用入口
├── model.py         # 模型推理封装
├── schemas.py       # Pydantic 数据模型
├── deps.py          # 依赖注入
├── middleware.py    # 中间件
└── config.py        # 配置
```

### `schemas.py`

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import math


class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=1, ge=1, le=20)
    timeout_ms: Optional[float] = Field(default=2000.0, gt=0)

    @field_validator("features")
    @classmethod
    def no_nan_inf(cls, v: List[float]) -> List[float]:
        if any(not math.isfinite(x) for x in v):
            raise ValueError("features 中不能包含 NaN 或 Inf")
        return v


class Prediction(BaseModel):
    class_id: int
    probability: float
    label: str


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    model_version: str
    inference_ms: float
    request_id: str
```

### `model.py`

```python
import torch
import torch.nn as nn
from pathlib import Path


class ModelInference:
    """线程安全的同步模型推理封装"""

    VERSION = "1.0.0"
    LABELS = [str(i) for i in range(10)]

    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        model = nn.Linear(784, 10)   # 简化示例
        if Path(weights_path).exists():
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.eval()
        self.model = model.to(self.device)

    @torch.no_grad()
    def predict(self, features: list, top_k: int = 1) -> list:
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        top_probs, top_ids = probs.topk(min(top_k, len(self.LABELS)))
        return [
            {
                "class_id": idx.item(),
                "probability": round(prob.item(), 4),
                "label": self.LABELS[idx.item()],
            }
            for prob, idx in zip(top_probs, top_ids)
        ]
```

### `main.py`

```python
import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model import ModelInference
from schemas import PredictRequest, PredictResponse, Prediction

# ─── 应用级状态 ─────────────────────────────────────
_state: dict = {}
_ready = asyncio.Event()
_inference_sem = asyncio.Semaphore(8)   # 最多 8 个并发推理


# ─── 生命周期 ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] 加载模型...")
    _state["engine"] = ModelInference("weights/model.pt")
    _state["stats"] = {"total": 0, "errors": 0}
    _ready.set()
    print("[startup] 服务就绪")

    yield

    _ready.clear()
    print("[shutdown] 清理完成")


app = FastAPI(title="异步推理网关", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 中间件 ──────────────────────────────────────────
@app.middleware("http")
async def request_tracking(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    t0 = time.perf_counter()

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ─── 依赖 ────────────────────────────────────────────
def get_engine() -> ModelInference:
    if not _ready.is_set():
        raise HTTPException(status_code=503, detail="Service not ready")
    return _state["engine"]


# ─── 后台日志 ────────────────────────────────────────
async def record_inference(request_id: str, success: bool, elapsed_ms: float):
    _state["stats"]["total"] += 1
    if not success:
        _state["stats"]["errors"] += 1
    print(f"[{request_id}] success={success} elapsed={elapsed_ms:.1f}ms")


# ─── 路由 ────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ready" if _ready.is_set() else "starting",
        "stats": _state.get("stats", {}),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request_body: PredictRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    engine: ModelInference = Depends(get_engine),
) -> PredictResponse:
    request_id = http_request.state.request_id
    timeout_s = (request_body.timeout_ms or 2000) / 1000.0
    t0 = time.perf_counter()

    try:
        async with asyncio.timeout(timeout_s):
            async with _inference_sem:
                result = await asyncio.to_thread(
                    engine.predict,
                    request_body.features,
                    request_body.top_k,
                )
    except asyncio.TimeoutError:
        background_tasks.add_task(record_inference, request_id, False, timeout_s * 1000)
        raise HTTPException(
            status_code=504,
            detail=f"Inference timeout after {timeout_s:.1f}s",
        )
    except Exception as exc:
        background_tasks.add_task(record_inference, request_id, False, 0.0)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000
    background_tasks.add_task(record_inference, request_id, True, elapsed_ms)

    return PredictResponse(
        predictions=[Prediction(**p) for p in result],
        model_version=ModelInference.VERSION,
        inference_ms=round(elapsed_ms, 2),
        request_id=request_id,
    )


@app.get("/metrics")
async def metrics():
    return _state.get("stats", {})
```

### 启动方式

```bash
# 开发模式（自动重载）
uvicorn inference_gateway.main:app --reload --host 0.0.0.0 --port 8000

# 生产模式（多 worker，但注意 asyncio 状态是进程级的）
uvicorn inference_gateway.main:app --host 0.0.0.0 --port 8000 --workers 4

# 生产模式（gunicorn + uvicorn workers）
gunicorn inference_gateway.main:app \
    -k uvicorn.workers.UvicornWorker \
    --workers 4 \
    --bind 0.0.0.0:8000
```

**注意**：使用多进程时，`_state`、`_ready`、`_inference_sem` 等对象是**进程级**的，不在进程间共享。  
如果需要跨进程共享状态（如全局并发计数），需要使用 Redis 等外部存储。

---

## 35.11 常见陷阱与调试建议

### 陷阱一：在 `async def` 路由里直接调用同步阻塞函数

```python
# ❌ 错误：阻塞事件循环
@app.post("/predict")
async def predict(request: PredictRequest):
    result = model.predict(request.features)   # 同步，卡死 loop
    return result

# ✅ 正确：放入线程池
@app.post("/predict")
async def predict(request: PredictRequest):
    result = await asyncio.to_thread(model.predict, request.features)
    return result
```

### 陷阱二：在 `lifespan` 之外初始化 asyncio 对象

```python
# ❌ 错误：模块加载时创建 Semaphore，此时事件循环可能还未启动
_sem = asyncio.Semaphore(8)   # 在某些 Python 版本中会有警告或错误

# ✅ 正确：在 lifespan 或第一次请求时延迟创建
_sem: asyncio.Semaphore | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _sem
    _sem = asyncio.Semaphore(8)
    yield
```

### 陷阱三：`BackgroundTasks` 里的异常被吞掉

```python
# ❌ 后台任务里的异常不会出现在响应中，容易被忽略
async def risky_task():
    raise RuntimeError("这个错误会被静默吞掉")

# ✅ 在后台任务里自行捕获并记录
async def safe_task():
    try:
        await risky_operation()
    except Exception as e:
        logger.error(f"background task failed: {e}")
```

### 调试建议

```python
# 开启 asyncio debug 模式（检测阻塞调用）
import asyncio
asyncio.get_event_loop().set_debug(True)

# 或通过环境变量
# PYTHONASYNCIODEBUG=1 uvicorn main:app
```

---

## 本章小结

| 概念 | 在 FastAPI 中的用法 |
|------|-------------------|
| `async def` 路由 | 在 asyncio 事件循环上运行，不能调用同步阻塞函数 |
| `asyncio.to_thread()` | 把同步推理放进线程池，不阻塞事件循环 |
| `Depends` | 注入共享资源（模型、Redis、DB），支持 `async def` 依赖 |
| `lifespan` | 启动/关闭时初始化/清理应用级资源 |
| `asyncio.Event` | 作为服务就绪屏障，阻止早于模型加载的请求 |
| `asyncio.Semaphore` | 限制并发推理数，防止显存 OOM |
| `asyncio.timeout()` | 为整个推理流程设置 deadline |
| `BackgroundTasks` | 响应发出后执行日志、通知等非关键任务 |
| `asyncio.create_task()` | 应用级长期后台任务（缓存刷新、指标上报） |
| WebSocket | 流式推理、实时交互 |
| `StreamingResponse` | SSE 流式文本生成 |
| 中间件 | 请求 ID 注入、计时、CORS |

---

## 35.12 结构化 SSE 事件系统：多类型 Event + asyncio.Queue

### 35.12.1 为什么需要结构化事件

基础 SSE 只发送纯文本流，前端无法区分"这条消息是 token、还是工具调用状态、还是错误"。  
**结构化事件系统**在 SSE 的 `event:` 字段中携带事件类型，`data:` 字段携带 JSON payload，让前端根据类型做差异化渲染：

```
event: status
data: {"seq":1,"session_id":"abc","stage":"loading_model","pct":20}

event: token
data: {"seq":2,"session_id":"abc","text":"Hello","finish":false}

event: tool_call
data: {"seq":3,"session_id":"abc","tool":"web_search","args":{"q":"asyncio"}}

event: error
data: {"seq":4,"session_id":"abc","code":"TIMEOUT","message":"推理超时"}

event: done
data: {"seq":5,"session_id":"abc","total_tokens":128}
```

`seq` 字段保证同 session 内事件有序，即使网络重排序也可在前端按序拼接。

---

### 35.12.2 SSE 协议速查

SSE 是纯文本协议，每条消息以空行结束：

| 字段 | 作用 | 示例 |
|------|------|------|
| `event:` | 事件类型（可选，默认 `message`） | `event: token` |
| `data:` | 消息体，可多行 | `data: {"text":"Hi"}` |
| `id:` | 消息 ID，断线重连时 `Last-Event-ID` 会带上 | `id: 42` |
| `retry:` | 客户端重连间隔(ms) | `retry: 3000` |

```
event: token\ndata: {"text":"Hi"}\nid: 42\n\n
```

---

### 35.12.3 Pydantic 事件类型定义

```python
# events.py
from __future__ import annotations
import time
from typing import Any, Literal
from pydantic import BaseModel, Field


class SseEvent(BaseModel):
    """所有 SSE 事件的基类"""
    seq: int = 0                   # 同 session 内的单调递增序号，由 EventBus 填充
    session_id: str = ""
    ts: float = Field(default_factory=time.time)


class StatusEvent(SseEvent):
    event_type: Literal["status"] = "status"
    stage: str                     # "queued" | "loading_model" | "generating"
    pct: int = 0                   # 进度 0-100


class TokenEvent(SseEvent):
    event_type: Literal["token"] = "token"
    text: str
    finish: bool = False           # True 表示这是最后一个 token


class ToolCallEvent(SseEvent):
    event_type: Literal["tool_call"] = "tool_call"
    tool: str                      # 工具名
    args: dict[str, Any] = {}
    result: str | None = None      # None 表示调用中，非 None 表示完成


class ErrorEvent(SseEvent):
    event_type: Literal["error"] = "error"
    code: str                      # "TIMEOUT" | "OOM" | "INVALID_INPUT" | ...
    message: str


class DoneEvent(SseEvent):
    event_type: Literal["done"] = "done"
    total_tokens: int = 0
    elapsed_ms: float = 0.0


# 联合类型，方便类型标注
AnyEvent = StatusEvent | TokenEvent | ToolCallEvent | ErrorEvent | DoneEvent
```

---

### 35.12.4 EventBus：session 级别的有序事件总线

```python
# event_bus.py
import asyncio
from collections import defaultdict
from events import AnyEvent, SseEvent


class EventBus:
    """
    维护 session_id → asyncio.Queue[AnyEvent] 的映射。
    - 每个 session 拥有独立队列，天然 FIFO，保证同 session 内有序。
    - seq 由 EventBus 统一分配，producer 无需关心。
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._queues: dict[str, asyncio.Queue[AnyEvent | None]] = {}
        self._seq:    dict[str, int] = defaultdict(int)
        self._maxsize = maxsize

    # ---------- session 生命周期 ----------

    def open(self, session_id: str) -> asyncio.Queue[AnyEvent | None]:
        """创建 session 队列（幂等）。"""
        if session_id not in self._queues:
            self._queues[session_id] = asyncio.Queue(maxsize=self._maxsize)
            self._seq[session_id] = 0
        return self._queues[session_id]

    def close(self, session_id: str) -> None:
        """销毁 session 队列，放入哨兵 None 唤醒挂起的 consumer。"""
        q = self._queues.pop(session_id, None)
        if q is not None:
            try:
                q.put_nowait(None)          # 哨兵：通知 consumer 停止
            except asyncio.QueueFull:
                pass
        self._seq.pop(session_id, None)

    # ---------- 生产侧 ----------

    async def publish(self, event: AnyEvent) -> None:
        """
        向指定 session 队列投递事件，自动填充 seq。
        若队列不存在（session 已关闭），静默丢弃。
        """
        sid = event.session_id
        q = self._queues.get(sid)
        if q is None:
            return
        self._seq[sid] += 1
        event.seq = self._seq[sid]
        await q.put(event)

    def publish_nowait(self, event: AnyEvent) -> None:
        """非阻塞版本，队列满时抛出 QueueFull。"""
        sid = event.session_id
        q = self._queues.get(sid)
        if q is None:
            return
        self._seq[sid] += 1
        event.seq = self._seq[sid]
        q.put_nowait(event)

    # ---------- 消费侧 ----------

    async def consume(self, session_id: str):
        """
        异步生成器：逐条 yield 事件，遇到哨兵 None 时停止。
        调用方负责在适当时机调用 close()。
        """
        q = self._queues.get(session_id)
        if q is None:
            return
        while True:
            event = await q.get()
            if event is None:               # 哨兵：session 已关闭
                break
            yield event
            q.task_done()
```

**关键设计点**：

| 设计 | 原因 |
|------|------|
| 每 session 独立 Queue | FIFO 保证有序；不同 session 互不干扰 |
| seq 由 EventBus 统一填充 | producer 只需关心业务语义，不处理计数 |
| 哨兵 `None` | 优雅唤醒阻塞在 `q.get()` 的 consumer |
| `maxsize=256` | 防止慢客户端撑爆内存；producer 会在队列满时 await 背压 |

---

### 35.12.5 Producer：推理流程向队列投递事件

```python
# producer.py
import asyncio
from events import StatusEvent, TokenEvent, ToolCallEvent, DoneEvent, ErrorEvent
from event_bus import EventBus


async def run_inference(
    session_id: str,
    prompt: str,
    bus: EventBus,
) -> None:
    """
    模拟推理流程：
      1. 发布 status(queued)
      2. 模拟加载模型 → status(loading_model, pct=50)
      3. 模拟工具调用
      4. 逐 token 流式生成
      5. 发布 done
    任何异常发布 ErrorEvent 后重新抛出（让上层决定是否重试）。
    """
    start = asyncio.get_event_loop().time()
    try:
        # ① 进入队列
        await bus.publish(StatusEvent(session_id=session_id, stage="queued", pct=0))

        # ② 加载模型（模拟 IO，实际用 to_thread）
        await asyncio.sleep(0.2)
        await bus.publish(StatusEvent(session_id=session_id, stage="loading_model", pct=50))

        # ③ 工具调用（可选阶段）
        await bus.publish(ToolCallEvent(
            session_id=session_id, tool="web_search",
            args={"q": prompt[:20]}
        ))
        await asyncio.sleep(0.1)
        await bus.publish(ToolCallEvent(
            session_id=session_id, tool="web_search",
            args={"q": prompt[:20]},
            result="找到 3 条相关结果"
        ))

        # ④ 流式生成 token
        await bus.publish(StatusEvent(session_id=session_id, stage="generating", pct=80))
        tokens = list(f"回答：{prompt} 的相关内容...")
        for i, ch in enumerate(tokens):
            await asyncio.sleep(0.03)       # 模拟每 token 30ms
            await bus.publish(TokenEvent(
                session_id=session_id,
                text=ch,
                finish=(i == len(tokens) - 1),
            ))

        # ⑤ 完成
        elapsed = (asyncio.get_event_loop().time() - start) * 1000
        await bus.publish(DoneEvent(
            session_id=session_id,
            total_tokens=len(tokens),
            elapsed_ms=elapsed,
        ))

    except asyncio.CancelledError:
        await bus.publish(ErrorEvent(
            session_id=session_id, code="CANCELLED", message="推理被取消"
        ))
        raise
    except Exception as exc:
        await bus.publish(ErrorEvent(
            session_id=session_id, code="INTERNAL", message=str(exc)
        ))
        raise
```

---

### 35.12.6 Consumer：将队列事件格式化为 SSE 流

```python
# sse_format.py
import json
from events import AnyEvent


def to_sse_message(event: AnyEvent) -> str:
    """
    将 Pydantic 事件对象序列化为 SSE 文本格式：
      event: <event_type>\n
      data: <json>\n
      id: <seq>\n
      \n
    """
    payload = event.model_dump()
    event_type = payload.pop("event_type", "message")
    lines = [
        f"event: {event_type}",
        f"data: {json.dumps(payload, ensure_ascii=False)}",
        f"id: {event.seq}",
        "",          # 消息结束的空行
        "",
    ]
    return "\n".join(lines)
```

---

### 35.12.7 FastAPI 路由：完整集成

```python
# main.py
from __future__ import annotations
import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from event_bus import EventBus
from producer import run_inference
from sse_format import to_sse_message


# ---------- 应用生命周期 ----------

bus = EventBus(maxsize=512)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # 应用关闭时清理所有仍在运行的 session（实际生产应维护 session 注册表）


app = FastAPI(lifespan=lifespan)


# ---------- 请求模型 ----------

class InferRequest(BaseModel):
    prompt: str
    session_id: str | None = None   # 不传则自动生成


# ---------- SSE 端点 ----------

@app.post("/infer/stream")
async def infer_stream(req: InferRequest, request: Request):
    """
    POST /infer/stream
    返回 SSE 流，每条消息携带 event_type 字段，前端可按类型渲染。

    流程：
      1. 分配 session_id，开启 EventBus session
      2. 以 create_task 启动 producer（非阻塞）
      3. StreamingResponse 拉取 consumer 生成器
      4. 客户端断开时取消 producer，关闭 session
    """
    session_id = req.session_id or str(uuid.uuid4())
    bus.open(session_id)

    # 启动 producer 为后台 Task，与 SSE generator 并发运行
    producer_task = asyncio.create_task(
        run_inference(session_id, req.prompt, bus),
        name=f"producer-{session_id}",
    )

    async def sse_generator():
        try:
            # 先发送 retry 指令，告诉客户端断线后 2s 重连
            yield "retry: 2000\n\n"

            async for event in bus.consume(session_id):
                yield to_sse_message(event)

                # DoneEvent 或 ErrorEvent 后主动结束流
                if event.event_type in ("done", "error"):       # type: ignore[union-attr]
                    break

        except asyncio.CancelledError:
            pass
        finally:
            # 客户端断开 → 取消 producer，释放资源
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except (asyncio.CancelledError, Exception):
                    pass
            bus.close(session_id)

    # producer_task 完成后若 consumer 还在等待，用 close() 发哨兵唤醒它
    def _on_producer_done(fut: asyncio.Future) -> None:
        if fut.cancelled() or fut.exception():
            bus.close(session_id)   # 异常时关闭，触发 consumer 退出

    producer_task.add_done_callback(_on_producer_done)

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",      # 关闭 Nginx 缓冲
            "X-Session-Id": session_id,
        },
    )


# ---------- 查询 session 进度（可选辅助接口）----------

@app.get("/infer/session/{session_id}")
async def session_status(session_id: str):
    """返回 session 是否仍在运行（简单示例）。"""
    active = session_id in bus._queues
    return {"session_id": session_id, "active": active}
```

**并发流程图**：

```
POST /infer/stream
       │
       ├─ bus.open(session_id)
       │
       ├─ create_task(run_inference)   ← Producer Task（事件循环异步运行）
       │        │
       │        ├─ publish(StatusEvent)  ─┐
       │        ├─ publish(TokenEvent)   ─┤─→ Queue[session_id] ──→ SSE Generator
       │        ├─ publish(DoneEvent)    ─┘         ↑
       │                                      await q.get()
       │
       └─ StreamingResponse(sse_generator)   ← Consumer（与 Producer 并发）
```

---

### 35.12.8 前端 JavaScript：按事件类型差异化渲染

```html
<!DOCTYPE html>
<html lang="zh">
<head><meta charset="UTF-8"><title>SSE 结构化事件演示</title></head>
<body>
<div id="status-bar">状态：等待中</div>
<div id="tool-info"></div>
<div id="output"></div>
<div id="error-msg" style="color:red"></div>
<script>
async function startInference(prompt) {
  // 1. 发送请求，获取 session_id
  const res = await fetch("/infer/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });

  const sessionId = res.headers.get("X-Session-Id");
  console.log("session:", sessionId);

  // 2. 用 EventSource 消费 SSE
  //    注意：EventSource 只支持 GET；若需 POST 可用 fetch + ReadableStream
  //    此处演示使用 fetch 手动读取流（支持 POST）
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // 按空行分割 SSE 消息
    const messages = buffer.split("\n\n");
    buffer = messages.pop() ?? "";   // 最后一段可能不完整，留到下次

    for (const msg of messages) {
      if (!msg.trim()) continue;
      handleSseMessage(parseSSE(msg));
    }
  }
}

// 解析单条 SSE 文本为 { eventType, data, id }
function parseSSE(raw) {
  const lines = raw.split("\n");
  let eventType = "message", dataLines = [], id = null;
  for (const line of lines) {
    if (line.startsWith("event: "))      eventType = line.slice(7).trim();
    else if (line.startsWith("data: "))  dataLines.push(line.slice(6));
    else if (line.startsWith("id: "))    id = line.slice(4).trim();
  }
  return { eventType, data: JSON.parse(dataLines.join("\n") || "{}"), id };
}

// 按 eventType 差异化渲染
function handleSseMessage({ eventType, data }) {
  switch (eventType) {
    case "status":
      document.getElementById("status-bar").textContent =
        `状态：${data.stage}（${data.pct}%）`;
      break;

    case "token":
      document.getElementById("output").textContent += data.text;
      if (data.finish) {
        document.getElementById("status-bar").textContent = "状态：生成完毕";
      }
      break;

    case "tool_call":
      const toolInfo = document.getElementById("tool-info");
      if (data.result === null || data.result === undefined) {
        toolInfo.textContent = `🔧 调用工具：${data.tool}（参数：${JSON.stringify(data.args)}）`;
      } else {
        toolInfo.textContent = `✅ 工具结果：${data.result}`;
      }
      break;

    case "error":
      document.getElementById("error-msg").textContent =
        `❌ 错误 [${data.code}]：${data.message}`;
      break;

    case "done":
      console.log(`完成：${data.total_tokens} tokens，耗时 ${data.elapsed_ms.toFixed(0)}ms`);
      break;
  }
}

// 启动
startInference("asyncio 中 Queue 如何保证消息有序？");
</script>
</body>
</html>
```

---

### 35.12.9 保证 Session 内有序的机制总结

```
Producer Task                    Queue (FIFO)               Consumer（SSE Generator）
─────────────────                ─────────────              ───────────────────────────
publish(seq=1, status)  ──put──▶  [seq=1]  ──get──▶  yield SSE(event:status, id:1)
publish(seq=2, token)   ──put──▶  [seq=2]  ──get──▶  yield SSE(event:token,  id:2)
publish(seq=3, token)   ──put──▶  [seq=3]  ──get──▶  yield SSE(event:token,  id:3)
publish(seq=4, done)    ──put──▶  [seq=4]  ──get──▶  yield SSE(event:done,   id:4)
bus.close() → None      ──put──▶  [None]   ──get──▶  break（停止生成）
```

**三层有序保证**：

1. **asyncio.Queue 的 FIFO 语义**：同一个 Queue 的 `put` / `get` 严格先进先出，Producer 的投递顺序 = Consumer 的消费顺序。
2. **单 Producer 单 Consumer 模型**：每个 session 只有一个 producer task 和一个 consumer generator，不存在竞争写入。
3. **SSE `id:` 字段 + `seq`**：即使 TCP 出现乱序（实际不会，但防御性设计），前端可按 `seq` 重排；客户端断线重连时，浏览器会携带 `Last-Event-ID`，服务端可从断点续传。

| 风险 | 应对 |
|------|------|
| Producer 比 Consumer 快，Queue 堆积 | `maxsize` + 背压：`await q.put()` 会阻塞 Producer |
| Consumer（客户端）断开 | `finally` 块取消 Producer task，`bus.close()` 释放 Queue |
| Producer 异常 | 发布 `ErrorEvent`，`_on_producer_done` 回调关闭 session |
| 服务重启，session 丢失 | 生产环境可将已发送的 `seq` 持久化到 Redis，断线重连时补发 |

---

### 35.12.10 生产级扩展：多 Worker 进程下的跨进程事件总线

单进程内 `asyncio.Queue` 足够；多 Worker（uvicorn `--workers N`）时，Queue 不跨进程共享，需替换为：

```python
# 方案一：Redis Streams（推荐）
import redis.asyncio as aioredis

class RedisEventBus:
    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url)

    async def publish(self, event: AnyEvent) -> None:
        stream_key = f"sse:{event.session_id}"
        await self.redis.xadd(stream_key, {
            "event_type": event.event_type,          # type: ignore
            "payload": event.model_dump_json(),
        })

    async def consume(self, session_id: str):
        stream_key = f"sse:{session_id}"
        last_id = "0"
        while True:
            results = await self.redis.xread({stream_key: last_id}, block=5000, count=10)
            for _, messages in results:
                for msg_id, fields in messages:
                    last_id = msg_id
                    yield fields          # 上层负责反序列化
            # 若无数据且 session 已关闭，退出
            ttl = await self.redis.ttl(stream_key)
            if ttl == -2:               # key 不存在
                break
```

> **学习重点**：多进程场景将 `asyncio.Queue` 替换为 Redis Streams，接口契约（publish/consume）保持不变，业务代码零修改。

---

| 组件 | 职责 | asyncio 原语 |
|------|------|-------------|
| `EventBus` | session 隔离、seq 分配、生命周期管理 | `asyncio.Queue` |
| Producer Task | 推理流程 → 事件投递 | `asyncio.create_task()` |
| Consumer Generator | Queue → SSE 格式化 → HTTP 流 | `async for` + `await q.get()` |
| `StreamingResponse` | HTTP 长连接 + 流式发送 | FastAPI 内置 |
| `add_done_callback` | Producer 异常时通知 Consumer | `Task.add_done_callback()` |

---

## 深度学习应用

本章知识在以下场景直接落地：

- **LLM 推理服务**：WebSocket 流式 token 输出 + Semaphore 限并发
- **特征聚合网关**：`asyncio.gather()` 并发抓取多路特征 + timeout 防超时
- **批量推理队列**：`asyncio.Queue` + 后台 worker + BackgroundTasks 汇报状态
- **模型热更新**：lifespan + Event 协调"旧模型继续服务 → 新模型就绪 → 切换"

---

## 练习题

1. 解释为什么在 `async def` 路由里直接调用 `model.predict()`（同步函数）会卡死整个服务，而 `await asyncio.to_thread(model.predict, ...)` 不会。
2. `Depends` 的 `yield` 依赖和普通 `return` 依赖有什么区别？分别在什么场景下使用？
3. `BackgroundTasks.add_task()` 和 `asyncio.create_task()` 各自的任务生命周期是什么？何时应该用哪个？
4. 用 `asyncio.Semaphore(4)` 和 `asyncio.timeout(2.0)` 实现一个最多 4 个并发、每个请求 2 秒超时的推理接口，超时时返回 504，并发满时等待最多 1 秒后返回 503。
5. 设计一个 WebSocket 接口，接收用户输入的文本，逐字符流式返回"翻转后的字符串"（每隔 50ms 发一个字符），并在客户端断开时正确清理资源。
