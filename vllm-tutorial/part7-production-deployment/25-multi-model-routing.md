# 第25章：多模型服务与路由

> 生产环境中很少只部署一个模型。你可能需要同时服务通用对话、代码生成、翻译等不同模型，这就需要一个智能的路由层。

---

## 学习目标

学完本章，你将能够：

1. 设计多模型服务架构
2. 实现基于规则的请求路由
3. 支持 A/B 测试和灰度发布
4. 管理模型版本和生命周期
5. 使用 API 网关整合多个 vLLM 实例

---

## 25.1 多模型服务架构

### 常见架构

```
                    ┌──────────────┐
                    │  API Gateway │
                    │   / Router   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ vLLM #1  │ │ vLLM #2  │ │ vLLM #3  │
        │ Chat 7B  │ │ Code 34B │ │ VLM 7B   │
        │ (2 GPU)  │ │ (4 GPU)  │ │ (1 GPU)  │
        └──────────┘ └──────────┘ └──────────┘
```

### 路由策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| 基于模型名 | 请求中指定 model 字段 | 通用 |
| 基于内容 | 分析请求内容决定路由 | 自动选模型 |
| 基于用户 | 不同用户/租户路由到不同模型 | 多租户 |
| 基于负载 | 路由到最空闲的副本 | 负载均衡 |
| A/B 测试 | 按比例分流到不同模型版本 | 版本评估 |

---

## 25.2 简单路由实现

### 使用 Nginx

```nginx
upstream chat_model {
    server vllm-chat:8000;
}

upstream code_model {
    server vllm-code:8000;
}

server {
    listen 80;

    location /v1/chat/ {
        proxy_pass http://chat_model;
    }

    location /v1/code/ {
        proxy_pass http://code_model;
    }
}
```

### 使用 Python 路由

```python
from fastapi import FastAPI, Request
import httpx

app = FastAPI()

MODEL_ROUTES = {
    "chat-model": "http://vllm-chat:8000",
    "code-model": "http://vllm-code:8000",
    "vision-model": "http://vllm-vision:8000",
}

@app.post("/v1/chat/completions")
async def route_request(request: Request):
    body = await request.json()
    model = body.get("model", "chat-model")

    backend = MODEL_ROUTES.get(model)
    if not backend:
        return {"error": f"Unknown model: {model}"}

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{backend}/v1/chat/completions",
            json=body,
            timeout=120,
        )
        return resp.json()
```

---

## 25.3 A/B 测试

### 按比例分流

```python
import random

@app.post("/v1/chat/completions")
async def ab_test(request: Request):
    body = await request.json()

    # 90% 流量给 v1，10% 给 v2
    if random.random() < 0.1:
        backend = "http://vllm-v2:8000"
        version = "v2"
    else:
        backend = "http://vllm-v1:8000"
        version = "v1"

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{backend}/v1/chat/completions", json=body, timeout=120,
        )
        result = resp.json()
        # 记录版本信息，用于后续分析
        result["_ab_version"] = version
        return result
```

---

## 25.4 模型版本管理

### 无缝切换

```bash
# 1. 在新端口启动新版本模型
vllm serve new-model --port 8001

# 2. 健康检查通过后切换路由
# 3. 等旧版本请求排空后关闭旧服务

# 使用 --served-model-name 保持 API 兼容
vllm serve new-model-v2 --served-model-name my-model --port 8001
```

### 多版本并存

```python
MODEL_VERSIONS = {
    "my-model:v1": "http://vllm-v1:8000",
    "my-model:v2": "http://vllm-v2:8000",
    "my-model:latest": "http://vllm-v2:8000",  # latest 指向 v2
}
```

---

## 25.5 负载均衡

### 多副本同模型

```python
import itertools

REPLICAS = itertools.cycle([
    "http://vllm-replica-1:8000",
    "http://vllm-replica-2:8000",
    "http://vllm-replica-3:8000",
])

@app.post("/v1/chat/completions")
async def load_balance(request: Request):
    backend = next(REPLICAS)  # 简单轮询
    body = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{backend}/v1/chat/completions", json=body, timeout=120,
        )
        return resp.json()
```

### 基于队列深度的路由

```python
async def get_least_loaded():
    """选择 waiting 请求最少的副本"""
    min_waiting = float('inf')
    best = None
    async with httpx.AsyncClient() as client:
        for replica in REPLICAS_LIST:
            metrics = await client.get(f"{replica}/metrics")
            waiting = parse_waiting_count(metrics.text)
            if waiting < min_waiting:
                min_waiting = waiting
                best = replica
    return best
```

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 架构 | API Gateway + 多个 vLLM 实例 |
| 路由 | 基于模型名、内容、用户或负载 |
| A/B 测试 | 按比例分流到不同模型版本 |
| 版本管理 | `--served-model-name` 保持 API 兼容 |
| 负载均衡 | 轮询或基于队列深度 |

---

## 练习题

### 实践题

1. 部署两个不同的 vLLM 实例，实现一个简单的路由层。

### 思考题

2. 在高可用场景下，如果一个 vLLM 实例崩溃，路由层应该怎么处理？
3. 基于队列深度的路由比轮询好在哪里？有什么额外开销？
