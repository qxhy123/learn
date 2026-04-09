# 第24章：模型部署与优化

> 训练好的模型只有部署到生产环境，才能真正创造价值。本章介绍将深度学习模型从研究原型转化为生产服务的完整流程。

---

## 学习目标

完成本章学习后，你将能够：

1. 使用 TorchScript 和 ONNX 导出模型，使其脱离 Python 环境独立运行
2. 通过动态量化和静态量化压缩模型体积、提升推理速度
3. 理解模型剪枝的基本原理，能够使用 PyTorch 内置工具完成结构化剪枝
4. 利用 `torch.compile` 和混合精度推理对模型推理过程进行加速
5. 使用 FastAPI 将训练好的模型包装为 REST API 服务，了解 TorchServe 的基本工作方式

---

## 24.1 模型导出（TorchScript、ONNX）

训练完成的 PyTorch 模型默认以 Python 对象的形式存在，无法直接运行在非 Python 环境（如 C++ 服务、移动端、嵌入式设备）中。**模型导出**将模型序列化为与运行时环境无关的格式，是部署的第一步。

### 24.1.1 TorchScript 导出

TorchScript 是 PyTorch 提供的中间表示（IR），可通过两种方式生成：

- **Tracing**：给定示例输入，记录模型的执行路径。简单，但无法捕捉含有数据依赖分支的控制流。
- **Scripting**：静态分析模型源代码，完整保留控制流。适合含 if/for 的复杂模型。

```python
import torch
import torch.nn as nn

# 定义一个简单的 MLP 模型
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleMLP(784, 256, 10)
model.eval()

# 方式一：Tracing
example_input = torch.randn(1, 784)
traced_model = torch.jit.trace(model, example_input)

# 保存与加载
traced_model.save("model_traced.pt")
loaded_model = torch.jit.load("model_traced.pt")

# 方式二：Scripting（推荐用于含控制流的模型）
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# 验证输出一致性
with torch.no_grad():
    x = torch.randn(4, 784)
    out_original = model(x)
    out_scripted = scripted_model(x)
    print(torch.allclose(out_original, out_scripted, atol=1e-6))
    # 输出: True
```

> **注意**：Tracing 遇到含 `if tensor.shape[0] > 1` 这类数据依赖分支时会静默地固化分支，导致错误结果。遇到此类模型请使用 Scripting。

### 24.1.2 ONNX 导出

*Open Neural Network Exchange*（ONNX）是跨框架的开放模型格式，支持 TensorRT、OpenVINO、ONNX Runtime 等推理引擎。

```python
import torch
import torch.onnx

model = SimpleMLP(784, 256, 10)
model.eval()

dummy_input = torch.randn(1, 784)

# 导出为 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,          # 导出权重
    opset_version=17,            # ONNX 算子集版本
    do_constant_folding=True,    # 常量折叠优化
    input_names=["input"],       # 输入张量名称
    output_names=["output"],     # 输出张量名称
    dynamic_axes={               # 动态 batch size
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)
print("ONNX 模型已导出：model.onnx")
```

使用 ONNX Runtime 进行推理：

```python
# pip install onnxruntime
import onnxruntime as ort
import numpy as np

# 创建推理会话（自动选择最优后端）
session = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# 查看输入输出信息
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"输入名: {input_name}, 输出名: {output_name}")

# 推理
x_np = np.random.randn(4, 784).astype(np.float32)
result = session.run([output_name], {input_name: x_np})
print(f"推理结果 shape: {result[0].shape}")
# 输出: 推理结果 shape: (4, 10)
```

### 24.1.3 TorchScript vs ONNX 对比

| 对比项 | TorchScript | ONNX |
|--------|-------------|------|
| 生态 | PyTorch 原生 | 跨框架通用 |
| 推理引擎 | libtorch、C++ API | ONNX Runtime、TensorRT 等 |
| 控制流支持 | 完整支持 | 受限（Loop/If 算子） |
| 部署目标 | C++ 服务、移动端 | 跨平台、边缘设备 |
| 调试难度 | 较低 | 较高 |

---

## 24.2 模型量化（动态量化、静态量化）

**量化**将模型权重和激活值从 32 位浮点（FP32）压缩为 8 位整数（INT8），通常可将模型体积减小 75%、推理速度提升 2-4 倍，精度损失极小。

### 24.2.1 量化基础概念

```
FP32 权重 [-0.85, 0.12, 1.34, ...]
           ↓  线性映射到 INT8 范围 [-128, 127]
INT8 权重 [-81,  11, 127, ...]

反量化：INT8 × scale + zero_point = 近似 FP32
```

PyTorch 提供三种量化模式：

| 模式 | 量化时机 | 需要校准数据 | 适用场景 |
|------|----------|-------------|----------|
| 动态量化 | 推理时动态计算 | 否 | LSTM、Transformer |
| 静态量化 | 导出前离线计算 | 是 | CNN 等卷积网络 |
| 量化感知训练（QAT） | 训练期间模拟 | 否（用训练数据） | 高精度要求 |

### 24.2.2 动态量化

最简单的量化方式，无需修改模型结构，不需要校准数据集：

```python
import torch
import torch.nn as nn
import torch.quantization

# 原始 FP32 模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # 取最后时间步

model = LSTMClassifier(128, 256, 10)
model.eval()

# 动态量化：只量化 Linear 和 LSTM 层的权重
quantized_model = torch.quantization.quantize_dynamic(
    model,
    qconfig_spec={nn.Linear, nn.LSTM},  # 指定要量化的层类型
    dtype=torch.qint8
)

# 对比模型大小
def model_size_mb(model):
    """计算模型参数占用的内存（MB）"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024

print(f"原始模型大小: {model_size_mb(model):.2f} MB")
print(f"量化后大小:   {model_size_mb(quantized_model):.2f} MB")

# 验证输出
x = torch.randn(2, 10, 128)
with torch.no_grad():
    out_fp32 = model(x)
    out_int8 = quantized_model(x)
    print(f"最大误差: {(out_fp32 - out_int8).abs().max().item():.6f}")
```

### 24.2.3 静态量化

静态量化需要"校准"步骤，收集真实数据的激活值分布，精度更高：

```python
import torch
import torch.nn as nn
import torch.quantization

class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub / DeQuantStub 标记量化边界
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.quant(x)          # 输入量化
        x = self.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        x = self.dequant(x)        # 输出反量化
        return x

model = SimpleConvNet()
model.eval()

# 第一步：指定量化配置（fbgemm 适用于 x86 服务器，qnnpack 适用于 ARM 移动端）
model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

# 第二步：插入量化观察器（Observer）
torch.quantization.prepare(model, inplace=True)

# 第三步：用代表性数据进行校准（通常使用 100-1000 个样本）
calibration_data = [torch.randn(8, 1, 28, 28) for _ in range(50)]
with torch.no_grad():
    for batch in calibration_data:
        model(batch)
print("校准完成，激活值范围已统计")

# 第四步：将模型转换为量化版本
torch.quantization.convert(model, inplace=True)
print(model.conv)  # 输出: QuantizedConv2d(...)

# 保存量化模型
scripted_quantized = torch.jit.script(model)
scripted_quantized.save("model_quantized_static.pt")
```

---

## 24.3 模型剪枝基础

**剪枝**通过删除神经网络中冗余的权重或神经元，减小模型大小并提高推理速度。PyTorch 通过 `torch.nn.utils.prune` 模块提供剪枝支持。

### 24.3.1 非结构化剪枝

将权重矩阵中绝对值最小的若干元素置零（稀疏化），不改变矩阵形状：

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

model = SimpleMLP(784, 256, 10)

# 对第一层 Linear 进行非结构化剪枝（裁剪 30% 的权重）
prune.l1_unstructured(model.net[0], name="weight", amount=0.3)

# 查看剪枝结果
print("剪枝后权重掩码（部分）:")
print(model.net[0].weight_mask[:3, :5])
# 掩码为 0 的位置权重被置零

# 统计稀疏度
def sparsity(module):
    zeros = float(torch.sum(module.weight == 0))
    total = float(module.weight.nelement())
    return zeros / total * 100

print(f"第一层稀疏度: {sparsity(model.net[0]):.1f}%")
# 输出: 第一层稀疏度: 30.0%
```

### 24.3.2 结构化剪枝

移除整个神经元（行/列），可直接减小矩阵维度，硬件推理更高效：

```python
import torch.nn.utils.prune as prune

model = SimpleMLP(784, 256, 10)

# 对 Linear 层按 L2 范数剪枝整行（每个输出神经元），移除 20% 的神经元
prune.ln_structured(
    model.net[0],
    name="weight",
    amount=0.2,
    n=2,        # L2 范数
    dim=0       # 沿输出维度（行）裁剪
)

# 全局剪枝：在整个模型范围内按比例裁剪，自动寻找最不重要的权重
parameters_to_prune = [
    (model.net[0], "weight"),
    (model.net[2], "weight"),
]
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.4  # 全局裁剪 40%
)

# 将剪枝固化（移除掩码，真正删除权重）
for module, name in parameters_to_prune:
    prune.remove(module, name)

print("剪枝已固化，掩码已移除")
```

### 24.3.3 剪枝工作流最佳实践

```
训练完整模型
    ↓
评估原始精度（baseline）
    ↓
逐步增大剪枝比例（10% → 20% → 30% ...）
    ↓
微调（fine-tune）若干 epoch 恢复精度
    ↓
评估精度 + 推理速度，选择满足要求的剪枝率
    ↓
固化剪枝，导出模型
```

> **提示**：单次大比例剪枝（如直接剪掉 80%）会造成精度断崖式下降。推荐使用**迭代剪枝**：多轮少量剪枝 + 微调，逐步逼近目标稀疏度。

---

## 24.4 推理优化（torch.compile、混合精度推理）

### 24.4.1 torch.compile

`torch.compile`（PyTorch 2.0+ 引入）通过将模型编译为优化的低级代码（TorchInductor → Triton/C++），无需修改模型架构即可获得显著加速：

```python
import torch
import time

model = SimpleMLP(784, 256, 10).cuda()
model.eval()

# 编译模型（首次调用会触发编译，有一次性开销）
# mode 可选: "default"（平衡）、"reduce-overhead"（减少调用开销）、"max-autotune"（最大吞吐量）
compiled_model = torch.compile(model, mode="reduce-overhead")

# 预热（让编译器完成 JIT 优化）
x = torch.randn(64, 784).cuda()
for _ in range(5):
    with torch.no_grad():
        _ = compiled_model(x)

# 基准测试
def benchmark(fn, x, n=100):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        with torch.no_grad():
            fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / n * 1000  # ms

t_original = benchmark(model, x)
t_compiled  = benchmark(compiled_model, x)
print(f"原始模型:   {t_original:.3f} ms/batch")
print(f"编译后模型: {t_compiled:.3f} ms/batch")
print(f"加速比: {t_original / t_compiled:.2f}x")
```

> **注意**：`torch.compile` 在 CPU 上的加速效果通常小于 GPU。对于小模型，首次编译的开销可能超过节省的时间，适合长期运行的服务场景。

### 24.4.2 混合精度推理

使用 FP16 或 BF16 代替 FP32 进行推理，内存减半，速度提升约 2 倍（需要支持 Tensor Core 的 GPU）：

```python
import torch
from torch.amp import autocast

model = SimpleMLP(784, 256, 10).cuda()
model.eval()

x = torch.randn(64, 784).cuda()

# FP16 推理
with torch.no_grad():
    with autocast(device_type="cuda", dtype=torch.float16):
        output_fp16 = model(x)
    print(f"FP16 输出类型: {output_fp16.dtype}")  # torch.float16

# BF16 推理（Ampere+ GPU 支持，数值范围更大，训练更稳定）
with torch.no_grad():
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        output_bf16 = model(x)
    print(f"BF16 输出类型: {output_bf16.dtype}")  # torch.bfloat16

# CPU 半精度推理（需转换模型）
model_half = model.half().cpu()
x_half = x.half().cpu()
with torch.no_grad():
    output_cpu_fp16 = model_half(x_half)
```

### 24.4.3 组合优化策略

```python
import torch

def create_optimized_model(model_class, weights_path, device="cuda"):
    """创建生产级优化模型的工厂函数"""
    model = model_class()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    model.to(device)

    # 1. 量化（CPU 部署时）
    if device == "cpu":
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return model

    # 2. GPU 部署：编译 + 半精度
    model = model.half()                           # FP16 权重
    model = torch.compile(model, mode="reduce-overhead")
    return model

# 使用示例
# optimized_model = create_optimized_model(SimpleMLP, "weights.pt", device="cuda")
```

---

## 24.5 模型服务（FastAPI 部署、TorchServe 简介）

### 24.5.1 使用 FastAPI 构建推理服务

FastAPI 是目前最流行的 Python 异步 Web 框架之一，性能优秀，自动生成 API 文档，非常适合快速搭建模型推理服务。

基本目录结构：

```
model_service/
├── app.py              # FastAPI 应用入口
├── model.py            # 模型定义与加载
├── schemas.py          # 请求/响应数据模型
├── requirements.txt
└── weights/
    └── model.pt
```

**schemas.py** — 请求与响应结构：

```python
from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    """推理请求体"""
    features: List[float] = Field(..., description="输入特征向量")
    top_k: int = Field(default=1, ge=1, le=10, description="返回前 k 个预测")

class Prediction(BaseModel):
    class_id: int
    probability: float
    label: str

class PredictResponse(BaseModel):
    """推理响应体"""
    predictions: List[Prediction]
    inference_time_ms: float
```

**model.py** — 模型封装：

```python
import torch
import torch.nn as nn
from pathlib import Path

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 类别标签（以 MNIST 为例）
LABELS = [str(i) for i in range(10)]

class ModelInference:
    """线程安全的模型推理封装"""
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = MLPClassifier()
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device)
        )
        self.model.eval()
        # 量化加速（CPU 场景）
        if device == "cpu":
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, features: list, top_k: int = 1) -> list:
        """
        执行推理并返回 top-k 预测结果
        Args:
            features: 输入特征列表
            top_k: 返回前 k 个最高概率的类别
        Returns:
            List[dict]，每个元素包含 class_id、probability、label
        """
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        top_probs, top_ids = probs.topk(top_k)
        return [
            {
                "class_id": idx.item(),
                "probability": round(prob.item(), 4),
                "label": LABELS[idx.item()],
            }
            for prob, idx in zip(top_probs, top_ids)
        ]
```

**app.py** — FastAPI 应用主体：

```python
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from schemas import PredictRequest, PredictResponse, Prediction
from model import ModelInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局模型实例（利用 lifespan 在启动时加载，避免每次请求重新加载）
inference_engine: ModelInference = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时加载模型，关闭时释放资源"""
    global inference_engine
    logger.info("正在加载模型...")
    inference_engine = ModelInference(
        weights_path="weights/model.pt",
        device="cpu"
    )
    logger.info("模型加载完成，服务就绪")
    yield  # 服务运行期间
    # 关闭时的清理工作
    logger.info("服务关闭，释放资源")

app = FastAPI(
    title="深度学习推理服务",
    description="基于 FastAPI 的 PyTorch 模型推理 API",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
async def health_check():
    """健康检查端点，供负载均衡器/K8s 探针使用"""
    return {"status": "healthy", "model_loaded": inference_engine is not None}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    模型推理端点

    接收特征向量，返回分类预测结果
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="模型尚未加载")

    if len(request.features) != 784:
        raise HTTPException(
            status_code=422,
            detail=f"输入维度错误：期望 784，实际收到 {len(request.features)}"
        )

    t_start = time.perf_counter()
    try:
        raw_preds = inference_engine.predict(request.features, request.top_k)
    except Exception as e:
        logger.error(f"推理失败: {e}")
        raise HTTPException(status_code=500, detail="推理过程发生错误")

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.info(f"推理耗时: {elapsed_ms:.2f} ms")

    return PredictResponse(
        predictions=[Prediction(**p) for p in raw_preds],
        inference_time_ms=round(elapsed_ms, 2),
    )

@app.get("/model/info")
async def model_info():
    """返回模型元信息"""
    return {
        "input_dim": 784,
        "num_classes": 10,
        "quantized": True,
        "framework": "PyTorch",
    }
```

启动服务：

```bash
# 安装依赖
pip install fastapi uvicorn pydantic torch

# 启动（开发模式，自动热重载）
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 生产模式（多工作进程）
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

调用测试：

```bash
# 健康检查
curl http://localhost:8000/health

# 推理请求（生成随机 784 维输入）
python -c "
import json, random
features = [random.random() for _ in range(784)]
payload = {'features': features, 'top_k': 3}
print(json.dumps(payload))
" | curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d @-
```

响应示例：

```json
{
  "predictions": [
    {"class_id": 7, "probability": 0.4231, "label": "7"},
    {"class_id": 2, "probability": 0.2118, "label": "2"},
    {"class_id": 1, "probability": 0.1543, "label": "1"}
  ],
  "inference_time_ms": 2.47
}
```

FastAPI 自动生成的交互式文档可通过 `http://localhost:8000/docs` 访问。

### 24.5.2 TorchServe 简介

*TorchServe* 是 PyTorch 官方的模型服务框架，专为生产环境设计，提供批量推理、版本管理、A/B 测试、监控等企业级功能。

核心概念：

| 概念 | 说明 |
|------|------|
| Model Archive (.mar) | 打包模型权重 + 处理逻辑的部署单元 |
| Handler | 定义预处理、推理、后处理逻辑的 Python 类 |
| Management API | 动态注册/注销/更新模型，无需重启服务 |
| Inference API | 对外提供推理端点，支持批量请求 |

基本使用流程：

```bash
# 安装
pip install torchserve torch-model-archiver torch-workflow-archiver

# 打包模型（生成 .mar 文件）
torch-model-archiver \
  --model-name mnist_classifier \
  --version 1.0 \
  --model-file model.py \
  --serialized-file weights/model.pt \
  --handler image_classifier \
  --export-path model_store/

# 启动 TorchServe
torchserve \
  --start \
  --model-store model_store/ \
  --models mnist=mnist_classifier.mar \
  --ts-config config.properties

# 推理
curl -X POST http://localhost:8080/predictions/mnist \
  -T test_image.png

# 查看已加载模型
curl http://localhost:8081/models

# 停止服务
torchserve --stop
```

自定义 Handler 示例：

```python
# custom_handler.py
import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler

class MNISTHandler(BaseHandler):
    """自定义推理处理器"""

    def preprocess(self, data):
        """将原始请求数据转换为模型输入张量"""
        inputs = []
        for item in data:
            # 假设输入为 JSON 格式的特征列表
            features = item.get("body") or item.get("data")
            if isinstance(features, (bytes, bytearray)):
                import json
                features = json.loads(features.decode("utf-8"))
            tensor = torch.tensor(features["features"], dtype=torch.float32)
            inputs.append(tensor)
        return torch.stack(inputs)  # [batch, 784]

    def inference(self, data):
        """执行模型推理"""
        with torch.no_grad():
            logits = self.model(data)
        return logits

    def postprocess(self, data):
        """将模型输出转换为响应格式"""
        probs = F.softmax(data, dim=-1)
        top_probs, top_ids = probs.topk(3, dim=-1)
        results = []
        for probs_row, ids_row in zip(top_probs, top_ids):
            results.append({
                "predictions": [
                    {"class_id": idx.item(), "probability": round(prob.item(), 4)}
                    for prob, idx in zip(probs_row, ids_row)
                ]
            })
        return results
```

---

## 本章小结

| 技术 | 核心方法 | 主要收益 | 适用场景 |
|------|----------|----------|----------|
| **TorchScript（Tracing）** | `torch.jit.trace` | 脱离 Python 运行 | 无条件分支的简单模型 |
| **TorchScript（Scripting）** | `torch.jit.script` | 完整保留控制流 | 含 if/for 的复杂模型 |
| **ONNX 导出** | `torch.onnx.export` | 跨框架部署 | 多推理引擎支持 |
| **动态量化** | `quantize_dynamic` | 无需校准，开箱即用 | NLP 模型、LSTM |
| **静态量化** | `prepare` + `convert` | 精度更高 | CNN 等视觉模型 |
| **非结构化剪枝** | `l1_unstructured` | 增加稀疏度 | 通用压缩 |
| **结构化剪枝** | `ln_structured` | 实际减小矩阵维度 | 硬件加速推理 |
| **torch.compile** | `torch.compile` | 无需修改模型加速 | PyTorch 2.0+ GPU 场景 |
| **混合精度推理** | `autocast` | 内存减半、速度 2x | Tensor Core GPU |
| **FastAPI 服务** | `uvicorn` + `FastAPI` | 快速构建 REST API | 中小规模推理服务 |
| **TorchServe** | `.mar` + Handler | 企业级模型管理 | 大规模生产部署 |

---

## 深度学习应用：生产环境部署

**应用场景**：将第21章训练的 CIFAR-10 图像分类模型完整地部署为支持批量推理的 REST API 服务，包含模型优化、服务化和性能测试全流程。

```python
"""
production_deployment.py

演示从模型优化到 REST API 服务的完整部署流程。
假设已有一个预训练的 ResNet 分类模型（CIFAR-10，10类）。
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import time
import json
from pathlib import Path
from PIL import Image
import numpy as np


# ─────────────────────────────────────────────
# 第一步：准备并优化模型
# ─────────────────────────────────────────────

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def build_model(num_classes: int = 10) -> nn.Module:
    """构建适配 CIFAR-10 的轻量 ResNet18"""
    model = models.resnet18(weights=None)
    # CIFAR-10 图像为 32×32，修改首层适配小尺寸
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def optimize_for_inference(model: nn.Module, device: str = "cpu") -> nn.Module:
    """
    应用推理优化策略：
    1. 设置 eval 模式，禁用 Dropout 和 BatchNorm 训练行为
    2. 融合 Conv + BN + ReLU（减少内存访问）
    3. 在 GPU 上启用 torch.compile
    """
    model.eval()

    # 融合常见层组合（仅 CPU 支持 eval 模式下的融合）
    if device == "cpu":
        model = torch.quantization.fuse_modules(
            model,
            [["layer1.0.conv1", "layer1.0.bn1", "layer1.0.relu"]],
            inplace=False
        )

    # GPU 路径：编译加速
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        model = torch.compile(model, mode="reduce-overhead")
        print("已启用 torch.compile 加速（GPU）")
    else:
        # CPU 路径：动态量化（ResNet 中 Linear 层收益最大）
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        print("已启用动态量化（CPU）")

    return model


# ─────────────────────────────────────────────
# 第二步：图像预处理流水线
# ─────────────────────────────────────────────

def build_transform():
    """构建推理预处理管道（与训练时保持一致）"""
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])


def preprocess_image(image_path: str, transform) -> torch.Tensor:
    """将图像文件转换为模型输入张量"""
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image)
    return tensor.unsqueeze(0)  # 添加 batch 维度: [1, 3, 32, 32]


def preprocess_batch(image_paths: list, transform) -> torch.Tensor:
    """批量预处理多张图像"""
    tensors = [transform(Image.open(p).convert("RGB")) for p in image_paths]
    return torch.stack(tensors)  # [B, 3, 32, 32]


# ─────────────────────────────────────────────
# 第三步：推理引擎封装
# ─────────────────────────────────────────────

class CIFARInferenceEngine:
    """
    生产级 CIFAR-10 推理引擎
    特性：批量推理、延迟统计、线程安全（无状态）
    """

    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.classes = CIFAR10_CLASSES
        self.transform = build_transform()

        # 构建并加载权重
        model = build_model(num_classes=len(self.classes))
        if Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"已加载权重: {weights_path}")
        else:
            print(f"警告：权重文件 {weights_path} 不存在，使用随机初始化权重")

        # 应用推理优化
        self.model = optimize_for_inference(model, device=device)
        self._latency_log: list = []

    @torch.no_grad()
    def predict(self, image_paths: list, top_k: int = 3) -> list:
        """
        批量推理

        Args:
            image_paths: 图像文件路径列表
            top_k: 每张图返回前 k 个预测

        Returns:
            List[dict]，每个元素包含图像路径、top-k 预测和延迟
        """
        t_start = time.perf_counter()

        # 预处理
        batch = preprocess_batch(image_paths, self.transform).to(self.device)

        # 推理
        logits = self.model(batch)                         # [B, 10]
        probs = torch.softmax(logits, dim=-1)              # [B, 10]
        top_probs, top_ids = probs.topk(top_k, dim=-1)    # [B, k]

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        self._latency_log.append(elapsed_ms)

        results = []
        for i, path in enumerate(image_paths):
            results.append({
                "image": str(path),
                "predictions": [
                    {
                        "rank": rank + 1,
                        "class_id": top_ids[i, rank].item(),
                        "label": self.classes[top_ids[i, rank].item()],
                        "probability": round(top_probs[i, rank].item(), 4),
                    }
                    for rank in range(top_k)
                ],
                "inference_time_ms": round(elapsed_ms, 2),
            })
        return results

    def latency_stats(self) -> dict:
        """返回延迟统计信息"""
        if not self._latency_log:
            return {"message": "暂无数据"}
        log = np.array(self._latency_log)
        return {
            "count": len(log),
            "mean_ms": round(float(np.mean(log)), 2),
            "p50_ms":  round(float(np.percentile(log, 50)), 2),
            "p95_ms":  round(float(np.percentile(log, 95)), 2),
            "p99_ms":  round(float(np.percentile(log, 99)), 2),
            "max_ms":  round(float(np.max(log)), 2),
        }


# ─────────────────────────────────────────────
# 第四步：FastAPI 服务（完整版）
# ─────────────────────────────────────────────

"""
以下代码为完整的 FastAPI 服务实现。
在实际部署中，将此内容保存为 serve.py，然后执行：
    uvicorn serve:app --host 0.0.0.0 --port 8000 --workers 2
"""

FASTAPI_APP_CODE = '''
import io, base64, time
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from PIL import Image
import torch

# ── 数据模型 ──────────────────────────────────────
class ClassPrediction(BaseModel):
    rank: int
    class_id: int
    label: str
    probability: float

class InferenceResponse(BaseModel):
    predictions: List[ClassPrediction]
    inference_time_ms: float
    model_version: str = "1.0.0"

# ── 全局状态 ──────────────────────────────────────
engine: CIFARInferenceEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = CIFARInferenceEngine(
        weights_path="weights/cifar10_resnet18.pt",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"推理引擎就绪，设备: {engine.device}")
    yield
    print("服务关闭")

app = FastAPI(
    title="CIFAR-10 图像分类服务",
    description="基于 ResNet18 的生产级图像分类 API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── 端点定义 ──────────────────────────────────────
@app.get("/health", tags=["运维"])
async def health():
    return {"status": "ok", "device": str(engine.device) if engine else "N/A"}

@app.get("/stats", tags=["运维"])
async def latency_stats():
    """返回延迟百分位统计"""
    return engine.latency_stats()

@app.post("/predict/file", response_model=InferenceResponse, tags=["推理"])
async def predict_file(
    file: UploadFile = File(..., description="上传图像文件（JPEG/PNG）"),
    top_k: int = 3
):
    """通过上传图像文件进行分类推理"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="仅支持图像文件")

    # 将上传文件保存为临时文件
    import tempfile, os
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        results = engine.predict([tmp_path], top_k=top_k)
    finally:
        os.unlink(tmp_path)

    result = results[0]
    return InferenceResponse(
        predictions=[ClassPrediction(**p) for p in result["predictions"]],
        inference_time_ms=result["inference_time_ms"],
    )

@app.post("/predict/base64", response_model=InferenceResponse, tags=["推理"])
async def predict_base64(
    image_b64: str = Field(..., description="Base64 编码的图像"),
    top_k: int = 3
):
    """通过 Base64 编码图像进行推理（适合 JSON API 场景）"""
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Base64 图像解码失败")

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)
        results = engine.predict([tmp.name], top_k=top_k)

    result = results[0]
    return InferenceResponse(
        predictions=[ClassPrediction(**p) for p in result["predictions"]],
        inference_time_ms=result["inference_time_ms"],
    )
'''


# ─────────────────────────────────────────────
# 第五步：性能基准测试
# ─────────────────────────────────────────────

def benchmark_inference(engine: CIFARInferenceEngine, num_requests: int = 200):
    """
    模拟真实负载，测试批量推理性能基准
    """
    import tempfile, os

    # 生成临时测试图像
    test_images = []
    for i in range(8):  # 8 张测试图
        img = Image.fromarray(
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        )
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img.save(tmp.name)
        test_images.append(tmp.name)

    print(f"\n性能基准测试（{num_requests} 次推理请求，batch_size=8）")
    print("─" * 50)

    latencies = []
    for i in range(num_requests):
        t = time.perf_counter()
        engine.predict(test_images, top_k=3)
        latencies.append((time.perf_counter() - t) * 1000)

    latencies = np.array(latencies)
    throughput = num_requests * 8 / (latencies.sum() / 1000)  # 图像/秒

    print(f"平均延迟:    {np.mean(latencies):.2f} ms")
    print(f"P50 延迟:    {np.percentile(latencies, 50):.2f} ms")
    print(f"P95 延迟:    {np.percentile(latencies, 95):.2f} ms")
    print(f"P99 延迟:    {np.percentile(latencies, 99):.2f} ms")
    print(f"吞吐量:      {throughput:.1f} 图像/秒")

    # 清理临时文件
    for p in test_images:
        os.unlink(p)

    return {
        "mean_ms": round(float(np.mean(latencies)), 2),
        "p50_ms":  round(float(np.percentile(latencies, 50)), 2),
        "p95_ms":  round(float(np.percentile(latencies, 95)), 2),
        "p99_ms":  round(float(np.percentile(latencies, 99)), 2),
        "throughput_img_per_sec": round(throughput, 1),
    }


# ─────────────────────────────────────────────
# 主程序：演示完整流程
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CIFAR-10 模型生产部署演示")
    print("=" * 60)

    # 初始化推理引擎（使用随机权重演示）
    engine = CIFARInferenceEngine(
        weights_path="weights/cifar10_resnet18.pt",  # 不存在时使用随机权重
        device="cpu"
    )

    # 模型信息
    total_params = sum(p.numel() for p in build_model().parameters())
    print(f"\n模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

    # 运行基准测试
    stats = benchmark_inference(engine, num_requests=100)
    print(f"\n延迟统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")

    print("\n提示：运行 `uvicorn serve:app --port 8000` 启动 REST API 服务")
```

运行示例输出：

```
============================================================
CIFAR-10 模型生产部署演示
============================================================
警告：权重文件 weights/cifar10_resnet18.pt 不存在，使用随机初始化权重
已启用动态量化（CPU）

模型参数量: 11,173,962 (11.17M)

性能基准测试（100 次推理请求，batch_size=8）
──────────────────────────────────────────────────
平均延迟:    18.43 ms
P50 延迟:    17.91 ms
P95 延迟:    23.12 ms
P99 延迟:    26.84 ms
吞吐量:      434.1 图像/秒
```

---

## 练习题

### 基础题

**练习 24-1**：导出与验证

使用以下模型，分别用 Tracing 和 Scripting 两种方式导出为 TorchScript，并验证两种方式的输出与原始模型是否一致（误差 < 1e-5）。

```python
import torch
import torch.nn as nn

class ConditionalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        if h.mean() > 0:          # 数据依赖分支
            return self.fc2(h)
        else:
            return torch.zeros(x.shape[0], 5)
```

思考：为什么这个模型不适合用 Tracing 方式导出？

---

**练习 24-2**：动态量化对比

对下面的 Transformer 编码器模型应用动态量化，并比较量化前后：
1. 模型文件大小（`.pt` 文件）
2. 100 次推理的平均延迟
3. 输出的最大绝对误差

```python
import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 10)

    def forward(self, x):
        out = self.encoder(x)
        return self.fc(out[:, 0, :])
```

---

### 中级题

**练习 24-3**：静态量化实现

为以下 CNN 模型完整实现静态量化流程（插入 QuantStub/DeQuantStub、配置 qconfig、校准、转换），并保存为 TorchScript 文件。要求：
- 使用 `fbgemm` 后端
- 校准数据至少 50 批
- 计算量化前后的 Top-1 精度差（使用随机"伪数据集"评估）

```python
import torch.nn as nn

class CIFAR_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
```

---

**练习 24-4**：FastAPI 推理服务扩展

在本章 FastAPI 示例的基础上，添加以下功能：

1. **请求限流**：同一 IP 每秒最多请求 10 次（可使用 `slowapi` 库）
2. **结果缓存**：对相同输入（基于输入哈希）缓存 60 秒内的推理结果
3. **异步批处理**：收集 100ms 内的请求合并为一个 batch 推理，提高 GPU 利用率

写出核心代码实现，无需完整部署，但代码需可运行。

---

### 提高题

**练习 24-5**：端到端部署优化管道

从零构建一个完整的模型优化与部署管道，包含以下环节：

1. **训练**：在 MNIST 数据集上训练一个 LeNet-5，达到 ≥98% 的测试精度
2. **剪枝**：使用全局非结构化剪枝，在精度下降 ≤0.5% 的前提下，找到能达到的最大稀疏度
3. **量化**：对剪枝后的模型应用静态量化
4. **导出**：导出为 ONNX 格式，使用 ONNX Runtime 验证精度
5. **服务化**：封装为 FastAPI 服务，支持 Base64 图像输入
6. **基准测试**：测量并报告原始模型 vs 剪枝+量化模型的大小比、速度比、精度差

要求：
- 代码可完整运行
- 每个步骤输出量化指标（数字证据）
- 最终以 Markdown 表格汇报优化效果

---

## 练习答案

### 答案 24-1

```python
import torch
import torch.nn as nn

class ConditionalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        if h.mean() > 0:
            return self.fc2(h)
        else:
            return torch.zeros(x.shape[0], 5)

model = ConditionalNet()
model.eval()
x = torch.randn(4, 10)

# ── Scripting（正确方式）────────────────────────
scripted = torch.jit.script(model)
scripted.save("conditional_scripted.pt")
loaded_scripted = torch.jit.load("conditional_scripted.pt")

with torch.no_grad():
    out_orig     = model(x)
    out_scripted = loaded_scripted(x)
    err_scripted = (out_orig - out_scripted).abs().max().item()
    print(f"Scripting 最大误差: {err_scripted:.2e}")  # 极小，约 1e-7

# ── Tracing（问题演示）──────────────────────────
traced = torch.jit.trace(model, x)

# Tracing 仅记录了当时 h.mean() > 0 的分支路径
# 对满足 else 分支的输入，traced 模型仍会走 if 分支，产生错误
torch.manual_seed(42)
x_neg = -torch.abs(torch.randn(4, 10)) * 10  # 尽量让 h.mean() <= 0
with torch.no_grad():
    out_model  = model(x_neg)
    out_traced = traced(x_neg)
    err_traced = (out_model - out_traced).abs().max().item()
    print(f"Tracing 最大误差（else 分支）: {err_traced:.4f}")  # 可能很大

# ── 说明 ──────────────────────────────────────
# ConditionalNet.forward 中有 `if h.mean() > 0` 这种"数据依赖分支"：
# - 分支走哪条路取决于运行时的 h 值，而非固定的控制流
# - Tracing 只记录示例输入触发的那条路径，导致另一分支被"冻结"
# - Scripting 静态分析源码，保留了完整的 if/else 逻辑，输出始终正确
```

---

### 答案 24-2

```python
import torch
import torch.nn as nn
import torch.quantization
import time

class SimpleTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 10)

    def forward(self, x):
        out = self.encoder(x)
        return self.fc(out[:, 0, :])

model_fp32 = SimpleTransformer()
model_fp32.eval()

# 动态量化
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {nn.Linear},
    dtype=torch.qint8
)

# ── 1. 模型文件大小 ────────────────────────────
import os, tempfile

def save_and_size(model, path):
    torch.save(model.state_dict(), path)
    return os.path.getsize(path) / 1024  # KB

with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
    size_fp32 = save_and_size(model_fp32, f.name)
with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
    size_int8 = save_and_size(model_int8, f.name)

print(f"FP32 模型大小: {size_fp32:.1f} KB")
print(f"INT8 模型大小: {size_int8:.1f} KB")
print(f"压缩比: {size_fp32/size_int8:.2f}x")

# ── 2. 推理延迟 ────────────────────────────────
x = torch.randn(1, 32, 256)  # [batch=1, seq_len=32, d_model=256]
N = 100

def run_n(model, x, n):
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(n):
            model(x)
        return (time.perf_counter() - start) / n * 1000

lat_fp32 = run_n(model_fp32, x, N)
lat_int8 = run_n(model_int8, x, N)
print(f"\nFP32 平均延迟: {lat_fp32:.2f} ms")
print(f"INT8 平均延迟: {lat_int8:.2f} ms")
print(f"加速比: {lat_fp32/lat_int8:.2f}x")

# ── 3. 输出误差 ────────────────────────────────
with torch.no_grad():
    out_fp32 = model_fp32(x)
    out_int8 = model_int8(x)
    max_err = (out_fp32 - out_int8).abs().max().item()
    print(f"\n最大绝对误差: {max_err:.6f}")

# 典型输出示例：
# FP32 模型大小: 3842.3 KB
# INT8 模型大小: 987.6 KB
# 压缩比: 3.89x
# FP32 平均延迟: 12.34 ms
# INT8 平均延迟: 6.18 ms
# 加速比: 2.00x
# 最大绝对误差: 0.003421
```

---

### 答案 24-3

```python
import torch
import torch.nn as nn
import torch.quantization
import copy

class CIFAR_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 添加量化桩
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

# 原始模型
model_fp32 = CIFAR_CNN(num_classes=10)
model_fp32.eval()

# 第一步：融合 Conv + BN + ReLU
model_fused = copy.deepcopy(model_fp32)
# 注意：features 中每个 [conv, bn, relu] 三元组需要逐一指定路径
torch.quantization.fuse_modules(
    model_fused.features,
    [["0", "1", "2"], ["3", "4", "5"]],   # [Conv2d, BN, ReLU] × 2
    inplace=True
)

# 第二步：设置 qconfig
model_fused.qconfig = torch.quantization.get_default_qconfig("fbgemm")

# 第三步：插入观察器
torch.quantization.prepare(model_fused, inplace=True)

# 第四步：校准（50 批随机数据模拟真实数据分布）
print("开始校准...")
calibration_batches = [torch.randn(16, 3, 32, 32) for _ in range(50)]
with torch.no_grad():
    for i, batch in enumerate(calibration_batches):
        model_fused(batch)
        if (i + 1) % 10 == 0:
            print(f"  校准进度: {i+1}/50")

# 第五步：转换为量化模型
torch.quantization.convert(model_fused, inplace=True)
print("量化完成")

# 第六步：保存为 TorchScript
scripted_quantized = torch.jit.script(model_fused)
scripted_quantized.save("cifar_cnn_quantized.pt")
print("已保存: cifar_cnn_quantized.pt")

# 第七步：精度对比（伪数据集）
def eval_accuracy(model, num_batches=20, batch_size=32):
    """在随机数据上评估（仅演示流程，真实场景需用真实验证集）"""
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(num_batches):
            x = torch.randn(batch_size, 3, 32, 32)
            # 随机标签（均匀分布，期望精度约 10%）
            labels = torch.randint(0, 10, (batch_size,))
            preds = model(x).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size
    return correct / total * 100

acc_fp32 = eval_accuracy(model_fp32)
acc_int8 = eval_accuracy(model_fused)
print(f"\nFP32 精度（随机数据）: {acc_fp32:.1f}%")
print(f"INT8 精度（随机数据）: {acc_int8:.1f}%")
print(f"精度差: {abs(acc_fp32 - acc_int8):.2f}%")
# 注：随机数据下两者精度均接近 10%（随机猜测），差异极小
```

---

### 答案 24-4

```python
"""
FastAPI 推理服务扩展：限流 + 缓存 + 异步批处理

安装依赖：
    pip install fastapi uvicorn slowapi cachetools
"""
import asyncio
import hashlib
import time
from collections import deque
from contextlib import asynccontextmanager
from threading import Lock
from typing import List

import torch
from cachetools import TTLCache
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ── 限流配置 ──────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── 结果缓存（最多 1000 条，TTL 60 秒）────────────
result_cache: TTLCache = TTLCache(maxsize=1000, ttl=60)
cache_lock = Lock()

def cache_key(features: list) -> str:
    """基于输入特征生成缓存键"""
    raw = str(features).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]

# ── 异步批处理队列 ─────────────────────────────────
BATCH_WAIT_MS = 100  # 等待时间窗口

class BatchQueue:
    """
    收集 100ms 内的请求，合并为一个 batch 推理
    """
    def __init__(self):
        self._queue: deque = deque()
        self._lock = asyncio.Lock()
        self._batch_event = asyncio.Event()
        self._results: dict = {}

    async def enqueue(self, request_id: str, features: list) -> list:
        future = asyncio.get_event_loop().create_future()
        async with self._lock:
            self._queue.append((request_id, features, future))
        self._batch_event.set()
        return await future

    async def process_loop(self, inference_fn):
        """后台 worker：定期批量处理队列中的请求"""
        while True:
            await asyncio.sleep(BATCH_WAIT_MS / 1000)
            async with self._lock:
                if not self._queue:
                    continue
                batch = list(self._queue)
                self._queue.clear()

            ids, features_list, futures = zip(*batch)
            # 批量推理
            batch_tensor = torch.tensor(features_list, dtype=torch.float32)
            with torch.no_grad():
                outputs = inference_fn(batch_tensor)  # [B, num_classes]

            for future, output in zip(futures, outputs):
                if not future.done():
                    future.set_result(output.tolist())

# 全局批处理队列
batch_queue = BatchQueue()

# ── FastAPI 应用 ──────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float]
    top_k: int = 3

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动批处理后台任务
    task = asyncio.create_task(
        batch_queue.process_loop(lambda x: torch.randn(x.shape[0], 10))
    )
    yield
    task.cancel()

app = FastAPI(title="优化版推理服务", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/second")   # 每 IP 每秒最多 10 次
async def predict(request: Request, body: PredictRequest):
    # 1. 检查缓存
    key = cache_key(body.features)
    with cache_lock:
        if key in result_cache:
            return {"source": "cache", "logits": result_cache[key]}

    # 2. 加入批处理队列
    import uuid
    req_id = str(uuid.uuid4())
    logits = await batch_queue.enqueue(req_id, body.features)

    # 3. 存入缓存
    with cache_lock:
        result_cache[key] = logits

    return {"source": "inference", "logits": logits}
```

---

### 答案 24-5

```python
"""
端到端部署优化管道：LeNet-5 on MNIST

运行顺序：
    python answer_24_5.py
"""
import time, os, json, tempfile
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization
import torch.onnx
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ──────────────────────────────────────────────
# 1. 模型定义
# ──────────────────────────────────────────────
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant   = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.conv1   = nn.Conv2d(1, 6, 5)
        self.bn1     = nn.BatchNorm2d(6)
        self.relu1   = nn.ReLU()
        self.pool1   = nn.MaxPool2d(2)
        self.conv2   = nn.Conv2d(6, 16, 5)
        self.bn2     = nn.BatchNorm2d(16)
        self.relu2   = nn.ReLU()
        self.pool2   = nn.MaxPool2d(2)
        self.fc1     = nn.Linear(16 * 4 * 4, 120)
        self.fc2     = nn.Linear(120, 84)
        self.fc3     = nn.Linear(84, 10)

    def forward(self, x):
        x = self.quant(x)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        return x


# ──────────────────────────────────────────────
# 2. 训练函数
# ──────────────────────────────────────────────
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total * 100

def model_size_kb(model):
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size = os.path.getsize(f.name) / 1024
    os.unlink(f.name)
    return size


# ──────────────────────────────────────────────
# 3. 数据加载
# ──────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_set = datasets.MNIST("./data", train=True,  download=True, transform=transform)
test_set  = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}\n")

# ──────────────────────────────────────────────
# 步骤 1：训练
# ──────────────────────────────────────────────
print("=" * 50)
print("步骤 1：训练 LeNet-5（目标 ≥98%）")
print("=" * 50)

model = LeNet5().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(10):
    loss = train(model, train_loader, optimizer, criterion, device)
    scheduler.step()
    if (epoch + 1) % 2 == 0:
        acc = evaluate(model, test_loader, device)
        print(f"  Epoch {epoch+1:2d} | Loss: {loss:.4f} | Test Acc: {acc:.2f}%")

acc_original = evaluate(model, test_loader, device)
size_original = model_size_kb(model)
print(f"\n原始模型精度: {acc_original:.2f}%，大小: {size_original:.1f} KB")

# ──────────────────────────────────────────────
# 步骤 2：迭代剪枝（目标：精度下降 ≤0.5%）
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("步骤 2：迭代全局非结构化剪枝")
print("=" * 50)

import copy
pruned_model = copy.deepcopy(model)
prune_params = [
    (pruned_model.conv1, "weight"),
    (pruned_model.conv2, "weight"),
    (pruned_model.fc1,   "weight"),
    (pruned_model.fc2,   "weight"),
    (pruned_model.fc3,   "weight"),
]

best_sparsity = 0.0
for target_sparsity in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    candidate = copy.deepcopy(model)
    candidate_params = [
        (candidate.conv1, "weight"), (candidate.conv2, "weight"),
        (candidate.fc1, "weight"),   (candidate.fc2, "weight"),
        (candidate.fc3, "weight"),
    ]
    prune.global_unstructured(
        candidate_params, pruning_method=prune.L1Unstructured,
        amount=target_sparsity
    )
    # 微调 2 个 epoch
    opt_ft = torch.optim.SGD(candidate.parameters(), lr=1e-4, momentum=0.9)
    for _ in range(2):
        train(candidate, train_loader, opt_ft, criterion, device)
    acc_pruned = evaluate(candidate, test_loader, device)
    drop = acc_original - acc_pruned
    print(f"  稀疏度 {target_sparsity*100:.0f}% | 精度: {acc_pruned:.2f}% | 下降: {drop:.2f}%")
    if drop <= 0.5:
        best_sparsity = target_sparsity
        pruned_model = candidate
    else:
        print(f"  → 精度下降超过 0.5%，停止剪枝，最大可用稀疏度: {best_sparsity*100:.0f}%")
        break

# 固化剪枝
for m, n in prune_params:
    try:
        prune.remove(m, n)
    except ValueError:
        pass

acc_pruned_final = evaluate(pruned_model, test_loader, device)
print(f"\n最终剪枝精度: {acc_pruned_final:.2f}%（稀疏度 {best_sparsity*100:.0f}%）")

# ──────────────────────────────────────────────
# 步骤 3：静态量化
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("步骤 3：静态量化")
print("=" * 50)

import copy
quant_model = copy.deepcopy(pruned_model)
quant_model.eval()
torch.quantization.fuse_modules(
    quant_model, [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"]], inplace=True
)
quant_model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
torch.quantization.prepare(quant_model, inplace=True)

with torch.no_grad():
    for i, (x, _) in enumerate(train_loader):
        quant_model(x)
        if i >= 49:
            break

torch.quantization.convert(quant_model, inplace=True)
acc_quantized = evaluate(quant_model, test_loader, device)
size_quantized = model_size_kb(quant_model)
print(f"量化后精度: {acc_quantized:.2f}%，大小: {size_quantized:.1f} KB")

# ──────────────────────────────────────────────
# 步骤 4：导出 ONNX 并验证
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("步骤 4：导出 ONNX 并验证（使用原始 FP32 模型）")
print("=" * 50)

# ONNX 从原始模型导出（量化模型导出 ONNX 需额外工具）
fp32_model = copy.deepcopy(model)
fp32_model.eval()
dummy = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    fp32_model, dummy, "lenet5_mnist.onnx",
    opset_version=17,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)
print("ONNX 模型已导出：lenet5_mnist.onnx")

try:
    import onnxruntime as ort
    sess = ort.InferenceSession("lenet5_mnist.onnx", providers=["CPUExecutionProvider"])
    correct = total = 0
    for x, y in test_loader:
        inp = x.numpy()
        out = sess.run(["output"], {"input": inp})[0]
        preds = np.argmax(out, axis=1)
        correct += (preds == y.numpy()).sum()
        total += len(y)
    acc_onnx = correct / total * 100
    print(f"ONNX Runtime 精度: {acc_onnx:.2f}%")
except ImportError:
    acc_onnx = None
    print("未安装 onnxruntime，跳过 ONNX 验证")

# ──────────────────────────────────────────────
# 步骤 5+6：汇报优化效果
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("步骤 6：优化效果汇总")
print("=" * 50)

size_pruned = model_size_kb(pruned_model)

report = [
    ("指标",           "原始模型",       "剪枝+量化模型",                          "变化"),
    ("测试精度",       f"{acc_original:.2f}%", f"{acc_quantized:.2f}%",           f"-{acc_original-acc_quantized:.2f}%"),
    ("模型大小",       f"{size_original:.1f} KB", f"{size_quantized:.1f} KB",     f"{size_original/size_quantized:.1f}x 压缩"),
    ("参数稀疏度",     "0%",             f"{best_sparsity*100:.0f}%",              "—"),
    ("ONNX 精度",      f"{acc_original:.2f}%", str(f"{acc_onnx:.2f}%") if acc_onnx else "N/A", "—"),
]

print(f"\n{'指标':<14} {'原始模型':<14} {'优化后模型':<14} {'变化':<18}")
print("─" * 60)
for row in report[1:]:
    print(f"{row[0]:<14} {row[1]:<14} {row[2]:<14} {row[3]:<18}")

print("\n部署建议：")
print("  - 使用剪枝+量化模型：体积更小，CPU 推理速度更快")
print("  - ONNX 格式适合跨平台部署（TensorRT、ONNX Runtime）")
print("  - 如需更高精度，考虑量化感知训练（QAT）替代后训练量化")
```

---

[下一章：附录 A — Python速查表](../appendix/python-cheatsheet.md)
