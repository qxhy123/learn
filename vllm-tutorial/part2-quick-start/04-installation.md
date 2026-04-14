# 第4章：安装与环境配置

> 工欲善其事，必先利其器。vLLM 的安装虽然比传统 Python 包稍微复杂（因为涉及 CUDA 编译），但官方已经做了大量工作让这个过程尽量简单。

---

## 学习目标

学完本章，你将能够：

1. 根据自己的环境选择合适的 vLLM 安装方式
2. 正确配置 CUDA 驱动和 GPU 环境
3. 验证 vLLM 安装并运行健康检查
4. 排查常见安装问题
5. 使用 Docker 快速启动 vLLM

---

## 4.1 环境要求

### 硬件要求

| 要求 | 最低 | 推荐 |
|------|------|------|
| GPU | NVIDIA，Compute Capability >= 7.0 | A100 / H100 / L40S |
| GPU 显存 | 16 GB（运行 7B 模型） | 24 GB+（更灵活的模型选择） |
| 系统内存 | 16 GB | 64 GB+（模型加载时需要） |
| 磁盘 | 50 GB 可用 | 200 GB+（存放模型权重） |

常见 GPU 及其 Compute Capability：

| GPU | Compute Capability | 显存 | 适用性 |
|-----|-------------------|------|--------|
| V100 | 7.0 | 16/32 GB | 可用，但性能有限 |
| T4 | 7.5 | 16 GB | 可用于小模型 |
| A10G | 8.6 | 24 GB | 云端常见 |
| A100 | 8.0 | 40/80 GB | 推荐 |
| L40S | 8.9 | 48 GB | 性价比高 |
| H100 | 9.0 | 80 GB | 最佳性能 |

### 软件要求

```
操作系统: Linux (推荐 Ubuntu 20.04/22.04)
Python:   >= 3.9
CUDA:     >= 12.1
驱动:     >= 525.60
PyTorch:  >= 2.4 (vLLM 会自动安装匹配版本)
```

⚠️ **注意**：vLLM 对 macOS 和 Windows 的原生支持有限。在这些平台上建议使用 Docker 或远程 Linux 服务器。

---

## 4.2 安装方式

### 方式一：pip 安装（推荐）

最简单的安装方式：

```bash
# 创建虚拟环境（强烈推荐）
python -m venv vllm-env
source vllm-env/bin/activate

# 安装 vLLM
pip install vllm
```

这会自动安装 vLLM 及其依赖（包括 PyTorch 和预编译的 CUDA 扩展）。

### 方式二：conda 安装

```bash
# 创建 conda 环境
conda create -n vllm python=3.11 -y
conda activate vllm

# 安装 vLLM
pip install vllm
```

### 方式三：从源码安装

适合需要修改 vLLM 源码或使用最新开发版的用户：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

⚠️ 源码编译需要较长时间（可能 10-30 分钟），并需要 CUDA 开发工具链。

### 方式四：Docker（最省心）

```bash
# 使用官方预构建镜像
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-1.5B-Instruct
```

Docker 方式的优势：
- 不需要担心 CUDA 版本兼容性
- 环境完全隔离
- 适合生产部署

---

## 4.3 验证安装

### 基本验证

```python
# 验证 vLLM 是否正确安装
import vllm
print(f"vLLM version: {vllm.__version__}")

# 验证 CUDA 是否可用
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f"    Memory: {mem:.1f} GB")
```

### 运行最小推理测试

```python
from vllm import LLM, SamplingParams

# 使用一个小模型做快速验证
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")
params = SamplingParams(max_tokens=50, temperature=0)
outputs = llm.generate(["Hello, world!"], params)
print(outputs[0].outputs[0].text)
print("✓ vLLM is working correctly!")
```

### 命令行验证

```bash
# 查看 vLLM 版本
python -c "import vllm; print(vllm.__version__)"

# 检查 GPU 状态
nvidia-smi
```

---

## 4.4 常见问题排查

### 问题 1：CUDA 版本不兼容

```
RuntimeError: The detected CUDA version (11.8) mismatches the version
that was used to compile PyTorch (12.1)
```

**解决方案**：

```bash
# 检查系统 CUDA 版本
nvcc --version
nvidia-smi  # 右上角显示支持的最高 CUDA 版本

# 安装匹配的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install vllm
```

### 问题 2：显存不足（OOM）

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解决方案**：

```bash
# 减少 GPU 显存使用比例
vllm serve model --gpu-memory-utilization 0.8

# 限制最大序列长度
vllm serve model --max-model-len 2048

# 使用量化模型
vllm serve model-awq --quantization awq
```

### 问题 3：模型下载失败

```bash
# 设置 Hugging Face 镜像（中国用户）
export HF_ENDPOINT=https://hf-mirror.com

# 或提前下载模型
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./models/qwen
vllm serve ./models/qwen
```

### 问题 4：`ninja` 编译错误

```bash
# 安装 ninja
pip install ninja

# 如果仍有问题，设置环境变量跳过自定义编译
export VLLM_USE_PRECOMPILED=1
pip install vllm
```

### 问题 5：多 GPU 时只使用了一张卡

```bash
# 显式指定使用的 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 启动时指定张量并行度
vllm serve model --tensor-parallel-size 4
```

---

## 4.5 模型下载与管理

### Hugging Face Hub

vLLM 默认从 Hugging Face Hub 下载模型：

```python
# 自动下载（首次使用时）
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# 模型默认缓存在 ~/.cache/huggingface/hub/
```

### 使用本地模型

```python
# 从本地路径加载
llm = LLM(model="/path/to/local/model")
```

### 需要认证的模型

某些模型（如 Llama 系列）需要先在 Hugging Face 上申请访问权限：

```bash
# 登录 Hugging Face
huggingface-cli login

# 或设置 token 环境变量
export HF_TOKEN=hf_xxxxxxxxxxxxx
```

### 推荐的入门模型

| 模型 | 大小 | 显存需求 | 说明 |
|------|------|---------|------|
| Qwen/Qwen2.5-0.5B-Instruct | 0.5B | ~2 GB | 快速测试 |
| Qwen/Qwen2.5-1.5B-Instruct | 1.5B | ~4 GB | 入门实验 |
| Qwen/Qwen2.5-7B-Instruct | 7B | ~16 GB | 实际使用 |
| meta-llama/Llama-3.1-8B-Instruct | 8B | ~18 GB | 需要申请 |
| mistralai/Mistral-7B-Instruct-v0.3 | 7B | ~16 GB | 开箱即用 |

---

## 本章小结

| 主题 | 要点 |
|------|------|
| 硬件要求 | NVIDIA GPU，Compute Capability >= 7.0，16GB+ 显存 |
| 推荐安装 | `pip install vllm` 或 Docker |
| 验证 | 导入 vllm、检查 CUDA、运行小模型推理 |
| 常见问题 | CUDA 版本、OOM、模型下载、多 GPU 配置 |
| 模型管理 | HF Hub 自动下载或本地路径加载 |

---

## 动手实验

### 实验 1：完整安装与验证

1. 创建虚拟环境并安装 vLLM
2. 运行基本验证脚本，确认版本和 GPU 信息
3. 加载一个小模型（如 Qwen2.5-0.5B-Instruct）并完成一次推理
4. 记录安装耗时和遇到的问题

### 实验 2：探索 GPU 显存

```bash
# 启动 vLLM 前后分别查看显存
nvidia-smi

# 加载模型
python -c "from vllm import LLM; llm = LLM('Qwen/Qwen2.5-1.5B-Instruct'); input('Press Enter...')"

# 在另一个终端观察显存变化
watch -n 1 nvidia-smi
```

---

## 练习题

### 基础题

1. vLLM 支持的最低 GPU Compute Capability 是多少？列出两款支持的 GPU。
2. 如果系统 CUDA 版本是 11.8，应该如何处理？

### 实践题

3. 在你的机器上安装 vLLM 并运行验证脚本。记录你的 GPU 型号、显存大小和 vLLM 版本。
4. 使用 Docker 方式启动 vLLM 服务器并成功完成一次 API 调用。

### 思考题

5. 为什么 vLLM 推荐使用 Linux 而不是 Windows？这与 CUDA 生态有什么关系？
