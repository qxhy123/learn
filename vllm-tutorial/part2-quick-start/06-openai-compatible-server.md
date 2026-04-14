# 第6章：OpenAI 兼容服务器

> vLLM 最强大的特性之一是提供 OpenAI 兼容的 API 服务器。这意味着你可以用现有的 OpenAI SDK 代码，零修改地切换到自部署的模型。

---

## 学习目标

学完本章，你将能够：

1. 启动 vLLM 的 OpenAI 兼容 API 服务器
2. 使用 Chat Completions 和 Completions 端点
3. 实现流式输出（streaming）
4. 使用 OpenAI Python SDK 调用 vLLM 服务
5. 理解 API 服务器的配置选项和管理端点

---

## 6.1 启动 API 服务器

### 最简启动

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

这会在 `http://localhost:8000` 启动一个 OpenAI 兼容的 API 服务器。

### 常用启动参数

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --max-num-seqs 128 \
    --api-key my-secret-key \
    --served-model-name my-model
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 监听地址 | `0.0.0.0` |
| `--port` | 监听端口 | `8000` |
| `--dtype` | 数据类型 | `auto` |
| `--gpu-memory-utilization` | GPU 显存使用比例 | `0.9` |
| `--max-model-len` | 最大序列长度 | 模型默认值 |
| `--max-num-seqs` | 最大并发请求数 | `256` |
| `--api-key` | API 密钥认证 | 无 |
| `--served-model-name` | 对外暴露的模型名 | 模型路径 |

---

## 6.2 Chat Completions API

这是最常用的端点，适合对话场景。

### 使用 curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手。"},
      {"role": "user", "content": "什么是机器学习？"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### 使用 OpenAI Python SDK

```python
from openai import OpenAI

# 指向本地 vLLM 服务器
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # 如果没设置 --api-key
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "解释一下 KV Cache。"},
    ],
    max_tokens=256,
    temperature=0.7,
)

print(response.choices[0].message.content)
```

### 响应结构

```json
{
  "id": "chat-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "KV Cache 是一种用于加速 Transformer 推理的技术..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 128,
    "total_tokens": 153
  }
}
```

---

## 6.3 流式输出（Streaming）

流式输出让用户在生成过程中就能看到结果，大幅改善体感延迟。

### curl 流式请求

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "写一首诗"}],
    "stream": true,
    "max_tokens": 200
  }'
```

### Python SDK 流式请求

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "讲一个关于编程的笑话。"}],
    stream=True,
    max_tokens=200,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # 换行
```

### 异步流式请求

```python
import asyncio
from openai import AsyncOpenAI

async def stream_chat():
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
    )

    stream = await client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
        max_tokens=200,
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

asyncio.run(stream_chat())
```

---

## 6.4 Completions API

适合纯文本补全场景（非对话）。

```python
response = client.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    prompt="Once upon a time, there was a",
    max_tokens=100,
    temperature=0.7,
)

print(response.choices[0].text)
```

### 批量补全

```python
# 一次请求多个 prompt
response = client.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    prompt=[
        "The capital of France is",
        "Python is a programming language that",
        "Machine learning is",
    ],
    max_tokens=50,
    temperature=0,
)

for choice in response.choices:
    print(f"[{choice.index}] {choice.text.strip()}")
```

---

## 6.5 额外端点与 vLLM 扩展

### 查看模型列表

```bash
curl http://localhost:8000/v1/models
```

```python
models = client.models.list()
for model in models.data:
    print(f"Model: {model.id}")
```

### 健康检查

```bash
curl http://localhost:8000/health
# 返回 200 表示服务健康
```

### Token 化端点

vLLM 提供了一个额外的 tokenize 端点：

```bash
curl http://localhost:8000/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "Hello, world!"
  }'
```

### 获取服务器版本

```bash
curl http://localhost:8000/version
```

---

## 6.6 并发请求与性能

### 模拟高并发

```python
import asyncio
import time
from openai import AsyncOpenAI

async def send_request(client, request_id):
    start = time.time()
    response = await client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[{"role": "user", "content": f"Count from 1 to 20. Request {request_id}"}],
        max_tokens=100,
    )
    elapsed = time.time() - start
    tokens = response.usage.completion_tokens
    return request_id, elapsed, tokens

async def benchmark(num_requests=50):
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
    )

    start = time.time()
    tasks = [send_request(client, i) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start

    total_tokens = sum(r[2] for r in results)
    avg_latency = sum(r[1] for r in results) / len(results)

    print(f"总请求数: {num_requests}")
    print(f"总时间: {total_time:.1f}s")
    print(f"平均延迟: {avg_latency:.2f}s")
    print(f"请求吞吐: {num_requests / total_time:.1f} req/s")
    print(f"Token 吞吐: {total_tokens / total_time:.0f} tokens/s")

asyncio.run(benchmark())
```

### 观察要点

- **请求吞吐**随并发增加而提升（连续批处理的效果）
- **单请求延迟**在高并发时可能增加（排队效应）
- **Token 吞吐**有上限（受 GPU 计算和显存限制）

---

## 6.7 API 认证

在生产环境中，应该为 API 添加认证。

### 服务端设置

```bash
# 使用 --api-key 设置密钥
vllm serve model --api-key sk-my-secret-key-12345

# 或通过环境变量
export VLLM_API_KEY=sk-my-secret-key-12345
vllm serve model
```

### 客户端使用

```python
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-my-secret-key-12345",
)
```

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 启动服务 | `vllm serve model_name` 一行命令 |
| Chat API | `/v1/chat/completions`，兼容 OpenAI 格式 |
| 流式输出 | `stream=True`，实时返回生成结果 |
| Completions API | `/v1/completions`，纯文本补全 |
| 并发性能 | vLLM 自动连续批处理，吞吐随并发提升 |
| 认证 | `--api-key` 参数设置 API 密钥 |

---

## 动手实验

### 实验 1：搭建完整的聊天服务

1. 启动 vLLM 服务器
2. 编写一个简单的命令行聊天客户端（支持多轮对话、流式输出）
3. 测试不同 temperature 对回答风格的影响

### 实验 2：并发压测

用上面的 benchmark 脚本，分别测试 10、50、100、200 个并发请求，记录吞吐和延迟变化。

### 实验 3：从 OpenAI 无缝切换

如果你有使用 OpenAI API 的现有代码，尝试只修改 `base_url` 和 `api_key`，验证代码是否可以直接运行。

---

## 练习题

### 基础题

1. vLLM API 服务器默认监听哪个端口？如何修改？
2. Chat Completions API 和 Completions API 的区别是什么？
3. 流式输出的优势是什么？它改变了总生成时间吗？

### 实践题

4. 启动 vLLM 服务器并用 curl 成功调用 Chat API。
5. 编写一个使用 OpenAI SDK 的多轮对话脚本，支持流式输出。

### 思考题

6. 为什么 vLLM 选择兼容 OpenAI API 格式？这对用户有什么价值？
7. 在高并发场景下，你观察到的吞吐提升是否有上限？这个上限由什么决定？
