# 附录A：vLLM CLI 与 API 速查

## 服务器启动参数

### 基础参数

```bash
vllm serve <model> \
    --host 0.0.0.0 \                    # 监听地址
    --port 8000 \                        # 监听端口
    --dtype auto \                       # 数据类型: auto/float16/bfloat16
    --served-model-name my-model \       # 对外模型名
    --api-key sk-xxx \                   # API 认证密钥
    --response-role assistant            # 响应角色名
```

### 资源参数

```bash
    --gpu-memory-utilization 0.9 \       # GPU 显存使用比例 (0-1)
    --max-model-len 4096 \               # 最大序列长度
    --max-num-seqs 256 \                 # 最大并发序列数
    --max-num-batched-tokens 8192 \      # 每 iteration 最大 token 数
    --block-size 16                      # KV Cache 块大小
```

### 并行参数

```bash
    --tensor-parallel-size 1 \           # 张量并行度
    --pipeline-parallel-size 1 \         # 流水线并行度
    --distributed-executor-backend ray   # 分布式后端: ray/mp
```

### 量化参数

```bash
    --quantization awq \                 # 量化方案: awq/gptq/fp8/squeezellm
    --kv-cache-dtype auto                # KV Cache 类型: auto/fp8
```

### 调度参数

```bash
    --max-num-partial-prefills 1 \       # 最大并发 partial prefill 数
    --max-long-partial-prefills 1 \      # 最大并发长 prompt partial prefill 数
    --long-prefill-token-threshold 0 \   # 长 prefill 阈值 (0=自动)
    --scheduling-policy fcfs \           # 调度策略: fcfs/priority
    --async-scheduling \                 # 启用异步调度
    --stream-interval 1                  # 流式输出间隔
```

### 高级参数

```bash
    --enable-prefix-caching \            # 启用前缀缓存
    --prefix-caching-hash-algo sha256 \  # 前缀缓存 hash 算法
    --enable-chunked-prefill \           # 启用分块 prefill
    --speculative-config '{"method":"ngram","num_speculative_tokens":4}' \  # 投机解码配置
    --structured-outputs-config.backend auto \  # 结构化输出后端
    --enable-lora \                      # 启用 LoRA
    --max-loras 4 \                      # 最大 LoRA 数
    --trust-remote-code                  # 信任远程代码
```

---

## SamplingParams

```python
from vllm import SamplingParams

params = SamplingParams(
    # 基础
    max_tokens=256,              # 最大生成 token 数
    min_tokens=0,                # 最小生成 token 数
    stop=None,                   # 停止词列表
    
    # 采样
    temperature=1.0,             # 温度 (0=贪心)
    top_p=1.0,                   # Nucleus sampling
    top_k=-1,                    # Top-k sampling (-1=禁用)
    
    # 惩罚
    repetition_penalty=1.0,      # 重复惩罚
    frequency_penalty=0.0,       # 频率惩罚
    presence_penalty=0.0,        # 存在惩罚
    
    # 多输出
    n=1,                         # 每请求生成数
    
    # 其他
    seed=None,                   # 随机种子
    structured_outputs=None,     # JSON/regex/choice/grammar 约束
    skip_special_tokens=True,    # 跳过特殊 token
    ignore_eos=False,            # 忽略 EOS
)
```

---

## API 端点

### Chat Completions

```
POST /v1/chat/completions

请求体:
{
    "model": "model-name",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": false
}
```

### Completions

```
POST /v1/completions

请求体:
{
    "model": "model-name",
    "prompt": "Hello",
    "max_tokens": 256,
    "temperature": 0.7
}
```

### 管理端点

```
GET  /health             # 健康检查
GET  /v1/models          # 模型列表
GET  /metrics            # Prometheus 指标
GET  /version            # 版本信息
POST /tokenize           # 分词
POST /detokenize         # 反分词
```

---

## LLM 离线 API

```python
from vllm import LLM, SamplingParams

# 初始化
llm = LLM(model="model-name", **engine_args)

# 文本生成
outputs = llm.generate(prompts, sampling_params)

# 对话生成
outputs = llm.chat(messages_list, sampling_params)

# 带 LoRA
outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
```

---

## 环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `CUDA_VISIBLE_DEVICES` | 可见 GPU | `0,1,2,3` |
| `VLLM_LOGGING_LEVEL` | 日志级别 | `DEBUG/INFO/WARNING` |
| `HF_TOKEN` | HuggingFace token | `hf_xxx` |
| `HF_ENDPOINT` | HuggingFace 镜像 | `https://hf-mirror.com` |
| `VLLM_API_KEY` | API 密钥 | `sk-xxx` |
| `NCCL_SOCKET_IFNAME` | NCCL 网络接口 | `eth0` |
