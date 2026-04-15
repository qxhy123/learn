# 第19章：张量并行

> 当前仓库里讲张量并行，应该先看 `docs/serving/parallelism_scaling.md`、V1 executor，以及模型实现里的并行线性层。真正需要掌握的不是一句“把矩阵切开”，而是 `tensor_parallel_size` 如何同时改变模型参数切分、KV cache 形状和 worker 进程拓扑。

---

## 学习目标

学完本章，你将能够：

1. 用当前 vLLM 文档的决策逻辑判断何时该用张量并行
2. 理解张量并行在模型层、KV cache 和 executor 中的真实落点
3. 识别当前源码里 attention head / KV head 的切分与复制规则
4. 正确配置 `tensor_parallel_size`，并用日志验证是否真的得到收益
5. 识别旧教程里关于 TP 的几个典型过时说法

---

## 19.1 先用当前官方策略判断：什么时候该用 TP？

`vllm/docs/serving/parallelism_scaling.md` 给出的当前建议非常直接：

1. **模型能放进单卡**：先不要上分布式
2. **单卡放不下，但单机多卡能放下**：优先考虑张量并行
3. **单机也放不下**：再把张量并行和流水线并行组合起来

这和很多旧文章最大的区别是：

- TP 不是“多卡默认最佳实践”
- TP 是**单副本模型**在单机多卡上的首选扩展路径
- 一旦跨节点，官方就建议把 `tensor_parallel_size` 和 `pipeline_parallel_size` 一起规划

### 当前文档里一个很重要的边界条件

官方文档专门指出了两个容易被忽略的场景：

1. **GPU 数量和模型切分不整齐**
2. **机器没有 NVLink，例如 L40S**

这两种情况下，`pipeline_parallel_size` 可能比盲目增大 `tensor_parallel_size` 更合适。也就是说，今天的 vLLM 文档已经不再把 TP 讲成唯一答案，而是把它放回硬件拓扑约束里讨论。

---

## 19.2 源码里，TP 不只是一行配置

很多入门文章会把 TP 描述成：

```text
设置 tensor_parallel_size=4
→ 权重自动切成 4 份
→ 结束
```

这在当前仓库里过于粗糙。真实路径至少分成三层：

### 第一层：配置层

入口参数来自：

- Python 侧 `vllm/vllm/entrypoints/llm.py`
- CLI 侧 `vllm/vllm/engine/arg_utils.py`

`tensor_parallel_size` 会进入 `ParallelConfig`，再影响 V1 engine 的 world size 计算。

### 第二层：executor 层

在 `vllm/vllm/v1/executor/multiproc_executor.py` 里，初始化时会校验：

```text
world_size = tensor_parallel_size × pipeline_parallel_size × prefill_context_parallel_size
```

也就是说，TP 不是只作用于模型结构，它还决定：

- 要拉起多少 worker
- 每个 worker 的 global rank / local rank
- 分布式通信组如何初始化

单机场景默认常见的是：

- 单卡：`UniProcExecutor`
- 多卡：`MultiprocExecutor`

这也是为什么 TP 的性能问题，常常既可能出在模型层，也可能出在 executor 和通信初始化层。

### 第三层：模型层

真正的参数切分发生在各模型实现里。以 `vllm/vllm/model_executor/models/llama.py` 为代表，可以看到：

- 通过 `get_tensor_model_parallel_world_size()` 读取 TP world size
- attention 使用 `QKVParallelLinear`
- 输出投影使用 `RowParallelLinear`
- MLP 里使用 `MergedColumnParallelLinear` 和 `RowParallelLinear`

这说明当前 vLLM 的 TP 不是“外部魔法”，而是各模型实现显式使用并行层完成的。

---

## 19.3 Attention 层到底是怎么按 TP 切的？

还是看 `llama.py` 这种当前主线实现，能得到几个关键事实。

### 1. Query heads 按 TP 均分

模型里通常会先拿到：

- `total_num_heads`
- `tp_size`

然后做：

- `num_heads = total_num_heads // tp_size`

所以对普通 decoder-only 模型来说，attention 头确实是“每个 rank 负责一部分”。

### 2. KV heads 不一定总是均分

这是很多旧教程最容易讲错的地方。

在当前实现里，KV head 有两种情况：

1. **`total_num_kv_heads >= tp_size`**
   这时 KV heads 会被真正切开
2. **`total_num_kv_heads < tp_size`**
   这时 KV heads 会在多个 TP rank 上复制

这正是 `Grouped Query Attention` / `Multi-Query Attention` 场景里必须注意的细节。

### 3. 因此，“KV cache 一定缩小为原来的 1/TP”并不总成立

如果是传统 MHA，KV heads 往往能跟着 TP 一起切。

但如果模型本身 KV heads 就少，或者用了 MLA / GQA 一类结构，那么：

- Query 计算仍然可以做 TP
- KV cache 却未必能线性随 TP 缩小

所以今天评估 TP，不能只看参数显存，还要看**KV 头布局**。

---

## 19.4 KV Cache 与 TP 的关系：当前仓库里该怎么理解？

从源码和官方文档一起看，更准确的说法应该是：

### 情况 A：KV heads 足够多

如果 `num_kv_heads` 能被 TP 正常切分，那么每个 rank 只保存自己那部分 KV：

- 单 rank 的 KV cache 压力下降
- 更容易把更多 token 留在 GPU cache 中

### 情况 B：KV heads 太少

如果 KV heads 少于 TP size，那么部分 KV 会复制到多个 rank：

- 参数层面仍然能做 TP
- 但 KV cache 不能再按 `1 / tp_size` 的理想比例下降

这也是 `docs/serving/context_parallel_deployment.md` 还要继续讨论 DCP 的原因：单靠 TP 并不能解决所有长上下文 KV 压力问题。

### 结合日志来验证，而不是靠脑补

`docs/serving/parallelism_scaling.md` 明确建议启动后关注两条日志：

```text
GPU KV cache size: ...
Maximum concurrency for ... tokens per request: ...
```

这两条日志比“我把 TP 从 2 改成 4 了，所以并发一定翻倍”更可信。

---

## 19.5 当前 vLLM 里的 TP 配置方式

### Python 离线推理

```python
from vllm import LLM

llm = LLM(
    model="facebook/opt-13b",
    tensor_parallel_size=4,
)
```

### CLI 服务模式

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve facebook/opt-13b \
    --tensor-parallel-size 4
```

### 显式指定 executor backend

当前官方文档给出的默认路线是：

- **单机**：`multiprocessing`
- **多节点**：`ray`

必要时你也可以显式指定：

```bash
vllm serve facebook/opt-13b \
    --tensor-parallel-size 4 \
    --distributed-executor-backend mp
```

对于单机 TP，这通常已经足够。

---

## 19.6 选 TP 大小时，当前更靠谱的经验法则

### 原则 1：只开到“刚好能放下”

TP 越大：

- 单 rank 参数越少
- 但通信越多

所以当前更稳妥的策略不是“卡越多越好”，而是：

```text
找到能让模型 + KV cache 放下的最小 TP
```

### 原则 2：尽量把 TP 限制在单机、尤其是 NVLink 域内

官方文档非常明确：

- 单机多卡且模型能放下：优先 TP
- 没有 NVLink：PP 可能更好
- 跨节点：通常 TP + PP 组合，而不是单纯把 TP 拉满到全 cluster

### 原则 3：别把旧教材里的固定通信公式当成真理

旧资料常说：

```text
每层固定两次 AllReduce
```

这只能当作早期 Megatron 风格实现的入门心智模型。
在当前仓库里，真实行为还会受这些因素影响：

- 模型结构是否标准 Transformer
- 是否是 GQA / MLA / MoE / hybrid model
- 使用的线性层、注意力后端和通信后端
- 是否开启了自定义 all-reduce 或其他优化路径

所以这句话可以帮助建立直觉，但不能拿来做精确性能预估。

---

## 19.7 当前章节最值得记住的源码锚点

| 主题 | 当前文件 |
|------|----------|
| 分布式扩展总说明 | `vllm/docs/serving/parallelism_scaling.md` |
| 单进程 executor | `vllm/vllm/v1/executor/uniproc_executor.py` |
| 多进程 executor | `vllm/vllm/v1/executor/multiproc_executor.py` |
| TP 参数入口 | `vllm/vllm/engine/arg_utils.py` |
| LLaMA 的 TP 实现示例 | `vllm/vllm/model_executor/models/llama.py` |
| Qwen2 的 PP/TP 兼容实现 | `vllm/vllm/model_executor/models/qwen2.py` |
| 长上下文下 TP 的边界 | `vllm/docs/serving/context_parallel_deployment.md` |

---

## 本章小结

| 结论 | 当前仓库里的正确理解 |
|------|--------------------|
| TP 什么时候用 | 模型放不下单卡，但能放进单机多卡时优先考虑 |
| TP 改变了什么 | 不只是权重切分，还改变 executor world size 和 worker 拓扑 |
| attention 怎么切 | Query heads 通常均分；KV heads 可能切分，也可能复制 |
| KV cache 会怎样 | 不一定严格变成原来的 `1 / tp_size` |
| 什么时候别盲目加 TP | 无 NVLink、切分不整齐、跨节点通信成本高时 |

---

## 动手实验

### 实验 1：观察 TP 对 KV cache 容量的影响

在同一台机器上分别用 `--tensor-parallel-size 1` 和 `2` 启动同一个模型，记录：

- `GPU KV cache size`
- `Maximum concurrency`

比较它们是否按你的直觉变化。如果没有线性变化，检查模型是否使用了较少的 KV heads。

### 实验 2：对比 TP 和 PP

如果你的机器没有 NVLink，尝试对同一个模型分别测试：

```bash
--tensor-parallel-size 4 --pipeline-parallel-size 1
```

和

```bash
--tensor-parallel-size 1 --pipeline-parallel-size 4
```

对比吞吐和延迟，验证官方文档里“无 NVLink 时 PP 可能更优”的提示。

---

## 练习题

### 基础题

1. 在当前仓库里，`tensor_parallel_size` 会同时影响哪两大类对象？
2. 为什么 `num_kv_heads < tp_size` 时，不能简单认为 KV cache 一定缩小为原来的 `1 / tp_size`？

### 思考题

3. 官方文档为什么会在“单机也能放下模型”的情况下，仍然建议你 benchmark `TP=1, PP=<GPU数>`？
4. 如果你把 TP 从 2 调到 4，最应该先看哪两条日志来确认这次改动是不是值得？
