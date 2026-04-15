# 第20章：流水线并行

> 旧教程常把流水线并行讲成“训练里的 micro-batch 技术移植到推理”。这在当前 vLLM 里太宽泛了。更准确的说法是：`pipeline_parallel_size` 让模型按层分段执行，而当前源码还额外要求模型类满足 `supports_pp` 接口契约，否则根本不能稳定跑起来。

---

## 学习目标

学完本章，你将能够：

1. 理解当前 vLLM 中流水线并行的真实适用场景
2. 知道 PP 在源码里依赖哪些模型接口，而不是把它当成“所有模型自动支持”
3. 读懂 `supports_pp`、`intermediate_tensors` 和 `PPMissingLayer` 的作用
4. 正确组合 `pipeline_parallel_size`、`tensor_parallel_size` 和 executor backend
5. 区分当前 vLLM 的 PP 心智模型与很多旧训练教程的差异

---

## 20.1 当前官方文档怎么定位 PP？

`vllm/docs/serving/parallelism_scaling.md` 给 PP 的定位很清楚：

1. **单机放不下模型**：TP + PP 组合
2. **GPU 切分不均匀**：可考虑 `tensor_parallel_size=1, pipeline_parallel_size=<GPU数>`
3. **没有 NVLink**：PP 可能比 TP 更划算

所以当前 vLLM 里的 PP，不只是“跨节点时才用”，也不是“训练才需要”。它在推理里主要解决两个工程问题：

- **单副本模型过大，单机塞不下**
- **硬件拓扑不适合重通信的 TP**

---

## 20.2 在当前源码里，PP 首先是一个模型接口契约

很多人第一次读 vLLM 会忽略这一点：**不是所有模型都天然支持 PP**。

`vllm/vllm/model_executor/models/interfaces.py` 里，`supports_pp(...)` 会检查三件事：

1. 模型是否声明 `supports_pp=True`
2. 是否实现 `make_empty_intermediate_tensors(...)`
3. `forward(...)` 是否接受 `intermediate_tensors`

也就是说，当前仓库里的 PP 不是简单“把层均分给多个 GPU”就结束。模型类必须显式告诉引擎：

- 中间激活怎么在 stage 间流转
- 非首段 / 非末段 rank 的 forward 如何工作
- 最后一段什么时候才真正产出 logits

### 这也是为什么旧资料容易误导

旧教程常把 PP 讲成：

```text
按层切
→ stage 之间传 hidden states
→ 完成
```

但在当前 vLLM 里，真正决定“这个模型能不能做 PP”的，是模型类有没有遵守上面的接口契约。

---

## 20.3 `Qwen2ForCausalLM` 是一个很好的当前范例

在 `vllm/vllm/model_executor/models/qwen2.py` 里，可以看到一个典型的 PP 兼容实现：

- 类本身混入了 `SupportsPP`
- `forward(...)` 接收 `intermediate_tensors`
- `self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors`
- 只有最后一个 PP rank 才保留 `lm_head`
- 非最后一段使用 `PPMissingLayer()`

这背后的设计含义是：

### 1. 非最后一段不负责产出最终 logits

如果某个 rank 不是 pipeline 的最后一段，它只需要继续传递中间张量，不需要持有完整的 `lm_head`。

### 2. 中间态是显式对象，不是隐式全局状态

这正是 `intermediate_tensors` 的价值：

- stage 之间传的不是“黑盒激活”
- 而是模型明确声明、可以继续消费的中间结果

### 3. PP 支持是模型级能力

因此你在做自定义模型接入时，如果希望它支持 PP，就必须一起设计：

- 中间张量结构
- stage 边界
- 最后一段的输出逻辑

这在第 27 章会再次出现。

---

## 20.4 executor 如何配合 PP？

### `MultiprocExecutor` 明确声明支持 PP

`vllm/vllm/v1/executor/multiproc_executor.py` 里有一行很关键：

```text
supports_pp: bool = True
```

再结合 `vllm/vllm/engine/arg_utils.py` 中的 `_check_feature_supported()`，可以得到当前规则：

- 如果 `pipeline_parallel_size > 1`
- 需要 backend 本身支持 PP
- `mp`、`ray`、`external_launcher` 都是当前认可的路径

### world size 不是只看 PP

当前 executor 的 world size 会一起考虑：

```text
world_size = tensor_parallel_size × pipeline_parallel_size × prefill_context_parallel_size
```

这意味着：

- `PP=2, TP=4` 不是“4 卡 + 再做 2 段”
- 而是**总共 8 个并行 rank**

这是部署时最容易算错的地方之一。

---

## 20.5 当前 vLLM 的 PP 运行形态

### 单机 PP

如果模型能放在单机，但你不想做重通信 TP，可以直接：

```bash
vllm serve <model> \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 4
```

这种方式特别适合：

- 无 NVLink 的多卡机器
- 模型按层切比按张量切更自然的场景

### TP + PP 组合

```bash
vllm serve <model> \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2
```

这种组合是当前官方文档推荐的多节点单副本常见形态：

- 节点内做 TP
- 节点间做 PP

### Python 离线推理

`LLM(...)` 也可以通过额外参数传入：

```python
from vllm import LLM

llm = LLM(
    model="facebook/opt-13b",
    tensor_parallel_size=2,
    pipeline_parallel_size=2,
)
```

不过只要进入多机多进程场景，CLI `serve` 路径通常更直观。

---

## 20.6 当前文档为什么不再强调“micro-batch 调流水线”？

很多训练教材一讲 PP，就会展开：

- micro-batch 数
- pipeline bubble 公式
- 1F1B 调度

这些知识有价值，但当前 vLLM 的服务文档没有把它们作为第一视角，原因很简单：

### 在推理服务里，用户首先关心的不是训练式调度公式

而是：

1. 模型能不能被切成多个 stage
2. executor backend 支不支持
3. 当前硬件上 TP 还是 PP 更划算
4. 最后能否换来更好的显存容量与吞吐

所以当前仓库里读 PP，优先级应该是：

```text
模型接口
→ executor 支持
→ 节点拓扑
→ 再去想 bubble 和 stage balance
```

而不是反过来。

---

## 20.7 什么时候优先考虑 PP？

### 场景 1：模型大到单机塞不下

这是最经典场景。当前官方建议就是：

- 单机内：先看 TP
- 超出单机：TP + PP

### 场景 2：GPU 间没有 NVLink

官方文档明确写了一个很实用的经验：

- 如果像 L40S 这种机器没有 NVLink
- 那么 PP 可能拥有更低通信开销

### 场景 3：模型切分不均匀

当模型结构、显存预算、GPU 数量不整齐时，PP 这种按层切分的方式更灵活。

---

## 20.8 读当前源码时，PP 该看哪些文件？

| 主题 | 当前文件 |
|------|----------|
| 并行策略总说明 | `vllm/docs/serving/parallelism_scaling.md` |
| PP 接口定义 | `vllm/vllm/model_executor/models/interfaces.py` |
| Qwen2 的 PP 兼容实现 | `vllm/vllm/model_executor/models/qwen2.py` |
| LLaMA / 其他模型的并行层实现 | `vllm/vllm/model_executor/models/*.py` |
| PP backend 校验 | `vllm/vllm/engine/arg_utils.py` |
| 多进程 executor | `vllm/vllm/v1/executor/multiproc_executor.py` |
| 多节点 headless 路径 | `vllm/vllm/entrypoints/cli/serve.py` |

---

## 本章小结

| 结论 | 当前仓库里的正确理解 |
|------|--------------------|
| PP 是什么 | 按层分段执行模型，而不是只靠一个抽象概念 |
| PP 是否自动支持所有模型 | 不是，模型类要满足 `supports_pp` 契约 |
| 关键接口 | `supports_pp=True`、`make_empty_intermediate_tensors`、`forward(..., intermediate_tensors=...)` |
| backend 约束 | 当前主线是 `mp`、`ray`、`external_launcher` |
| 适用场景 | 超大模型、无 NVLink、多节点单副本、切分不均匀 |

---

## 动手实验

### 实验 1：确认一个模型是否支持 PP

找一个当前仓库里的模型实现文件，例如：

- `qwen2.py`
- `llama.py`

检查它是否：

- 声明 `SupportsPP`
- 实现 `make_empty_intermediate_tensors`
- 在 `forward` 中接收 `intermediate_tensors`

### 实验 2：单机对比 TP 与 PP

在 4 卡、无 NVLink 机器上，对同一个模型分别测试：

```bash
--tensor-parallel-size 4 --pipeline-parallel-size 1
```

和

```bash
--tensor-parallel-size 1 --pipeline-parallel-size 4
```

观察吞吐、首 token 延迟和 GPU 利用率。

---

## 练习题

### 基础题

1. 在当前仓库里，一个模型若想支持 PP，至少要满足哪三个接口条件？
2. 为什么说 PP 不是“所有模型自动支持”的能力？

### 思考题

3. 单节点没有 NVLink 时，为什么官方文档会建议把 PP 也纳入 benchmark？
4. `TP=4, PP=2` 在当前 V1 executor 里意味着多少个并行 rank？为什么这件事在部署时很重要？
