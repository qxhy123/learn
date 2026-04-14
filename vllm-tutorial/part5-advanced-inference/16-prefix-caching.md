# 第16章：前缀缓存与 Prompt 复用

> “相同前缀只算一次”这句话本身并不新，但当前仓库里 vLLM V1 对它的实现已经非常工程化：按完整 block 做 hash，命中后直接接回请求的 KV block 链，并通过 free queue + LRU + `cache_salt` 解决复用、逐出和隔离问题。

---

## 学习目标

学完本章，你将能够：

1. 理解当前仓库里前缀缓存的真实命中粒度与 hash 规则
2. 结合源码理解 prefix cache 如何嵌入调度与内存管理
3. 使用 `enable_prefix_caching`、`cache_salt`、hash 算法配置等真实接口
4. 识别哪些请求会天然绕过 prefix cache
5. 分析命中率、TTFT 和 KV 容量之间的关系

---

## 16.1 为什么前缀缓存几乎是“白赚”的优化？

在真实业务里，很多请求都长这样：

```text
[系统提示词 / few-shot 示例 / 文档上下文] + [每个用户自己的尾部问题]
```

如果每次都从头 prefill：

- GPU 会重复计算同一段 prompt
- KV Cache 会重复写入同样的前缀块
- TTFT 会被长前缀拖慢

而前缀缓存做的事情很直接：

- 把“已经完整算过的前缀 block”缓存起来
- 下一次同样前缀出现时直接复用已有 KV block

它不会改变输出分布，因此在大多数场景下属于非常值得优先启用的优化。

---

## 16.2 当前仓库里，前缀缓存到底按什么命中？

### 不是按字符串，也不是按任意 token 前缀

当前仓库的 prefix caching 是：

- **按 block 粒度**
- **按完整 block**
- **按从开头开始的最长公共前缀**

工作的最小单位不是“一个 token”，而是“一个完整的 KV block”。

### 当前 block hash 的组成

官方设计文档 `vllm/docs/design/prefix_caching.md` 明确说明，一个 block 的 hash 不是只看当前块内 token，而是大致包含：

1. 父 block 的 hash
2. 当前 block 的 token
3. 额外哈希信息

可以概括成：

```text
block_hash = hash(parent_hash, block_tokens, extra_hashes)
```

这样设计的意义是：

- 当前块内容相同，但前缀不同，不会误命中
- 多模态输入、LoRA、tenant 隔离等都可以进入 hash

### 什么是 `extra_hashes`？

结合文档和源码，它可能包括：

- LoRA ID
- 多模态输入 hash
- `cache_salt`

所以前缀缓存并不只是“文本前缀相同就命中”，而是“在当前执行上下文下可安全复用时才命中”。

---

## 16.3 为什么只缓存完整 block？

这是理解命中率上限最关键的一点。

当前仓库明确规定：

- **只有完整 block 才会进入 prefix cache**

假设 `block_size=16`：

```text
请求 A:
  前 32 个 token 正好是两个完整 block
  第 33~40 个 token 只填了半个 block

那可缓存的只有前两个完整 block
第 3 个半满 block 不会作为可复用前缀命中
```

这会带来两个直接后果：

1. 前缀缓存命中不是“逐 token 线性增长”，而是阶梯式增长
2. 相同 prompt 即使看起来几乎完全相同，最后半块也可能仍要重算

---

## 16.4 当前 V1 里前缀缓存如何和调度器配合？

前缀缓存不是在调度外面“包一层 memoization”，它直接进入了 waiting 请求的接纳路径。

### 第一步：请求对象先准备 block hashes

`vllm/vllm/v1/request.py` 中，`Request` 会在 token 序列变化后更新自己的：

- `block_hashes`

### 第二步：waiting 队列调度时先查命中

在 `vllm/vllm/v1/core/sched/scheduler.py` 中，新请求进入 waiting 路径时，会先调用：

```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
```

这一步会：

- 找出已经命中的完整前缀块
- 统计这些块对应了多少“已计算 token”

### 第三步：`allocate_slots()` 把命中的块接到请求上

随后 `KVCacheManager.allocate_slots(...)` 会：

1. 把命中的 `new_computed_blocks` 并入请求 block 列表
2. 只为真正还没算过的 token 申请新块
3. 如果新的 block 之后被填满，再把它们写回 cache

这就是为什么“前缀缓存命中”本质上会直接降低：

- 本轮调度需要发出去的 token 数
- KV 新分配的块数

---

## 16.5 当前实现里的几个重要细节

### 细节 1：即使全部命中，最后也常常要重算一点

`KVCacheManager.get_computed_blocks()` 里有个重要限制：

- 当整段 prompt 都命中时
- 最后一个 token 往往仍需要重新计算
- 这样才能拿到当前请求所需的 logits

所以“100% 命中”在实现层面也不总是等于“0 token prefill”。

### 细节 2：`prompt_logprobs` 会绕过 prefix cache 读取

当前仓库里，`SamplingParams` 会在某些场景下设置：

- `skip_reading_prefix_cache=True`

最典型的情况就是：

- 你请求 `prompt_logprobs`

因为这时系统需要重新 prefill 整段 prompt 来生成对应 logprobs，而不是简单复用已有 KV。

### 细节 3：多模态前缀缓存会把媒体 hash 也算进去

官方设计文档举了多模态例子：

- 图片会被映射成 placeholder token
- 但真正参与 block hash 的还包括图片内容 hash

这样才能避免：

- 文本模板相同
- 占位符数量相同
- 但图片不同

时发生危险的误复用。

### 细节 4：V1 的 block table 偏 append-only

当前设计里，一个新请求如果又生成出一个与旧缓存 block 完全相同的完整块，短时间内可能出现“重复缓存块”并存。
这不是 bug，而是 V1 append-only block table 的一个实现取舍，通常会在请求释放后清理。

---

## 16.6 如何启用当前仓库里的前缀缓存？

### 离线推理

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enable_prefix_caching=True,
)

params = SamplingParams(temperature=0, max_tokens=64)

outputs = llm.generate(
    [
        "System: You are helpful.\nUser: Hello!",
        "System: You are helpful.\nUser: What is AI?",
    ],
    sampling_params=params,
)
```

### 服务模式

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-prefix-caching
```

### 选择 hash 算法

当前仓库支持通过：

```bash
--prefix-caching-hash-algo
```

配置 hash 算法，官方文档列出的选项包括：

- `sha256`
- `sha256_cbor`
- `xxhash`
- `xxhash_cbor`

一般来说：

- 想要更稳妥、隔离性更强，优先 `sha256*`
- 想要更快但接受更高碰撞风险，再考虑 `xxhash*`

---

## 16.7 `cache_salt`：多租户环境下怎么隔离缓存？

前缀缓存一旦共享，就会有一个自然问题：

> 不同租户之间，能不能通过命中时间差推断彼此 prompt？

当前仓库给出的应对方式之一是：

- 在请求里设置 `cache_salt`

这样它会进入 block hash，只有 salt 相同的请求才会共享前缀缓存。

示意：

```python
extra_body = {
    "cache_salt": "team-a"
}
```

这适合：

- 同组织内部允许共享缓存
- 但不同组织之间需要隔离

的场景。

---

## 16.8 性能收益该怎么看？

### 对 TTFT 的影响最直接

命中的前缀越长：

- 需要真正 prefill 的 token 越少
- TTFT 通常下降越明显

尤其是：

- 系统 prompt 很长
- few-shot 示例很多
- 文档上下文固定

时最划算。

### 对吞吐的影响是间接的

命中前缀后：

- GPU 少做重复 prefill
- KV block 分配压力下降
- 调度器更容易在同样容量下接更多请求

所以你通常会同时看到：

- TTFT 改善
- 整体并发承载能力也提高

### 但命中率不该只看“前缀长度”

你还要看：

- 前缀是否在 block 边界上对齐
- 是否有动态字段插在最前面
- 是否用了不同 `cache_salt`
- 是否带 prompt logprobs

---

## 16.9 怎么设计 prompt 才更容易命中？

### 推荐做法

```text
[系统提示词]
[共享 few-shot 示例]
[共享文档上下文]
[用户差异内容]
```

### 不推荐做法

```text
[当前时间 / request_id / trace_id / 每次变化字段]
[系统提示词]
[共享上下文]
[用户问题]
```

因为只要一开始就变了，最长公共前缀就会立刻断掉。

一个简单原则：

> 越稳定、越长、越值得缓存的部分，越应该放在 prompt 最前面。

---

## 16.10 源码对照：本章该看哪些文件？

| 主题 | 关键文件 | 重点 |
|------|----------|------|
| 请求如何维护 block hash | `vllm/vllm/v1/request.py` | `update_block_hashes()` |
| prefix cache 查询 | `vllm/vllm/v1/core/kv_cache_manager.py` | `get_computed_blocks()` |
| block 分配与缓存写回 | `vllm/vllm/v1/core/kv_cache_manager.py` | `allocate_slots()` |
| block 元数据 / free queue | `vllm/vllm/v1/core/kv_cache_utils.py` | `KVCacheBlock`、LRU |
| 设计文档 | `vllm/docs/design/prefix_caching.md` | hash 规则、逐出、隔离 |

---

## 本章小结

| 概念 | 当前仓库中的真实语义 |
|------|----------------------|
| 命中粒度 | 完整 block |
| hash 组成 | 父 hash + 当前 block token + extra hashes |
| 调度结合点 | waiting 请求接纳前先查 `get_computed_blocks()` |
| 安全隔离 | `cache_salt` |
| 常见绕过场景 | `prompt_logprobs` 等需要重算 prompt 的请求 |

---

## 练习题

### 基础题

1. 为什么 prefix cache 只能命中从开头开始的最长公共前缀？
2. 为什么只缓存完整 block？
3. `cache_salt` 解决了什么问题？

### 实践题

4. 在 `kv_cache_manager.py` 中找到 `get_computed_blocks()`，确认它在什么条件下会直接返回“无命中”。
5. 打开 `docs/design/prefix_caching.md`，总结 block hash 的组成部分。

### 思考题

6. 如果你的系统 prompt 很长，但前面总会插入一个动态时间戳，为什么前缀缓存几乎等于没开？
7. 如果 TTFT 仍然很高，你该如何判断问题是“前缀缓存没命中”，还是“KV 容量已经打满导致调度拥塞”？
