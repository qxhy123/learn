# 第27章：自定义模型接入

> 当前仓库里讲“自定义模型接入”，主线已经不是“去改 `vllm/model_executor/models/__init__.py` 里的注册表”。真正的主线是：Hugging Face 配置如何被读取，`architectures` 如何映射到 `ModelRegistry`，以及你什么时候该写 out-of-tree plugin，什么时候必须进一步写 platform plugin。

---

## 学习目标

学完本章，你将能够：

1. 说清楚一个模型从 `config.json` 到 `ModelRegistry` 再到真实类加载的当前调用链
2. 区分“已有模型直接跑”“`trust_remote_code` 兼容”“model plugin”“platform plugin”这四条路径
3. 理解当前 vLLM-native 模型类的基本接口约定
4. 用 `ModelRegistry.register_model(...)` 以 out-of-tree 方式注册新架构
5. 识别旧教程里关于 `_MODELS` 直改注册和 `trust_remote_code` 的两个常见误区

---

## 27.1 先校准：当前模型解析链路是怎样的？

结合 `docs/design/huggingface_integration.md`、`vllm/vllm/transformers_utils/config.py` 和 `vllm/vllm/model_executor/models/registry.py`，当前链路可以概括成：

```text
model=/path/or/hf_repo
  ↓
读取 config.json / params.json
  ↓
必要时通过 AutoConfig.from_pretrained(...)
  ↓
得到 architectures
  ↓
ModelConfig 初始化时调用 registry.inspect_model_cls(...)
  ↓
registry.resolve_model_cls(...)
  ↓
worker / model loader 真正实例化模型并加载权重
```

### 这里有两个当前版本的关键点

#### 1. `architectures` 是真正的模型分发键

在 `vllm/vllm/config/model.py` 里，`ModelConfig` 初始化时会调用 registry 去检查：

- 这是不是文本生成模型
- 是不是 pooling 模型
- 支不支持 PP
- 最终该解析成哪个 architecture

所以你接入自定义模型时，真正需要对齐的是：

```text
HF 配置里的 architectures
↔
vLLM 的 ModelRegistry
```

#### 2. 当前 registry 已经不只是一个静态字典

`vllm/vllm/model_executor/models/registry.py` 里的 `ModelRegistry` 现在支持：

- 直接加载内置模型
- lazy import 外部模型
- 在必要时 fallback 到 transformers backend
- 在子进程里做 model inspect，避免主进程误初始化 CUDA

这已经和早期“`_MODELS = {...}` 静态映射”不是同一个复杂度了。

---

## 27.2 四条接入路径：先别急着写新模型文件

当前更实用的做法是，先判断你到底属于哪一类问题。

| 场景 | 当前更合适的路径 |
|------|------------------|
| 架构已经被 vLLM 原生支持 | 直接加载，不需要接入 |
| HF 端有自定义 config / class，但 vLLM 或 transformers 已能识别 | 先尝试 `trust_remote_code=True` |
| 你要新增一个 out-of-tree 模型架构 | 写 **model plugin**，用 `ModelRegistry.register_model(...)` |
| 你要新增一个硬件平台、worker、attention backend 或设备通信后端 | 写 **platform plugin** |

### 这张表很重要

很多“自定义模型接入失败”的根因，不是代码没写完，而是一开始就选错了扩展层级：

- 只是架构没注册，却跑去改 worker
- 只是 HF config 没被识别，却误以为要改 attention kernel
- 明明需要新平台，却只写了模型类

---

## 27.3 `trust_remote_code=True` 在当前仓库里到底能做什么？

从 `vllm/vllm/transformers_utils/config.py` 和官方设计文档看，`trust_remote_code=True` 的作用主要是：

1. 允许 `AutoConfig.from_pretrained(...)` 加载模型仓库里的自定义配置类
2. 必要时通过 `auto_map` 动态导入 HF 侧模块
3. 帮 vLLM 拿到更完整的 HF config / transformers model class 信息

### 但它不能替你做三件事

#### 1. 它不会自动给你生成 vLLM-native 模型实现

也就是说，它不会自动替你补出：

- `load_weights(...)`
- 并行线性层替换
- paged attention 路径
- `SupportsPP` 所需接口

#### 2. 它不会自动把新 architecture 注册进 `ModelRegistry`

如果最终 `architectures` 对不上 registry，vLLM 仍然会报：

- 不支持
- 或提示你该安装对应 plugin

#### 3. 它不会替你解决新硬件 / 新后端问题

如果你要跑的是一个新平台，那问题不在 HF 代码信任，而在：

- worker
- platform
- attention backend
- device communicator

### 当前更准确的理解

`trust_remote_code=True` 是：

```text
让 vLLM 更容易读懂 HF 侧的自定义配置/类
```

而不是：

```text
自动完成自定义模型接入
```

---

## 27.4 out-of-tree 模型接入：当前推荐主线是 plugin system

`vllm/docs/design/plugin_system.md` 已经把推荐路径写得很明确：

- 使用 Python `entry_points`
- 入口组使用 `vllm.general_plugins`
- 在 plugin 函数里调用 `ModelRegistry.register_model(...)`

### 一个最小可工作的注册示意

```python
# setup.py
from setuptools import setup

setup(
    name="vllm_add_my_model",
    version="0.1.0",
    packages=["vllm_add_my_model"],
    entry_points={
        "vllm.general_plugins": [
            "register_my_model = vllm_add_my_model:register",
        ]
    },
)
```

```python
# vllm_add_my_model/__init__.py
def register():
    from vllm import ModelRegistry

    if "MyModelForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "MyModelForCausalLM",
            "vllm_add_my_model.my_model:MyModelForCausalLM",
        )
```

### 为什么这里推荐传字符串，而不是直接传类对象？

`ModelRegistry.register_model(...)` 当前允许两种形式：

1. 直接传 `nn.Module` 类
2. 传 `"<module>:<class>"` 字符串

第二种在 plugin 场景里更常用，因为 `registry.py` 明确提到：

- lazy import 可以避免过早导入模型模块
- 也更容易规避 “fork 后重新初始化 CUDA” 一类问题

---

## 27.5 plugin 为什么必须是“可重入”的？

`docs/design/plugin_system.md` 特别强调 plugin 函数要 **re-entrant**，这是因为当前 vLLM 会在多个进程里加载 plugin。

从源码看，至少这几个地方都会触发 general plugins：

- `vllm/vllm/model_executor/models/registry.py`
- `vllm/vllm/v1/worker/worker_base.py`

这背后的原因很简单：

- 主进程要知道 registry 里有哪些模型
- worker 进程也要知道
- registry inspect 的子进程也要知道

所以一个安全的 plugin 注册函数通常应该：

1. 先检查 architecture 是否已存在
2. 已存在则直接返回
3. 不要依赖只执行一次的全局副作用

---

## 27.6 当前 vLLM-native 模型类长什么样？

如果你去看当前主线模型，例如：

- `vllm/vllm/model_executor/models/qwen2.py`
- `vllm/vllm/model_executor/models/llama.py`

会发现模型类已经比较统一。

### 典型构造函数

当前常见风格是：

```python
def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
    ...
```

这意味着模型实现依赖的核心上下文，不再只是一个 HF config，而是完整的：

- `vllm_config.model_config`
- `vllm_config.quant_config`
- cache / parallel / compilation 等配置

### 典型 forward 形态

对于当前主线 decoder-only 模型，常见形式是：

- `input_ids`
- `positions`
- 可选的 `intermediate_tensors`
- 可选的 `inputs_embeds`

如果模型支持 PP，`intermediate_tensors` 就不是可有可无的装饰，而是接口契约的一部分。

### 典型权重加载形态

当前模型一般会实现：

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    ...
```

并通过这些手段之一完成映射：

- `AutoWeightsLoader`
- `stacked_params_mapping`
- 自定义参数名前缀转换

所以今天的“接入模型”本质上是在做三件事：

1. **搭模型骨架**
2. **声明并行/PP 能力**
3. **把 HF 权重名映射到 vLLM 参数名**

---

## 27.7 如果你的模型还想支持 PP，需要额外满足什么？

当前 `vllm/vllm/model_executor/models/interfaces.py` 已经把要求说得很清楚：

1. 设置 `supports_pp=True`
2. 实现 `make_empty_intermediate_tensors(...)`
3. `forward(...)` 接收 `intermediate_tensors`

这也是为什么很多“我先跑单卡成功了”的模型，一开 `pipeline_parallel_size > 1` 就暴露问题。

### 一个当前实现里的典型模式

像 `Qwen2ForCausalLM` 这种实现会：

- 把 `make_empty_intermediate_tensors` 委托给内部 backbone
- 在非最后一段用 `PPMissingLayer()`
- 只有最后一段才真正持有 `lm_head`

这说明：

```text
PP 支持不是额外打一行 flag
而是模型结构设计的一部分
```

---

## 27.8 如果你的问题不只是“新模型”，而是“新平台”，那就该写 platform plugin

这也是当前 plugin 文档非常值得重视的地方。

如果你要扩展的是：

- 新硬件平台
- 新 worker
- 新 attention backend
- 新 device communicator
- 新 custom ops

那么只写 model plugin 不够。

`docs/design/plugin_system.md` 给出的 platform plugin 设计里，核心扩展点包括：

- `Platform`
- `WorkerBase`
- `AttentionBackend`
- `DeviceCommunicatorBase`

### 什么时候你会走到这一步？

例如：

1. 你的设备不是当前内置平台
2. 你需要自己的 attention kernel
3. 你需要自己的 all-reduce / all-gather 实现
4. 你要在新平台上启用 graph mode、spec decode、LoRA 等能力

这时问题已经不是“模型接入”，而是“平台接入”。

---

## 27.9 V1 guide 给出的当前生态提示

`vllm/docs/usage/v1_guide.md` 里已经给出几个很重要的现实结论：

### 1. 非原生支持的 encoder-decoder 模型，当前推荐走 plugin

文档明确写到：

- Whisper 原生支持
- 其他 encoder-decoder 模型建议通过 plugin system 支持
- 官方给了 `bart-plugin` 作为参考

### 2. 更多硬件也在通过 plugin 生态扩展

V1 guide 直接列出了几类平台插件生态，例如：

- `vllm-ascend`
- `vllm-spyre`
- `vllm-gaudi`
- `vllm-openvino`

这意味着当前 vLLM 的扩展思路已经很清楚：

```text
模型能力用 ModelRegistry 扩
平台能力用 plugin system 扩
```

---

## 27.10 旧教程里最该删掉的两种说法

### 过时说法 1：去 `models/__init__.py` 里改 `_MODELS`

这在今天已经不是推荐路径，原因包括：

- 不利于 out-of-tree 维护
- 不利于多进程加载一致性
- 每次跟 upstream 同步都容易冲突

当前推荐是：

- 用 `ModelRegistry.register_model(...)`
- 通过 plugin system 注册

### 过时说法 2：`trust_remote_code=True` 基本等于“自定义模型接入完成”

不对。

它最多解决的是：

- HF config / class 发现

而不是：

- vLLM-native 模型实现
- 权重映射
- attention / worker / platform 扩展

---

## 当前最值得读的源码锚点

| 主题 | 当前文件 |
|------|----------|
| HF 集成说明 | `vllm/docs/design/huggingface_integration.md` |
| plugin system 设计 | `vllm/docs/design/plugin_system.md` |
| V1 guide 对 plugin 的定位 | `vllm/docs/usage/v1_guide.md` |
| config 解析 | `vllm/vllm/transformers_utils/config.py` |
| model config 初始化 | `vllm/vllm/config/model.py` |
| registry 实现 | `vllm/vllm/model_executor/models/registry.py` |
| plugin 加载 | `vllm/vllm/plugins/__init__.py` |
| worker 中的 plugin 加载 | `vllm/vllm/v1/worker/worker_base.py` |
| 模型实现范例 | `vllm/vllm/model_executor/models/qwen2.py` / `llama.py` |

---

## 本章小结

| 结论 | 当前仓库里的正确理解 |
|------|--------------------|
| 自定义模型接入主线 | `ModelRegistry + plugin system` |
| `trust_remote_code` 的边界 | 帮你读懂 HF 侧自定义配置，不等于自动完成 vLLM 接入 |
| vLLM-native 模型类核心接口 | `__init__(*, vllm_config, prefix)`、`forward(...)`、`load_weights(...)` |
| 想支持 PP 还要做什么 | 满足 `supports_pp` 契约 |
| 什么时候需要 platform plugin | 新设备、新 worker、新 attention backend、新通信后端 |

---

## 动手实验

### 实验 1：顺着一次模型解析链读源码

任选一个模型仓库路径，从这些文件顺序追：

1. `transformers_utils/config.py`
2. `config/model.py`
3. `model_executor/models/registry.py`
4. 真实模型实现文件

把你的调用链手画出来。

### 实验 2：做一个最小 plugin 原型

不一定真的实现完整模型，只做两件事：

1. 创建一个 `vllm.general_plugins` entry point
2. 在 `register()` 里调用 `ModelRegistry.register_model(...)`

然后验证：

- `ModelRegistry.get_supported_archs()` 里是否出现了你的 architecture

---

## 练习题

### 基础题

1. 当前推荐的 out-of-tree 模型接入路径是什么？为什么不建议继续把直改 `models/__init__.py` 当主线？
2. `trust_remote_code=True` 在当前仓库里能帮你做什么？又不能帮你做什么？

### 思考题

3. 什么情况下只写 model plugin 不够，还需要继续写 platform plugin / `WorkerBase` / attention backend？
4. 如果你准备让一个新模型支持 `pipeline_parallel_size > 1`，除了“先跑通单卡”之外，还必须补哪类接口？
