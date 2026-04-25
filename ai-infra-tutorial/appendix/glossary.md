# 附录A：AI Infra 术语表

| 术语 | 简要解释 |
|------|----------|
| AI Infra（AI Infrastructure） | 承载 AI 数据、训练、评测、部署、推理、监控与治理的基础设施体系 |
| GPU（Graphics Processing Unit） | 适合高吞吐并行计算的加速设备，常用于训练和推理 |
| 显存（GPU Memory / VRAM） | GPU 上的高速内存，决定模型、batch、缓存能否放下 |
| Checkpoint | 训练过程中的可恢复状态，通常包含模型参数、优化器状态和 step 信息 |
| SafeTensors | 更窄、更安全的张量序列化格式，常用于替代依赖 `pickle` 的权重保存方式 |
| 模型包（Model Artifact） | 面向部署的模型产物，通常包含权重、配置、tokenizer 与推理元数据 |
| 数据并行（Data Parallelism） | 多个设备各自处理不同 batch 分片，并同步梯度 |
| 张量并行（Tensor Parallelism） | 将同一层内部计算切分到多个设备上执行 |
| 流水线并行（Pipeline Parallelism） | 将模型不同层分配到不同设备或阶段执行 |
| Sequence Parallelism | 在张量并行组内继续切分序列维度上的部分计算，以降低激活和显存压力 |
| Context Parallelism | 切分 attention 的序列维度来支持更长上下文训练的并行方式 |
| Ring Attention | 一类用环形通信实现 Context Parallelism 的 attention 方案 |
| FSDP（Fully Sharded Data Parallel） | PyTorch 官方的全分片训练实现，按参数 / 梯度 / 优化器状态切分显存压力 |
| ZeRO（Zero Redundancy Optimizer） | 通过分片优化器状态、梯度和参数来降低单卡显存占用的技术族 |
| MFU（Model FLOPs Utilization） | 实际模型有效计算吞吐占理论峰值的比例，强调“算得值不值” |
| HFU（Hardware FLOPs Utilization） | 从硬件角度衡量总 FLOPs 利用率，通常比 MFU 更宽泛 |
| NCCL（NVIDIA Collective Communications Library） | 常用于 GPU 间通信的集合通信库 |
| All-reduce | 多个进程聚合数据并把结果分发给所有进程的通信操作 |
| Fat-tree Topology | 一类提供较均衡跨节点带宽的数据中心网络拓扑，常见于大规模训练集群 |
| Rail-optimized Topology | 让每个 GPU 或节点优先走固定 rail 的网络设计，用更低成本换可接受带宽 |
| FlashAttention | 通过重排 attention 计算和显存访问，降低显存带宽压力的注意力优化方法 |
| KV Cache（Key-Value Cache） | LLM 推理中缓存历史 key/value，减少 decode 阶段重复计算 |
| Continuous Batching | LLM 服务中动态组织正在生成的请求，提高 decode 吞吐的批处理方式 |
| PagedAttention | 将 KV Cache 分块管理以降低显存碎片和预分配浪费的思路 |
| Prefix Caching | 复用相同输入前缀的 KV Cache，减少重复 prefill 计算 |
| Disaggregated Serving | 将 prefill 与 decode 拆成不同服务层或不同资源池的推理架构 |
| Speculative Decoding | 用小模型先生成草稿、大模型再验证，从而加速解码的推理策略 |
| Inference-Time Compute | 在推理阶段额外投入计算，如思维链、搜索、工具调用或树搜索 |
| MoE（Mixture of Experts） | 由多个专家子网络组成、按 token 路由激活部分专家的模型结构 |
| LoRA（Low-Rank Adaptation） | 通过低秩增量矩阵做参数高效微调的方法 |
| QLoRA（Quantized LoRA） | 把量化和 LoRA 结合起来，以更低显存做微调的方法 |
| Multi-LoRA Serving | 一个 base model 实例挂载多个 LoRA adapter，并按请求切换的服务模式 |
| RLHF（Reinforcement Learning from Human Feedback） | 基于人类反馈训练奖励或策略模型的后训练方法 |
| DPO（Direct Preference Optimization） | 直接利用偏好对做优化、避免在线强化学习环节的对齐方法 |
| PPO（Proximal Policy Optimization） | RLHF 中常见的策略梯度优化算法 |
| GRPO（Group Relative Policy Optimization） | 通过同组多个采样结果的相对奖励做优化、常见于去掉 critic 的后训练路线 |
| SLO（Service Level Objective） | 服务等级目标，用于定义可用性、延迟、错误率等目标 |
| RAG（Retrieval-Augmented Generation） | 检索增强生成，把外部知识检索结果引入模型上下文 |
| Embedding | 将文本、图片等对象映射为向量表示 |
| 向量索引（Vector Index） | 支持近似最近邻检索的数据结构或服务 |
| MIG（Multi-Instance GPU） | NVIDIA 的硬件级 GPU 切分能力，可把一张卡分成多个隔离实例 |
| MPS（Multi-Process Service） | NVIDIA 的多进程共享机制，让多个进程复用同一 GPU 上下文 |
| Time-Slicing | 在平台层按时间片复用 GPU 的方式，隔离弱但门槛低 |
| Straggler | 分布式训练里显著慢于其他 worker、拖慢整体同步节奏的慢节点 |
| Elastic Training | 允许训练过程中动态增减 worker，并保持作业继续推进的能力 |
| Spot Instance | 云上可被抢占的低价实例，适合能从 checkpoint 恢复的离线作业 |
| GGML | 面向轻量推理的张量与推理实现项目，常见于端侧 / CPU 推理生态 |
| GGUF | GGML / llama.cpp 生态常见的模型封装格式，便于本地与端侧分发 |
| Canary Release | 让新版本先接入少量真实流量，再逐步放量的发布方式 |
| Blue-Green Deployment | 准备两套独立环境，通过切流快速完成发布或回滚的部署方式 |
| DRF（Dominant Resource Fairness） | 面向多资源系统的公平分配思路，关注租户占用的“主导资源”比例 |
| SLSA（Supply-chain Levels for Software Artifacts） | 用于提升软件供应链可追溯性和可信度的分级框架 |
| 灰度发布（Progressive Delivery / Canary Rollout） | 让新模型或新服务先接收小比例流量，再逐步放量 |
| 成本归因（Cost Attribution / Chargeback） | 将 GPU、存储、网络等资源成本归属到团队、项目、任务或模型 |

---

## 使用建议

阅读正文时，如果遇到术语含义不清，优先回到本表查找大致定义；如果需要更深入理解，再回到对应章节阅读上下文。
