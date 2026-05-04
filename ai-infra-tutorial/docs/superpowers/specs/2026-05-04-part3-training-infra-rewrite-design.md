# Part 3 Training Infra 深度重写设计

> 日期：2026-05-04
> 范围：`part3-training-infra/` 与对应 `html/part3/`
> 决策：保留现有 6 个章节编号，整体重写内容，不拆新章节。

## 1. 背景与问题

当前 Part 3 已经覆盖单机训练、数据并行、模型/流水并行、内存与 checkpoint、后训练、fine-tuning 等主题，但整体气质仍偏“训练概念介绍 + 若干工程建议”。这不符合 AI Infra 资深工程师读者的定位。

目标读者不是只想知道 DDP、FSDP、PPO、LoRA 是什么，而是要能负责 LLM 训练平台、训练任务上线、分布式框架选型、容量评估、效率优化、故障排除和事故复盘。因此 Part 3 需要从“知识科普”改成“训练系统工程手册”。

本次改造保留现有章节编号：

- `07-single-node-training.md`
- `08-data-parallel.md`
- `09-model-pipeline-parallel.md`
- `10-memory-checkpointing-and-recovery.md`
- `10b-alignment-and-post-training.md`
- `10c-finetuning-and-multi-adapter.md`

## 2. 目标

重写后，Part 3 每章都必须回答同一类工程问题：

1. 这个训练能力到底是什么，不是什么，和相邻概念边界在哪里。
2. 它在训练系统中的控制路径、数据路径、状态路径、故障路径是什么。
3. 它背后的核心原理如何从 step time、显存、通信、拓扑、调度、checkpoint、恢复一致性推导出来。
4. 它在 PyTorch、FSDP、DeepSpeed、Megatron、NCCL、TorchElastic、Ray/Accelerate 等框架中如何落地。
5. 生产环境如何配置、发布、回滚、准入、观测、治理。
6. 如何做容量、效率、成本和可靠性估算。
7. 出问题时如何按症状、证据、根因、动作排障。
8. 给定真实规模训练任务时，如何设计方案并解释取舍。

## 3. 非目标

本次不拆分 Part 3 章节，不新增 `07a/08a/09a` 等子章。

本次不把 Part 3 写成机器学习算法教程。可以解释必要公式和训练方法，但重点必须落在系统架构、资源路径、工程实现、故障证据链和方案设计。

本次不追求覆盖所有训练框架细枝末节。框架内容服务于工程边界和可执行配置，而不是替代官方文档。

## 4. 统一章节结构

每章重写后使用统一骨架，允许根据主题微调标题，但必须覆盖以下内容：

1. 第一性原理拆解 + 学习大纲。
2. 概念边界：是什么、不是什么、相邻概念边界。
3. 系统架构：控制路径、数据路径、状态路径、故障路径、责任边界。
4. 核心原理：从不可化简的问题推导机制。
5. 框架实现：PyTorch / FSDP / DeepSpeed / Megatron / NCCL / TorchElastic / Ray / Accelerate 等相关实现。
6. 工程化落地：配置模板、版本矩阵、作业准入、preflight、发布、回滚、观测、治理。
7. 容量与效率：MFU/HFU、tokens/s、samples/s、GPU hours、显存预算、通信开销、checkpoint RPO/RTO。
8. 故障排除：症状、证据、根因、处理动作表。
9. 方案设计：真实规模 worked example。
10. 反模式、checklist、本章小结、练习题。

## 5. 逐章设计

### 5.1 第 7 章：单机训练系统

定位：所有训练系统的最小可验证闭环。

重写重点：

- 讲清一个 training step 的完整执行路径：dataset 读取、CPU preprocessing、DataLoader worker、page cache、pinned memory、H2D、forward、loss、backward、optimizer、AMP、logging、checkpoint。
- 建立单机容量模型：参数、梯度、optimizer state、activation、temporary buffer、CUDA allocator fragmentation。
- 解释 MFU、HFU、GPU utilization、SM occupancy、tokens/s 的边界，避免把 utilization 当成训练效率。
- 覆盖 Mixed Precision / AMP / BF16 / FP8 在单机训练中的工程含义。
- 给出 torch.profiler、Nsight Systems、Nsight Compute、DCGM、iostat、perf 的排障链路。

必备 worked example：

- LLaMA-7B 单机 8xH100 训练基线。
- 推演显存预算、microbatch、gradient accumulation、吞吐、MFU/HFU。
- 给出 step timeline，并定位 DataLoader、H2D、kernel、optimizer、checkpoint 中的瓶颈。

### 5.2 第 8 章：数据并行

定位：复制模型换吞吐，但立刻制造通信和同步问题。

重写重点：

- 对比 DDP、FSDP、ZeRO 的边界：复制什么、切分什么、通信什么、保存什么。
- 讲清 AllReduce、ReduceScatter、AllGather 在 step timeline 中的位置。
- 覆盖 bucket、overlap、gradient accumulation、global batch、loss scale、straggler、data skew、通信拓扑。
- 解释 NCCL ring/tree、rail、NIC、IB/RoCE、环境变量和日志证据。
- 明确什么时候数据并行是正确方案，什么时候应转向 FSDP、TP、PP、CP 或混合并行。

必备 worked example：

- 8 节点 64 GPU 训练任务 step time 拆解。
- 计算 compute time、communication time、overlap 后的 exposed communication。
- 给出 NCCL timeout、带宽不足、rank straggler、dataset skew 的排障表。

### 5.3 第 9 章：模型并行与流水并行

定位：大模型放不下一张卡，也不能只靠数据并行扩展。

重写重点：

- 系统讲 TP、PP、SP、CP、EP、FSDP/ZeRO、3D parallel、interleaved pipeline、zero bubble。
- 解释 microbatch、pipeline bubble、virtual stage、activation placement、sequence/context 切分的本质取舍。
- 给出并行策略选型方法：模型规模、序列长度、GPU 拓扑、节点内 NVLink/NVSwitch、节点间 IB/RoCE、框架支持、checkpoint 格式、恢复方式。
- 覆盖 Megatron-style 配置、DeepSpeed pipeline、FSDP hybrid sharding 的工程边界。
- 明确并行策略如何影响 checkpoint、optimizer state、故障恢复和推理转换。

必备 worked example：

- 70B 与 405B 两档模型训练方案推演。
- 给出 TP/PP/DP/CP/FSDP 组合、microbatch、global batch、通信路径、checkpoint 形态。
- 对比至少两种并行配置的吞吐、显存、网络压力和恢复复杂度。

### 5.4 第 10 章：内存优化、Checkpoint 与恢复

定位：训练系统的长期运行可靠性控制面。

重写重点：

- 把显存优化从技巧提升为资源调度问题：activation checkpointing、offload、optimizer state sharding、mixed precision、FP8、allocator fragmentation。
- 把 checkpoint 从“保存文件”提升为“恢复协议”：保存什么、谁保存、何时可见、如何验证、如何清理、如何跨并行策略恢复。
- 覆盖 checkpoint schema、sharded checkpoint、async checkpoint、atomic visibility、metadata、retention、RPO/RTO。
- 覆盖 TorchElastic、elastic restart、preflight validation、straggler detection、NCCL hang 排障。
- 讲清恢复一致性：模型参数、optimizer、scheduler、RNG、dataset cursor、global step、parallel metadata 必须一致。

必备 worked example：

- 千卡训练中断恢复事故。
- 从告警、rank 状态、NCCL 日志、checkpoint metadata、存储指标、scheduler 事件建立证据链。
- 给出恢复方案、数据损失评估、RPO/RTO 复盘和后续治理动作。

### 5.5 第 10b 章：对齐训练与后训练基础设施

定位：多模型、多角色、多阶段训练系统。

重写重点：

- 分清 pretraining、SFT、RM、PPO、DPO、GRPO 的系统形态。
- 把 PPO/RLHF 写成系统架构：actor、reference、reward、critic、rollout engine、training engine、sample generation、reward scoring、replay/buffer。
- 讲清 rollout 与训练之间的吞吐匹配、资源切分、checkpoint 多模型一致性。
- 覆盖评测门禁、实验追踪、数据版本、prompt/config 版本、失败恢复。
- 对比 DPO/GRPO 相对 PPO 的平台化难度和资源形态。

必备 worked example：

- LLaMA-7B 或 70B 的 PPO/RLHF pipeline。
- 给出 actor/ref/reward/critic 的 GPU 布局、rollout 吞吐、训练吞吐、瓶颈定位。
- 覆盖 reward model 延迟、样本队列堆积、多模型 checkpoint 不一致的排障。

### 5.6 第 10c 章：Fine-tuning 基础设施与多 Adapter 服务

定位：面向租户的 FTaaS + Adapter 生命周期系统。

重写重点：

- 对比 full fine-tune、LoRA、QLoRA、DoRA 的资源交换：显存、存储、训练时间、推理挂载成本、质量风险。
- 设计 FTaaS 控制面：数据准入、训练队列、配额、镜像、基础模型版本约束、adapter registry、审批、产物发布。
- 讲清 adapter 与 base model 的兼容性：模型架构、tokenizer、rank、target modules、quantization、license、safety policy。
- 覆盖 merge deployment 与 dynamic attach 的边界。
- 覆盖 multi-LoRA serving、adapter hot load、cache、A/B、回滚、权限和审计。
- 打通训练产物进入推理服务的完整路径。

必备 worked example：

- 多租户 LoRA 平台方案。
- 给出队列、配额、显存预算、adapter registry schema、服务热加载路径。
- 排查 adapter 不兼容、热加载失败、显存碎片、质量回退、租户隔离问题。

## 6. 内容质量验收

每章必须满足：

- 至少一个“是什么 / 不是什么 / 相邻概念边界”小节。
- 至少一张架构图或路径图，说明控制路径、数据路径、状态路径或故障路径。
- 至少一个容量、效率或可靠性公式。
- 至少一个框架配置示例或伪配置，能映射到 PyTorch/FSDP/DeepSpeed/Megatron/NCCL/TorchElastic/Ray/Accelerate 中的实际参数。
- 至少一个症状-证据-根因-动作排障表。
- 至少一个真实规模 worked example，包含数字、推理链和取舍。
- 至少一个方案设计 checklist。
- 避免只写“应该”“可以”“通常”，关键建议必须落到证据、指标、命令、配置、边界或反例。

## 7. 实施策略

实现阶段建议使用 6 个 subagent 并行，每个 subagent 负责一个 Markdown 章节。主 agent 做两阶段 review：

1. 第一阶段 review：检查章节结构、深度、是否符合统一骨架，必要时要求 subagent 返工。
2. 第二阶段 review：统一交叉引用、术语、HTML 同步、链接校验和导航数据。

每个 subagent 只允许编辑自己负责的 Markdown 文件。HTML 重新生成和 `tutorial-data.js` 检查由主 agent 完成，避免并行冲突。

## 8. 验收命令

实现完成后至少运行：

```bash
wc -l part3-training-infra/*.md
rg -n "TODO|FIXME|TBD|附加推演|补充推演|边界问题" part3-training-infra html/part3
rg -n "是什么|不是什么|架构|原理|工程化|故障排除|方案设计|Worked Example" part3-training-infra/*.md
git diff --check -- part3-training-infra html/part3 html/assets/tutorial-data.js
```

还需要运行本地 HTML 链接检查，确保 `html/part3/*.html` 中没有坏链、残留 `.md` 链接或未转换 Mermaid。
