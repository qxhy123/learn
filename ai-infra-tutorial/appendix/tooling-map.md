# 附录B：工具生态地图

> 本附录不是工具清单，而是 Wave 2 evidence gate 的速查表。每个工具条目都要回答：它证明哪一层问题、进入 EvidenceBundle 的哪一栏、触发哪条 retest 命令、是否改变 CapacityLedger。

## 1. Wave 2 证据门槛

| 证据对象 | 必须回答的问题 | 进入哪些章节 |
|----------|----------------|--------------|
| EvidenceBundle | 现象、时间线、op/kernel 归因、主机/设备健康、修复前后对比是否完整 | Part 0 CPU/内存、Part 2 GPU/IO/runtime |
| CapacityLedger | 优化是否改变 CPU、内存、显存、PCIe/NVLink、NIC、storage、Graph static buffer 或 workspace 容量假设 | Part 0/2/5/6 |
| BenchmarkProtocol | workload、shape、warmup、采样窗口、threshold、重复次数和版本是否固定 | 所有性能章节 |
| retest threshold | 修复是否超过噪声，P50/P95/P99、吞吐、显存峰值和正确性是否达标 | 所有排障章节 |

## 2. Part 0/2 常用证据工具

| 类别 | 工具 / 命令 | EvidenceBundle 字段 | 常见 retest 命令 / 指标 | 章节 |
|------|-------------|----------------------|--------------------------|------|
| CPU 计数器 | `perf stat -d -d -d -- <cmd>`、`perf stat -p <pid>` | Host CPU：cycles、instructions、IPC、cache-misses、dTLB-load-misses、context-switches | 固定 workload 跑 3 次，看 IPC、cache/TLB miss、context switch 是否随修复改善 | 0a、0b、06a、06d |
| CPU 热点 | `perf record -F 99 -g -- <cmd>`、`perf report`、`perf top` | Host stack：DataLoader、tokenizer、dispatcher、IO、Python extension 热点 | 修复后热点 self time 下降，端到端 threshold 达标 | 0a、0b、06a、06d |
| Cache/NUMA | `perf c2c`、`perf mem`、`numactl -H`、`numastat`、`hwloc-ls` | NUMA/cache：remote access、false sharing、socket locality | 绑定 CPU/GPU/NIC 后复测 H2D、DataLoader 和 step time | 0a7、0b3、05b |
| 内存压力 | `vmstat 1`、`sar -B`、`free -h`、`/proc/meminfo` | Host memory：page fault、swap、dirty/writeback、page cache | 固定输入下 major fault、swap、dirty backlog 不再随 step 抖动 | 0b、05d |
| TLB/huge page | `perf stat -e dTLB-load-misses,iTLB-load-misses`、`/proc/meminfo` | Memory translation：TLB miss、THP/HugeTLB 状态 | huge page/布局调整后 miss rate 与尾延迟下降 | 0b1、0b2 |
| PCIe/拓扑 | `lspci -tv`、`nvidia-smi topo -m`、`hwloc-ls` | Topology：GPU/NIC/NVMe/CPU socket 路径 | NUMA 绑定或节点选择后复测 H2D、NCCL、IO | 05b、05c、06b |
| GPU 系统时间线 | `nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas -o run python train.py` | Timeline：CUDA API、Memcpy、NCCL、stream、CUDA HW gaps、NVTX | 修复后 HW gaps、串行 Memcpy、NCCL tail 或 launch overhead 下降 | 04、05b、05c、06a、06b、06d |
| PyTorch 映射 | `torch.profiler` with CPU/CUDA activities, shapes, memory, stacks | Framework：op/module、CPU self time、CUDA time、shape、memory | 小 op、同步点、graph break、fallback 数下降；端到端达标 | 06a、06b、06d |
| Kernel 微观 | `ncu --set full --kernel-name <regex> --launch-skip N --launch-count M -o report <cmd>` | Kernel：occupancy、stall、HBM/L2、Tensor Core、register、shared memory、spill | 只对主瓶颈 kernel 使用；resource pressure 改善且端到端超过 threshold | 04a、04b、06c、06d |
| GPU 健康/集群 | DCGM exporter、`dcgmi dmon`、`dcgmi diag`、`nvidia-smi dmon` | Fleet：SM/memory util、clock、power、temp、ECC、XID、throttle | 排除降频/错误后再解释性能；修复后无 XID/ECC/throttle 异常 | 04d、06d、21 |
| CUDA 计时 | CUDA event、`torch.cuda.Event(enable_timing=True)` | Timing：设备区间时间，不含 CPU 排队 | 只测 GPU work；与端到端计时分开记录 | 04、06b、06d |
| 强制同步 debug | `CUDA_LAUNCH_BLOCKING=1`、`torch.cuda.synchronize()` | Debug：错误栈、同步边界 | 只用于正确性定位；性能 retest 必须关闭 | 06b、06d |
| 带宽 microbench | `bandwidthTest`、自写 H2D/D2H event benchmark、`fio`、`iperf3`、`ib_write_bw` | Bandwidth：PCIe/H2D、storage、TCP/RDMA 链路上限 | 链路修复后 microbench 与业务 workload 都要复测 | 05b、05c、05d |
| NCCL 验收 | `nccl-tests`、`NCCL_DEBUG=INFO`、`NCCL_TOPO_DUMP_FILE=topo.xml` | Collective：AllReduce 带宽、算法、拓扑选择、socket fallback | 修拓扑/bucket 后复测 NCCL tail 和业务 step time | 05c、06b、06d |
| 存储压测 | `fio`、`iostat -xz 1`、`pidstat -d 1`、`mdtest`、`ior`、`s5cmd` | Storage：throughput、IOPS、await、queue depth、metadata/list latency | checkpoint/data pipeline 修复后 storage benchmark 与 step 抖动同时改善 | 05d |
| Web/服务观测 | Prometheus、Grafana、OpenTelemetry、日志系统 | Service：P50/P95/P99、tokens/s、error、queue、cost | 线上灰度按 threshold 回滚或放量 | 15、16、21 |

## 3. 第 6 章专项速查

| 问题 | 首选证据路径 | 典型命令 | retest / threshold |
|------|--------------|----------|--------------------|
| kernel launch overhead | `nsys` 看 CUDA API/HW gaps，`torch.profiler` 找小 op，`perf stat` 排除主机热点 | `nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas -o launch python train.py` | P50 改善 >= 5%，kernel count 或 CUDA API time 下降，数值一致 |
| implicit sync | `torch.profiler` 找 `.item()`/D2H/synchronize，`nsys` 看同步后空洞 | profiler CPU/CUDA activities + `nsys` 短窗口 | per-step synchronize 次数下降，P95/P99 不再有固定空洞 |
| bad stream overlap | `nsys` Memcpy/CUDA HW/stream row，检查 pinned/non_blocking/copy stream | `nsys profile --trace=cuda,nvtx,osrt -o overlap python train.py` | overlap_efficiency 提升，measured time 低于串行估算 |
| CUDA Graph regression | Graph hit/fallback/recapture 指标，`nsys` 请求窗口，allocator peak | 服务指标 + `nsys` + memory summary | P50 改善且 P99 不超过阈值，CapacityLedger 记录 static buffer |
| kernel resource pressure | `ncu` 下钻少数主瓶颈 kernel | `ncu --set full --kernel-name <regex> --launch-count 1 -o kernel <cmd>` | occupancy/stall/spill 改善能解释端到端收益 |
| fusion trade-off | `torch.profiler`/`nsys` 看 kernel count 与端到端，`ncu` 看 register/shared/spill | before/after `nsys` + targeted `ncu` | fusion_gain 假设成立，真实模型而非单点 microbench 达标 |
| H2D/PCIe/NUMA | `nsys` + topology + host memory/NUMA | `nvidia-smi topo -m`、`numactl -H`、H2D benchmark | copy/compute 重叠改善，NUMA 远端访问下降 |
| NCCL tail | 多 rank `nsys` + NCCL debug + topology | `NCCL_DEBUG=INFO`、`nccl-tests`、`NCCL_TOPO_DUMP_FILE` | exposed comm tail 下降，最慢 rank 根因消失 |

## 4. 训练与框架

| 类别 | 常见示例 | 证据用途 |
|------|----------|----------|
| 深度学习框架 | PyTorch、JAX、TensorFlow | 定义模型、训练循环、自动求导；profile 时要映射 op/module 到 runtime 事件 |
| 分布式训练 | PyTorch DDP、FSDP、DeepSpeed、Megatron-LM | 多卡 / 多机训练、状态切分、并行策略；证据重点是 bucket、rank skew、NCCL overlap |
| 加速库 | CUDA、cuDNN、cuBLAS/cuBLASLt、NCCL、Triton | GPU 计算、通信与算子优化；证据重点是库路径、kernel 名、workspace、版本 |
| 编译 / 图优化 | `torch.compile`、XLA、TensorRT、TensorRT-LLM | 减少 Python/dispatcher/launch 和融合算子；必须记录 compile/capture warmup、graph break、fallback |

## 5. 数据、实验与工件

| 类别 | 常见示例 | 证据用途 |
|------|----------|----------|
| 数据版本 | DVC、LakeFS、对象存储版本策略 | BenchmarkProtocol 要固定数据版本，避免把数据变化误判为性能变化 |
| 实验追踪 | MLflow、Weights & Biases、ClearML | 保存 EvidenceBundle、趋势、阈值、profile 附件和版本 |
| 模型仓库 | MLflow Model Registry、Hugging Face Hub、自建 registry | 关联模型版本、权重、tokenizer、推理引擎和性能基线 |
| 特征 / 样本管理 | Feast、自建 feature store、dataset manifest | 保证训练和线上 profile 的输入分布可解释 |

## 6. 调度、平台与服务

| 类别 | 常见示例 | 证据用途 |
|------|----------|----------|
| 容器 | Docker、containerd | 固定 driver/runtime/library 用户态版本，报告镜像 digest |
| 编排 | Kubernetes、Slurm | 记录节点、GPU 分配、MIG/MPS、CPU/GPU/NIC 拓扑和邻居干扰 |
| 队列调度 | Volcano、Kueue、Slurm fairshare | 解释排队、抢占和资源公平性，不要混入单 step 性能结论 |
| 通用模型服务 | KServe、BentoML、Triton Inference Server | 关联请求 P50/P95/P99、batching、模型版本和 GPU 时间线 |
| LLM Serving | vLLM、TensorRT-LLM、TGI | 证据重点是 prefill/decode 分桶、KV cache、CUDA Graph、batch occupancy、tokens/s |
| API 网关 | Envoy、NGINX、Kong、LiteLLM、Portkey | 区分 gateway latency、provider fallback、限流与模型 runtime latency |

## 7. 网络、RDMA 与链路验收

| 类别 | 常见示例 | 证据用途 |
|------|----------|----------|
| TCP / socket 观察 | `ss`、`ip route`、`nstat`、`tcpdump`、`iperf3` | 排查连接状态、重传、路由、MTU、吞吐和包级异常 |
| 网卡配置 | `ethtool`、`ip link`、`tc` | 查看速率、offload、队列、RSS、MTU、ECN/PFC 相关配置 |
| RDMA 设备状态 | `ibstat`、`ibv_devinfo`、`rdma link`、`rdma resource` | 查看 HCA、端口、GID、QP/CQ 和 RDMA 资源 |
| RDMA 性能测试 | `ib_write_bw`、`ib_read_bw`、`ib_send_bw`、`perftest` | 验证 RDMA 带宽、延迟、消息大小和双向性能 |
| InfiniBand / RoCE 管理 | `iblinkinfo`、`ibnetdiscover`、`perfquery`、`mlxlink` | 检查链路错误、速率、拓扑、PFC/ECN、交换机端口状态 |
| NCCL 网络验证 | `nccl-tests`、`NCCL_DEBUG=INFO`、`NCCL_TOPO_DUMP_FILE` | 验证 AllReduce 性能、识别 socket fallback、查看 NCCL 拓扑选择 |

## 8. 存储、对象存储与文件系统

| 类别 | 常见示例 | 证据用途 |
|------|----------|----------|
| 块设备与系统 IO | `iostat -xz 1`、`pidstat -d 1`、`sar -d`、`blktrace` | 观察磁盘利用率、队列深度、await、吞吐和进程级 IO |
| 通用 IO 压测 | `fio`、`dd`、`diskspd` | 构造顺序/随机、读/写、direct/buffered、不同 block size 的基准 |
| 文件系统基准 | `mdtest`、`ior`、`fsbench` | 测试小文件元数据、大文件吞吐、并行文件系统 stripe |
| 文件系统观察 | `df`、`du`、`stat`、`filefrag`、`xfs_info`、`zpool iostat` | 查看容量、extent、XFS/ZFS 状态、碎片和池级吞吐 |
| 对象存储压测 | `s5cmd`、`aws s3`、`rclone`、自建 multipart benchmark | 验证 list/get/put/multipart、并发和尾延迟 |
| 并行文件系统工具 | Lustre `lfs`、GPFS `mmlsfs` / `mmdiag`、BeeGFS `beegfs-ctl` | 查看 stripe、MDS/OSS 状态、客户端挂载和服务端健康 |

## 9. 可观测性、安全与文档工具

| 类别 | 常见示例 | 证据用途 |
|------|----------|----------|
| 指标监控 | Prometheus、Grafana、DCGM exporter | 记录服务指标、GPU 健康、capacity 趋势和告警 |
| 日志 | Loki、Elasticsearch / OpenSearch | 追踪 checkpoint、logger、GC、异常和 XID 时间点 |
| Trace | OpenTelemetry、Jaeger | 区分网关、排队、模型 runtime、下游依赖延迟 |
| 成本与审计 | 云账单系统、自建 cost attribution、审计日志 | 把性能收益换算成成本，记录配置变更责任 |
| 安全扫描 | Trivy、cosign、Vault | 镜像漏洞、签名、secret 管理；属于上线门槛，不替代性能证据 |
| Markdown / 图表 | Mermaid CLI、MkDocs、Docusaurus、VitePress、Playwright screenshot、lychee | 检查教程图表、链接、代码块和 HTML 渲染，不改变 runtime 证据 |

## 10. 版本线 / 关键里程碑速记

> 版本变化很快，这里不把某一天的精确版本号当成长期事实。真正选型、升级或写 BenchmarkProtocol 前，应回到官方 release 页面确认。

| 工具 | 更稳妥的理解方式 | 证据注意事项 |
|------|------------------|--------------|
| vLLM | LLM serving runtime 快速迭代主线 | 重点记录 batching、KV cache、CUDA Graph、attention backend 和 engine 参数 |
| TensorRT-LLM | NVIDIA 生态高性能推理主线 | 重点记录 engine build、plugin、workspace、GPU 代际和版本 |
| Triton | Python 生态自定义 kernel / compiler | 重点记录 Triton 版本、autotune、编译缓存和 kernel 名 |
| FlashAttention | Transformer attention 优化主线之一 | 重点记录 GPU 代际、mask、varlen、dropout、decode 支持 |
| bitsandbytes / GPTQ / AWQ 生态 | 低精度训练/推理常见路线 | 重点记录量化格式、kernel backend、数值质量和 fallback |
| ONNX Runtime / TensorFlow Lite / llama.cpp | 跨平台、移动端、本地推理路线 | 重点记录 backend、线程数、量化格式和设备能力 |

## 11. 选型建议

1. 先明确证据问题：是 control path、data path、kernel resource、host CPU、topology、storage 还是 fleet health。
2. 先跑通最小 BenchmarkProtocol，再引入复杂平台。
3. 工具输出必须进入 EvidenceBundle，不能只作为截图存在。
4. 所有优化都要写 retest threshold，并更新 CapacityLedger。
5. 对核心路径保留可替换边界，避免过早绑定单一实现。
