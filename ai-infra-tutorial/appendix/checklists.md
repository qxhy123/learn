# 附录C：上线与排障检查清单

## 训练任务上线前检查

- 数据快照是否固定，是否可回溯
- 镜像、代码版本、配置文件是否一起归档
- GPU、CPU、内存、磁盘配额是否明确
- Checkpoint 保存频率是否与恢复目标一致
- 指标、日志、告警是否已接入
- 失败重试策略是否会导致重复写入或资源泄漏

## 大规模训练作业预检清单

- `nccl-tests` 或等价 all-reduce benchmark 是否通过
- 所有节点 GPU ECC 错误是否为零
- 所有节点间 IB / RoCE 链路是否健康
- 驱动版本和 CUDA 版本是否与镜像兼容矩阵匹配
- Checkpoint 写入路径是否可用、空间是否充足
- Elastic training 的 rendezvous 配置是否正确
- 慢节点检测和自动替换机制是否就绪

## 推理服务上线前检查

- 请求协议、超时、限流策略是否明确
- 模型版本、tokenizer、配置是否完全匹配
- 模型、prompt、few-shot、guard rail、检索配置是否作为同一发布单元管理
- 冷启动时间是否在可接受范围内
- 批处理、缓存、并发策略是否验证过
- 是否有金丝雀、灰度、回滚路径
- 是否有业务质量监控，而不只是系统指标

## Agent / Tool Use 上线前检查

- 最大步数、最大 token 预算、最大 wall-clock 时间是否已配置
- 工具白名单是否明确，是否按任务类型限制可调用工具
- 工具执行是否在沙箱 / 隔离环境内完成
- credentials 是否做到 scoped、短期、最小权限
- 有副作用动作是否具备 approval gate 或人工接管路径
- step trace、工具调用日志、失败原因是否可审计
- 超预算后的回退路径是否验证过

## 成本与治理检查

- GPU 利用率是否可见
- 资源是否可以按团队或项目归因
- 长时间空闲实例是否有回收策略
- 敏感数据与模型权重是否做权限隔离
- 是否记录了关键运维与发布操作的审计日志

## 安全上线检查清单

- 镜像是否已完成签名或来源校验
- 镜像和依赖是否已完成漏洞扫描，且高危项有结论
- Secrets 是否通过 Secret Manager / K8s Secret 注入，而不是硬编码
- 运行用户是否为非 root，容器能力是否做过最小化
- 网络策略、入口鉴权和租户隔离是否已配置
- 日志、trace、错误返回里是否避免输出敏感数据
- 模型、索引、prompt 和配置是否都具备版本与责任归属
- 回滚路径是否同样满足权限和安全规则

## 性能压测检查清单

- 是否拿到单副本吞吐与延迟基线
- 多副本扩展后吞吐是否接近预期线性增长
- P95 / P99 延迟是否在目标 SLO 内
- 压测期间是否出现 OOM、CUDA error 或 GPU reset
- 冷启动、预热和扩缩容过程是否单独测过
- 连续运行 30 分钟以上后指标是否仍然稳定

## CPU 性能排查清单

- 是否先区分瓶颈在 GPU 计算、CPU 主机侧、数据加载、网络等待还是存储等待
- 是否用 `perf stat` 记录 cycles、instructions、IPC/CPI、branch-misses、cache-misses、context-switches
- 是否用 `perf record/report` 或 VTune 找到 DataLoader、tokenizer、preprocessing、网关或序列化热点
- 是否检查热点循环是否被 SIMD 向量化，编译器报告里是否存在 alias、分支或非连续内存访问阻止向量化
- 是否检查线程数增加后吞吐是否线性增长，还是出现 lock contention、false sharing 或调度开销
- 是否用 `perf c2c`、cache miss 指标或 padding 实验验证伪共享，而不是只凭感觉增加 worker
- 是否检查 L1/L2/L3 miss、working set 大小和 cache line 访问模式
- 是否检查 NUMA locality：CPU core、内存、GPU、NIC 是否在同一 socket 或合理拓扑下
- 是否把 page fault、TLB miss、THP/HugeTLB 状态纳入大内存任务排查
- 是否记录优化前后的固定输入、固定线程数、固定 CPU 频率和可复现实验命令

## 文件系统选型清单

- workload 是 checkpoint 大文件写、dataset 顺序读、小文件随机读、向量索引读写，还是归档冷存
- 关键目标是吞吐、IOPS、尾延迟、崩溃一致性、快照、压缩、成本还是 POSIX 语义
- 是否明确 `write()`、`fsync()`、rename、manifest、对象存储 multipart 的一致性边界
- Page Cache 是期望的加速层，还是会污染 benchmark 或挤压训练进程内存
- ext4 的 journal、XFS 的并发元数据路径、ZFS 的 COW/ARC/快照是否匹配当前读写模式
- 并行文件系统是否正确设置 stripe，MDS 是否会被小文件、目录列表或频繁 stat 打爆
- 对象存储是否只承担归档和 shard 分发，还是被错误当成本地 POSIX 文件系统使用
- 是否用 `fio`、`ior`、`mdtest`、真实 shard 读取和真实 checkpoint 写入分别验收
- 是否为 checkpoint 发布设计临时文件、完整性校验、manifest 原子切换和失败清理
- 是否规划冷热分层：本地 NVMe / 并行 FS / 对象存储之间的数据生命周期和回收策略

## 网络配置健康检查清单

- 所有训练节点的 NIC 速率、duplex、MTU、offload、RSS 队列数是否一致
- RoCE / IB 链路是否无错误计数增长，`ibstat`、`perfquery`、交换机端口状态是否健康
- MTU 是否端到端一致；启用 jumbo frame 时，主机、交换机、路由和容器网络是否同时支持
- RoCE v2 是否配置 ECN/PFC，并验证没有 PFC storm、丢包或拥塞标记异常
- GPU、NIC、CPU socket 的拓扑是否匹配，跨 socket GPU-to-NIC 路径是否被调度器避开
- `ib_write_bw`、`ib_read_bw`、`iperf3` 是否能达到单链路预期带宽和延迟
- `nccl-tests` 是否覆盖单机、多机、不同消息大小、不同节点组合和不同 rail
- NCCL 日志是否显示使用预期的 IB/RDMA path，而不是 fallback 到 socket
- TCP control plane 是否检查重传、SYN backlog、连接数、epoll/io_uring 事件循环延迟
- 是否有链路故障、交换机拥塞、ECN 标记、PFC pause、NCCL hang 的统一告警和 runbook

## Mermaid / 文档构建检查清单

- 每章 mermaid 代码块是否使用受支持的图类型和语法
- mindmap、flowchart、sequenceDiagram、stateDiagram 在 HTML 浅色主题下是否可读
- 图表是否过宽、文字是否溢出移动端或 sidebar 布局
- 离线 mermaid bundle 是否随 HTML 站点一起分发，避免生产浏览依赖外网 CDN
- 构建脚本是否能在 mermaid 渲染失败时给出文件名和代码块位置
- 修改章节顺序后，sidebar、prev/next、附录链接是否同步

## 常见排障入口

### 训练太慢

- 先看 GPU 利用率，再看数据加载与网络等待
- 区分是单机瓶颈还是分布式通信瓶颈
- 先定位瓶颈层，再决定是否需要优化代码

### 推理延迟飙升

- 看请求量变化、批处理策略和缓存命中率
- 看是否发生频繁扩缩容或模型重新加载
- 看下游依赖是否拖慢整个链路

### 成本异常升高

- 看是否存在资源闲置、过度副本、错误实例规格
- 看是否缺少配额与自动回收
- 看是否把离线任务错误地放到高价在线集群
