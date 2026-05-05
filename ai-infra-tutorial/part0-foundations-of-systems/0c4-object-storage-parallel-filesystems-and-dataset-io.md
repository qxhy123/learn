# 第 0c4 章 对象存储、并行文件系统与 Dataset IO

> **关联章节**：本章把 [0c1](0c1-vfs-inode-dentry-and-block-layer.md) 的 VFS 思维扩展到非本地后端，并承接 [0c3](0c3-storage-semantics-fsync-direct-io-and-checkpoints.md) 的发布语义。重点是对象存储、并行文件系统、manifest、stripe、MDS 和小文件治理。

## 1. 第一性原理拆解 + 学习地图

### 拆：不可化简的问题

AI 数据平台既要喂饱 GPU，又要支持多租户共享、版本管理、失败恢复和低成本存储。
单机 POSIX 文件系统不能覆盖所有需求。
对象存储提供廉价、弹性、HTTP API 和强大的持久性，但不是 inode + rename 模型。
并行文件系统提供共享命名空间和高带宽，但元数据服务和条带策略会成为系统设计的一部分。

### 推：从问题推出机制

- 对象存储没有目录 inode，key 前缀只是命名约定，所以 dataset 入口应是 manifest 而不是递归 LIST。
- 大对象上传不能一次请求完成，所以需要 multipart upload、part checksum、complete 和 abort。
- 多 worker 读取不能全靠小对象随机 GET，所以需要 shard、range read、本地 cache 和 index。
- 并行文件系统要把数据分散到多个 OSS/OST 或 NSD 上，所以需要 stripe size/count。
- 海量小文件会打爆 MDS 或对象存储请求预算，所以要治理文件粒度。

### 绘：三种存储模型

```mermaid
flowchart TD
  App[Training workers] --> POSIX[POSIX path API]
  App --> SDK[Object SDK]
  POSIX --> PFSClient[Parallel FS client]
  PFSClient --> MDS[MDS/MDT metadata]
  PFSClient --> OSS1[OSS/OST 1]
  PFSClient --> OSS2[OSS/OST 2]
  SDK --> Manifest[manifest.json]
  Manifest --> Obj1[shard-0001.tar]
  Manifest --> Obj2[shard-0002.tar]
  Obj1 --> Range[Range GET]
  Obj2 --> Range
  POSIX --> LocalCache[local NVMe cache]
  SDK --> LocalCache
```

### 导：本章读完后能做什么

1. 解释对象存储的 PUT/GET/LIST/multipart 与 POSIX 文件系统的差异。
2. 用 manifest 设计 dataset 和 checkpoint 的发布入口。
3. 判断并行文件系统瓶颈来自 MDS、OSS/OST、stripe、客户端 cache 还是网络。
4. 把小文件 dataset 改造成 shard + index + cache 的读取路径。
5. 写出 dataset IO 的观测命令、SOP 和验收指标。

## 2. 对象存储不是 POSIX 文件系统

对象存储的基本单位是 object。
object 有 key、bytes、metadata、etag/version 等属性。
所谓目录通常只是 key 前缀，例如 `dataset/v3/train/shard-0001.tar`。
没有本地文件系统意义上的 inode、hard link、目录 fd、跨对象原子 rename。

常见操作：

| 操作 | 含义 | 与 POSIX 的差异 |
|---|---|---|
| PUT object | 写入一个完整 object | 不是 `write()` 流式可见文件 |
| GET object | 读取 object | 可用 Range 读取片段 |
| LIST prefix | 枚举某个前缀 | 不是目录 inode 遍历，成本和一致性要按服务理解 |
| Multipart upload | 分片上传大 object | complete 前 object 通常不可见 |
| Copy + delete | 模拟 rename | 非原子，成本高，失败状态复杂 |

现代主流对象存储**目前**提供强一致读写语义，但这是历史不长的现状：

- **AWS S3 自 2020-12 起**才提供 strong read-after-write 和 strong list-after-write；之前是 read-after-write 强一致 + list 最终一致。许多博客、论文、SDK 文档仍写着旧语义。
- GCS 一直提供 strong read-after-write；LIST 在跨 bucket / 大量 prefix 场景下仍有"最终一致"的尾巴要小心。
- Azure Blob Storage 是 strong read-after-write。
- MinIO、Ceph RGW、各类对象存储网关、跨区域复制的 bucket：一致性语义差别大，要看具体配置和版本。
- **FUSE 挂载层会重新引入弱一致性窗口**：s3fs、goofys、JuiceFS、ossfs 等基于元数据缓存（默认几秒到几十秒），写完不能立刻读到——这是"挂着像 POSIX 实际不是 POSIX"的最常见坑。

即便底层对象存储是强一致，应用仍不应把 LIST 当成 dataset 真相源。LIST 成本高、分页复杂、跨账号/跨区域/网关/FUSE 层语义可能变化，而且训练启动时全量 LIST 会制造控制面尖峰。

## 3. Multipart upload

Multipart upload 把大对象拆成多个 part 上传。
完成流程通常是：initiate、upload parts、complete。
complete 前，最终 object 不应作为已发布数据被 reader 使用。
失败时要 abort，避免遗留未完成分片占用成本。

设计要点：

- part size 不要太小；太小会增加请求数和服务端元数据压力。
- 每个 part 记录 checksum，complete 后记录 object checksum 或组合校验。
- 上传 worker 控制并发，避免把带宽、CPU checksum、TLS 和服务端 throttling 混在一起。
- 失败重试必须幂等，part number 和内容要稳定。
- complete 成功后再发布 manifest 指针。

S3 的硬约束（其他对象存储类似但数字略有出入，以官方文档为准）：

- 单个 part：5 MB 起，5 GB 止（最后一个 part 可小于 5 MB）。
- 单次上传最多 10000 parts。
- 单 object 上限 5 TB。
- 这三条决定 shard size 的可行区间：5 GB shard 用 1 part；50 GB shard 至少 10 parts，每 part 5 GB；500 GB shard 必须 part 大小 ≥ 50 MB；接近 5 TB 时 part 必须 ≥ 500 MB。
- 实际选择常落在 part 16-128 MB，单 shard 1-50 GB——既有合理并发，又留容错重试空间。
- 未 complete 的 multipart upload 不会自动清理，**要配 lifecycle rule 在 N 天后 abort**，否则不可见的"半成品"会持续吃存储费用。

示例命令：

```bash
aws s3api create-multipart-upload --bucket my-bucket --key dataset/v3/shard-0001.tar
aws s3api upload-part --bucket my-bucket --key dataset/v3/shard-0001.tar \
  --part-number 1 --body part-0001 --upload-id <upload-id>
aws s3api complete-multipart-upload --bucket my-bucket --key dataset/v3/shard-0001.tar \
  --upload-id <upload-id> --multipart-upload file://parts.json
```

## 4. Manifest 是发布入口

Manifest 是对象存储和并行数据集里最重要的应用层边界。
它把“有哪些 shard、每个 shard 多大、校验和是什么、样本索引在哪里、版本是什么”写成一个小对象或小文件。
训练入口读取 manifest，而不是实时扫描目录或 LIST prefix。

一个最小 manifest：

```json
{
  "dataset": "imagenet-sharded",
  "version": "v2026-05-04",
  "format": "tar+idx",
  "shards": [
    {"key": "train/shard-000000.tar", "bytes": 1073741824, "sha256": "...", "samples": 8192},
    {"key": "train/shard-000001.tar", "bytes": 1073741824, "sha256": "...", "samples": 8192}
  ],
  "index": "train/index-v2026-05-04.parquet"
}
```

发布协议：

1. 上传所有 shard，完成 multipart。
2. 校验 size 和 checksum。
3. 上传 manifest 到不可变 key，例如 `manifests/v2026-05-04.json`。
4. 更新小指针 `current.json`，内容只包含当前 manifest key 和版本。
5. reader 读取 `current.json`，再读取 manifest，再按 manifest 读取 shard。

这样即使前缀下有临时对象、旧版本或失败上传残留，reader 也不会把它们纳入训练。

**Content-addressed key（CAS）模式**：把 object key 包含内容 sha256（如 `blobs/sha256/ab/abcd1234.../shard.tar`），manifest 引用这个 key。好处：

- 上传天然幂等：重复上传同 key 同内容是 no-op，重复上传同 key 不同内容是 bug 报警点。
- 跨数据集去重：同一 shard 在多个 manifest 引用，只存一份。
- 缓存 key 直接用 sha256，cache invalidation 不需要版本号约定。
- 清理只删"没有任何 manifest 引用的 blob"，回收策略简单。

代价是 key 不可读，需要双层目录结构（manifest → CAS key），开发期定位文件不直观。学术 dataset、企业训练平台多采用这一模式。

## 5. LIST 的成本和陷阱

LIST 适合管理和审计，不适合作为每个训练 job 的启动路径。
一个包含 5000 万对象的数据集，如果每个 job 启动都分页 LIST，会产生高延迟、高费用和控制面压力。
分页过程中如果还有写入或删除，应用还要处理重复、遗漏、排序和 continuation token。

更稳的模式：

- 数据生产时生成 manifest 和 index。
- 训练时只读 manifest。
- 定期离线审计 LIST 与 manifest 是否一致。
- 清理任务只删除 manifest 不再引用的对象。
- 多版本并存时用版本前缀或内容地址 key，避免覆盖更新。

观测对象存储请求：

```bash
aws s3 ls s3://bucket/dataset/v3/ --recursive --summarize
aws s3api list-objects-v2 --bucket bucket --prefix dataset/v3/ --max-items 1000
aws s3api head-object --bucket bucket --key dataset/v3/train/shard-0001.tar
```

## 6. 对象存储读取路径：Range GET、cache、prefetch

训练读取 shard 时通常不下载整个 dataset。
worker 根据 index 定位 shard 和 byte range，再发起 Range GET 或从本地 cache 读取。
本地 cache 可以是节点 NVMe、daemon cache、FUSE cache 或框架自带 cache。

设计参数：

| 参数 | 影响 | 建议 |
|---|---|---|
| shard size | 请求数、并行度、失败重试成本 | 常从 256MB 到 2GB 起测 |
| sample grouping | 随机性和顺序读效率 | 同一 shard 内打乱，跨 shard 分配 |
| prefetch depth | 隐藏网络延迟 | 与 worker 数、内存、带宽一起调 |
| cache key | 复用和一致性 | dataset version + shard checksum |
| retry budget | 尾延迟和成本 | 区分 404、429、5xx、timeout |

不要让所有 rank 同时读取同一个 shard 开头。
可以按 global rank 对 shard 列表做确定性切分，并在 epoch 间改变顺序。

读对象存储到 GPU 的现代捷径：

- **NVIDIA DALI**（数据加载/解码 pipeline）和 **cuFile/GDS** 配合，可以把对象存储 → 节点 NVMe cache → GPU 的解码和拷贝大幅 offload；JPEG/视频解码可直接在 GPU 上做。
- 对纯 IO 密集的 dataset（fp16 tensor、tokenized text），GDS 直读 NVMe shard cache 比"先到 CPU 再 cudaMemcpy"省一份内存带宽。GDS 路径细节见 [0c1 §8.5](0c1-vfs-inode-dentry-and-block-layer.md#85-gpudirect-storage-与-cufile) 和 [0d3c](0d3-rdma-roce-infiniband-and-gpudirect.md)。

PyTorch DataLoader + 对象存储常见坑：

- `fsspec` / `s3fs` / `boto3.client` 的连接池在 `fork` 后不可重用；`num_workers > 0` 时必须在 worker init（`worker_init_fn`）里**重新创建** S3 client，否则会出现 `BotoCoreError` 或随机 hang。
- 默认 boto3 连接池上限 10，单 worker 高并发 Range GET 时是隐形瓶颈，要显式 `botocore.config.Config(max_pool_connections=64)`。
- `requests`/`urllib3` 的 keepalive 可以让一个 worker 复用 TLS 连接，省下握手；但每个 worker 进程独立连接池，跨 worker 不复用，total connections = workers × pool_size，注意被服务端限流。
- 重试策略：区分 `404`（key 真不存在，立刻失败）、`429/503`（限流，指数退避）、`5xx`（短重试）、network timeout（短重试 + 换连接）。一律重试会把限流问题放大成雪崩。

## 7. 并行文件系统模型

Lustre、GPFS/Spectrum Scale、BeeGFS 等并行文件系统提供 POSIX 风格共享命名空间。
它们的核心是把元数据路径和数据路径拆开。
MDS/MDT 负责目录、inode、权限、layout。
OSS/OST 或 NSD 负责数据块。
客户端把 VFS 请求转成网络协议和后端请求。

| 组件 | 负责 | 典型瓶颈 |
|---|---|---|
| client | VFS 接入、cache、RPC | CPU、网络、缓存失效 |
| MDS/MDT | lookup、create、unlink、stat | 小文件、目录扫描、锁竞争 |
| OSS/OST | 数据读写 | 顺序带宽、热点 OST |
| network | client 到服务端 | 拥塞、包丢、RDMA 配置 |
| coordinator | quota、锁、恢复 | 大规模 job 同步风暴 |

并行文件系统能提供很高 aggregate bandwidth，但不意味着 `stat()` 无限快。
DataLoader 如果每个 sample 都触发 open/stat，小文件元数据仍会先打满 MDS。

并行 FS 尾延迟的隐形来源——**分布式锁**：

- Lustre 用 **LDLM（Lustre Distributed Lock Manager）**：每个 inode/extent 范围由 server 颁发锁给 client，写共享区域需要 revoke 其他 client 的锁。多 rank 写同一目录、追加同一 log 文件，会触发频繁锁回收，体现为客户端 stall。
- GPFS/Spectrum Scale 用 **token manager**：类似机制，token revoke 在跨节点 workload 上是常见尾延迟来源。某些版本 token manager 单点会成为瓶颈。
- **`fsync(parent_dir)` 在 Lustre 上是一次到 MDS 的同步 RPC**——所有 rank 同时 fsync 同一目录就是把 MDS 当串行队列用，是 §13 mini case 的本质问题。
- 诊断：Lustre 用 `lctl get_param ldlm.namespaces.*.pool.stats`、`llstat`；GPFS 用 `mmdiag --tokenmgr`、`mmpmon`。

### 7.1 Lustre 对象模型：FID、Layout EA、OST object

Lustre 的"client/MDS/OSS 三层"是表层架构。要真正调它，必须理解它的对象模型——一次 `read("/mnt/lustre/data/shard.tar")` 在内部是怎么落到具体存储节点上的。

**FID（File Identifier）** 是 Lustre 的核心命名实体：

- 128-bit 全局唯一标识（`seq:oid:ver`），相当于 Lustre 版的 inode number。
- 不同于本地 inode，FID 在整个 Lustre 文件系统范围内唯一，跨 MDT 不冲突。
- 客户端操作文件时，路径解析的输出是 FID；之后所有对该文件的 RPC 都以 FID 为 key——不再依赖路径。
- 这就是 Lustre 支持"open-by-FID"的基础：进程持有 FID 后，文件被 rename 不影响读写（因为 FID 不变），类似本地 FS 的 fd 在 unlink 后仍能用。

**MDT（Metadata Target）** 上每个文件是一个 MDT inode：

- 存储常规元数据（mode、owner、size、mtime 等）。
- 关键扩展属性 **Layout EA（`trusted.lov` 或 `trusted.lmv`）** 记录这个文件由哪些 OST object 组成、stripe 策略是什么。
- `lfs getstripe <path>` 实际就是读这个 EA。

**Layout EA 的内容**（简化）：

```text
magic = LOV_MAGIC_V3
pattern = RAID0 (stripe)
stripe_size = 1MB
stripe_count = 4
objects = [
  {ost_idx=2, object_id=FID_2_obj_xxx},
  {ost_idx=5, object_id=FID_5_obj_yyy},
  {ost_idx=7, object_id=FID_7_obj_zzz},
  {ost_idx=11, object_id=FID_11_obj_www}
]
```

**OST（Object Storage Target）** 上每个 object 是一个独立的存储对象，由该 OST 后端文件系统（典型 ldiskfs 即修改版 ext4，或 ZFS）存储。**file 不是直接放在 OST 上**——file 的字节按 stripe 切分后，每个 stripe 写到对应 OST object 的相应 offset。

读 `shard.tar` offset 0 到 4MB 的实际过程（stripe_size=1MB, count=4）：

1. 客户端有 FID 和 Layout EA（`open` 时从 MDS 取，缓存在内存）。
2. 0-1MB 落 ost_idx=2 的 object 的 0-1MB；1-2MB 落 ost_idx=5 的 object 的 0-1MB；以此类推。
3. 客户端**并行**向 4 个 OST 发 read RPC（走 LNET）。
4. 4 个 OST 各自读自己后端 ldiskfs/ZFS，返回数据。
5. 客户端组装成连续 buffer 返回给应用。

理解这个模型后，调优就有锚点了：

- **stripe_count 提升大文件吞吐**因为是物理上多 OST 并发；但小文件 stripe_count > 1 增加 RPC 数和锁开销。
- **DoM**（Data on MDT）是把小文件的"前 N KB"直接存在 MDT 的 inode 关联区域，**完全跳过 OST**——`open + read` 一次 MDS RPC 完成，对 manifest、index 这种小热文件 latency 显著降低。
- **PFL** 让 Layout EA 支持"按 offset 区间用不同 stripe 策略"，本质是 Layout EA 编码多个 component。

### 7.2 LNET：Lustre 的网络抽象

Lustre 不直接跑在 TCP 或 RDMA 上，而是跑在 **LNET（Lustre Networking）** 上。LNET 是一个抽象的消息层，下面挂不同的 LND（LNet Network Driver）：

- `socklnd`：跑在 TCP 上，部署最容易，性能上限有限。
- `o2iblnd`：跑在 InfiniBand verbs 上，AI 训练集群典型选择。
- `kfilnd`：跑在 OFI/libfabric 上，Slingshot 等新一代网络。

每个 Lustre 节点（client、MDS、OSS）都有一个 NID（Network ID，格式 `<IP>@<lnd>`，如 `10.0.0.5@o2ib`）。客户端 mount 时指定 MGS 的 NID，从 MGS 拿到整个 cluster 的 NID 拓扑和 router 配置。

LNET 关键能力：

- **multi-rail**：一个节点多张 NIC 可以聚合成一个 LNET interface，自动负载均衡和 failover。
- **LNET routers**：跨网段（不同 LND 之间）的网关，让 IB 子网的 client 访问 TCP 子网的 OSS。
- **discovery**：节点发现和健康监测在 LNET 层，比上层 RPC 更早发现链路问题。

诊断：

```bash
lctl list_nids                              # 本节点 NID
lctl which_nis                              # LNET 接口状态
lnetctl net show -v                         # 详细网络拓扑
lctl get_param 'osc.*.import' | grep state  # client → OSS 连接状态
lfs check servers                           # 所有 server 是否可达
```

### 7.3 LDLM 锁的类型与命名空间

LDLM（Lustre Distributed Lock Manager）是 Lustre 一致性和并发控制的核心。锁分多种类型，挂在不同命名空间下：

**锁类型**（按强度排序，可兼容性递增）：

| 类型 | 含义 | 兼容 |
|---|---|---|
| `EX`（Exclusive） | 独占写 | 不兼容任何其他锁 |
| `PW`（Protected Write） | 保护写 | 兼容其他 PW 和更弱锁的部分组合 |
| `CW`（Concurrent Write） | 并发写 | 多 client 同时写不同 extent |
| `PR`（Protected Read） | 保护读 | 兼容其他读和 CR |
| `CR`（Concurrent Read） | 并发读 | 兼容性最高 |
| `NL`（Null） | 占位 | 与所有兼容 |

**锁命名空间**：每个 server 上有一个 LDLM namespace，client 上对应的 import 也是 namespace。锁按资源类型分：

- **MDC（Metadata Client）锁** `mdc-*`：锁 MDT 上的 inode、目录项。`open` 拿 inode 锁，`readdir` 拿目录锁。
- **OSC（Object Storage Client）锁** `osc-*`：锁 OST object 的某个 extent 范围。`read` 拿 PR 锁、`write` 拿 PW 锁，可以是 `[0, EOF]` 整文件锁，也可以是某个 byte range。

锁的关键机制是 **lock revocation**：

- 一个 client 拿了 PW 锁，另一个 client 来要 EX 锁，server 必须先回收前者的锁——发 BL_AST（Blocking AST）通知 client 释放或降级。
- client 收到 BL_AST 时，必须先 flush 所有相关 dirty page、推进 in-flight RPC，才能 release 锁。
- **这就是多 rank 写同一文件、同一目录时尾延迟抖动的根源**：锁回收链上任何一个 client 慢，整条链阻塞。

实战观察：

```bash
lctl get_param ldlm.namespaces.*.lock_count             # 各 namespace 锁数量
lctl get_param ldlm.namespaces.*.pool.granted           # 已授予锁
lctl get_param ldlm.namespaces.*.lru_size               # 锁 LRU 大小
lctl set_param ldlm.namespaces.*.lru_size=clear         # 清 LRU（紧急）
lctl get_param osc.*.rpc_stats                          # OST RPC 统计
lctl get_param mdc.*.rpc_stats                          # MDT RPC 统计
```

设计 checkpoint workload 时的几条经验：

- **每个 rank 写独立的文件**，避免在同一 OST object 上拿锁。
- **预创建分目录**让锁分散到不同 MDT inode 命名空间。
- **避免所有 rank fsync 同一个父目录**：dir fsync 触发 MDT 上的 inode 锁强制同步，512 个并发 = 串行队列。
- **大文件 PFL + DoM** 让小文件不占 OST 锁、大文件 stripe 到多 OST 让锁分散。
- **DLM lock cancel 风暴**（大量 BL_AST 在短时间内涌出）是 Lustre 节点 OOM 或重启后常见现象，可在 `/var/log/messages` 看到。

## 8. Stripe 策略

Stripe 决定一个文件的数据如何分布到多个数据服务端。
stripe count 越高，大文件可并行读写的服务端越多。
但小文件使用过高 stripe count 会增加元数据和锁成本。
stripe size 决定连续多少字节放在同一 stripe 上。

Lustre 示例：

```bash
lfs getstripe /mnt/lustre/dataset
lfs setstripe -c 8 -S 16M /mnt/lustre/checkpoints
lfs df -h
lfs osts
```

经验方向：

| 文件类型 | stripe count | stripe size | 理由 |
|---|---|---|---|
| 10GB+ checkpoint shard | 多 OST | 8MB 到 64MB 起测 | 提升大文件并行带宽 |
| 512MB tar shard | 中等 | 4MB 到 16MB 起测 | 平衡吞吐和管理成本 |
| 100KB 小图片 | 1 | 默认或小 stripe | 避免为小文件放大元数据 |
| manifest/index 小文件 | 1 | 默认 | 低延迟、简单 |

Stripe 不是越大越好。
如果所有 rank 同时写很多大文件，高 stripe count 可能让所有文件打到所有 OST，制造全局竞争。
更好的方案可能是按 rank 或目录分布 stripe，使热点分散。

现代 Lustre（2.10+）的两个关键 layout 机制，在大型训练集群里直接决定吞吐：

- **PFL（Progressive File Layout）**：同一文件在不同 offset 区间用不同 stripe 策略。例如：
  - 0-1MB：stripe count=1（小文件友好，少占 OST）。
  - 1MB-1GB：stripe count=4（中等并发）。
  - 1GB+：stripe count=16（大文件追求并行带宽）。
  - 这样写小文件不浪费 OST，写大 checkpoint 又能展开到多 OST。`lfs setstripe -E 1M -c 1 -E 1G -c 4 -E -1 -c 16 /mnt/lustre/data`。
- **DoM（Data on MDT）**：极小文件（典型阈值 64KB-1MB）数据直接存在 MDT 上，`open + read` 一次 RPC 到 MDS 完成，不再触达 OST。对 manifest、index、tokenizer 这种小但热的文件 latency 显著降低。代价是吃 MDT 容量，必须留足。`lfs setstripe -E 64K -L mdt -E -1 -c 4`。

Mixed PFL + DoM 是 AI 训练 Lustre 的现代默认布局。设置错或没设，单租户能看到正常带宽，多租户混跑时 MDS 和 OST 利用率严重不均。

## 9. 小文件治理

小文件问题不是“文件系统不够快”，而是 IO 形状不匹配。
每个样本一个对象或一个文件，会产生大量 open/stat/GET/head 请求。
当单样本只有几十 KB，元数据、TLS、RPC、调度和 Python overhead 可能超过数据传输本身。

治理方向：

- 打包成 tar、zip、RecordIO、Parquet、Arrow、TFRecord、WebDataset 等 shard。
- 为 shard 建 index，记录 sample id 到 `(shard, offset, length)`。
- 保持 shard 不可变，用 manifest 发布版本。
- 把随机性放在 index 和 sampler 层，而不是依赖随机小文件读取。
- 本地 cache 以 shard 为单位，不以单样本为单位。
- 对异常样本建立 sidecar blacklist，不重写整个 dataset。

文件大小分布观测：

```bash
find /mnt/dataset -type f -printf '%s\n' \
  | awk '{n++; s+=$1; if($1<65536)a++; if($1<1048576)b++} END {print "files",n,"avg",s/n,"<64K",a,"<1M",b}'
find /mnt/dataset -type f | sed 's#/[^/]*$##' | sort | uniq -c | sort -nr | head -20
```

## 10. Dataset 读取架构

一个稳定的 dataset IO 架构通常包含四层。

```text
manifest/current pointer
  -> shard manifest and index
    -> object store or parallel FS shards
      -> node-local cache
        -> worker prefetch and decode
```

职责分离：

| 层 | 负责 | 不应该负责 |
|---|---|---|
| manifest | 版本、完整性、shard 列表 | 实时发现对象 |
| index | sample 到 byte range | 数据持久性 |
| storage | 保存 shard bytes | 训练随机性 |
| cache | 降低重复读取成本 | 版本真相 |
| worker | prefetch、decode、batch | 全局目录扫描 |

训练 worker 只需要知道当前 epoch 分到哪些 sample 或 shard。
它不应在热路径里递归目录、LIST 对象前缀、或为每个样本做远端 HEAD。

## 11. 命令观测：对象、并行 FS、本地 cache

对象存储侧：

```bash
aws s3api head-object --bucket bucket --key dataset/v3/manifest.json
aws s3api list-objects-v2 --bucket bucket --prefix dataset/v3/train/ --max-items 10
curl -I https://example-bucket.s3.amazonaws.com/dataset/v3/train/shard-0001.tar
```

并行文件系统侧：

```bash
findmnt -T /mnt/shared -o TARGET,SOURCE,FSTYPE,OPTIONS
df -hT /mnt/shared
df -ih /mnt/shared
lfs getstripe /mnt/shared/path 2>/dev/null || true
mmlsfs all 2>/dev/null | head || true
beegfs-ctl --getentryinfo /mnt/shared/path 2>/dev/null || true
```

训练进程侧：

```bash
pidstat -d -p <pid> 1
strace -f -c -e trace=openat,newfstatat,read,close -p <pid>
iostat -x 1
ss -tinp | head
```

如果对象存储请求 p99 高，但本地 cache hit 率低，先优化 cache key、prefetch 和 shard 切分。
如果并行文件系统 MDS busy，而 OST 带宽不高，先减少小文件和目录扫描。
如果 OST 带宽满而 MDS 空闲，再调 stripe、worker 并发和压缩解码。

## 12. Worked example：2000 万小图片迁移

场景：原始数据是 2000 万张小图片，平均 80KB，按类别目录存放。
训练启动要扫描目录生成 sample list，启动 20 分钟，GPU 经常等待。
存储后端从本地 NFS 迁到对象存储后，直接一文件一对象更慢。

迁移方案：

1. 离线扫描原始目录，生成全量样本表：sample id、label、原始路径、size、checksum。
2. 按类别和随机种子打包为 1024MB 左右 tar shard。
3. 为每个 shard 生成 index：sample id、offset、length、label、checksum。
4. 上传 shard 到对象存储，使用 multipart 和 checksum。
5. 上传全局 Parquet index 和 manifest。
6. 训练读取 `current.json`，按 rank 切分 shard，Range GET 或本地 cache 读取。
7. 每个 epoch 在 index 层 shuffle，不在对象存储层 LIST。

收益评估：

| 指标 | 迁移前 | 迁移后目标 |
|---|---|---|
| 训练启动 | 20 分钟目录扫描 | 读取 manifest/index 小于 30 秒 |
| 请求数 | 每 epoch 数千万 open/stat/GET | 每 shard 少量 GET/Range |
| GPU idle | 随机尖峰 | decode 或训练成为主要瓶颈 |
| 版本管理 | 目录覆盖 | manifest 不可变版本 |

失败处理：如果某个 shard checksum 不匹配，reader 标记该 shard 不可用并停止训练，不能静默跳过样本。
清理任务只删除没有被任何 manifest 引用的 shard。

## 13. Mini case：并行文件系统 checkpoint 热点

场景：64 节点训练，每节点 8 rank，每 rank 写一个 8GB 文件到 Lustre。
所有 rank 写到同一个目录，默认 stripe count 为 1。
现象是总吞吐远低于存储标称，MDS CPU 高，部分 OST 空闲。

分析：

- 同一目录创建 512 个文件，MDS 处理 create、layout、lock。
- stripe count 1 让每个大文件只落到一个 OST，分布可能不均。
- **所有 rank 同时 `rename` + `fsync(parent_dir)` 等于把 MDS 当串行队列**：每次 dir fsync 都是一次到 MDS 的同步 RPC，512 个并发请求在 MDS 上排队完成。这是本案瓶颈的核心。

改法：

- 预创建分层目录，例如 `node-000/rank-000.bin`，减少单目录热点（每个子目录独立 LDLM 锁域）。
- 对大 rank 文件设置合适 stripe count，例如 4 或 8，并压测 stripe size；或直接用 PFL layout 让大段自动 stripe 多 OST。
- rank 文件写完后由协调者**串行**或**分批**发布 manifest，避免所有 rank 同时操作同一父目录。每个节点选 1 个 rank 做发布代理，节点间 reduce 后再写 manifest 是常见模式。
- 把最终长期存储异步转移到对象存储，训练热路径只保留最近 checkpoint。
- 进一步：用 reflink（如果底层是 XFS scratch + 异步同步到 Lustre）或对 Lustre 直接配置 PFL/DoM。

验证：看 MDS/OST 指标、`lfs getstripe`、`iostat`、训练端 fsync p99。
不要只看 aggregate bandwidth 的单次峰值。

## 13.5 Mini case：FUSE/s3fs 假装 POSIX 训练慢

场景：团队为了"代码不改"，把 S3 bucket 用 s3fs（或 goofys、ossfs）FUSE 挂到 `/mnt/dataset`，DataLoader 直接 `Image.open(path)` 读小图。
现象：单节点训练 GPU 利用率 < 30%，`iostat` 看不到块设备流量，`ss -t` 看到大量 https 连接，单 step 的 batch wait 时间是本地 NVMe 的 10× 以上。

为什么慢，按层次拆：

1. **每次 `open()` 都触发一次 HEAD 请求**：FUSE 没有真目录，s3fs 必须问对象存储"这个 key 存在吗、多大"。HEAD 单次 ~30ms（同区域），1 万张图就是 5 分钟纯 metadata。
2. **FUSE 上下文切换**：每个 syscall（`open/read/getattr`）都从内核态跳到 s3fs 用户态进程再跳回，2-5× 延迟惩罚。
3. **元数据缓存默认弱一致**：写后立刻读可能拿到旧值或 404；为了稳定常配置 short TTL，进一步放大请求量。
4. **没有 prefetch、没有 Range 优化**：每张图一次完整 GET，没有打包读、没有 keep-alive 复用。
5. **被 throttle**：单 prefix 高频 GET 会触发 5xx，s3fs 默认重试更放大请求。

正确做法：

- **不要把 FUSE 当 POSIX 用做训练热路径**。FUSE 适合管理工具、调试、低频访问。
- 改造为 manifest + shard + 节点本地 cache：原始小图打包为 WebDataset/tar shard，训练直接 `boto3.client.get_object` 走 SDK，cache 到本地 NVMe。
- 如果一定要 POSIX 接口，用专为训练设计的客户端（JuiceFS、3FS、MountPoint for S3 with prefetching、Lustre-on-S3 等），并明确其元数据缓存语义、prefetch 行为、强一致窗口。
- 验证：迁移前后对比 `pidstat -d`、TCP 连接数、GPU 利用率、单 step batch wait p99；FUSE 路径下 batch wait p99 通常是新路径的 5-20×。

## 14. Dataset IO SOP

1. 统计文件数量、大小分布、目录 fanout、样本读取顺序和解码成本。
2. 明确后端：对象存储、并行文件系统、本地 NVMe、NFS、FUSE 或混合。
3. 规定版本入口：`current.json` 或等价 manifest 指针。
4. 数据生产阶段生成 shard、index、checksum 和 manifest。
5. 训练阶段禁止热路径全量 LIST 或递归目录扫描。
6. 按 rank/worker 做确定性切分，避免所有 worker 抢同一 shard。
7. 设置本地 cache 容量、水位、淘汰策略和 cache key。
8. 记录请求 p50/p95/p99、cache hit、decode time、GPU wait、MDS/OST 指标。
9. 做失败演练：缺 shard、坏 checksum、对象存储 429、MDS 抖动、cache 满。
10. 把清理策略绑定 manifest 引用，不按前缀时间盲删。

## 15. Checklist

- 是否把对象存储 key 前缀当成命名约定，而不是 POSIX 目录？
- 是否使用 manifest 作为 dataset 入口？
- 是否避免训练启动时全量 LIST？
- 是否为 multipart upload 记录 part 和最终 checksum？
- 是否按 shard 而不是单小文件设计 cache？
- 是否知道并行文件系统的 MDS 和数据服务端是否分别饱和？
- 是否为大文件设置并验证 stripe，而不是照抄默认值？
- 是否把小文件治理放在数据格式层，而不是只换后端？
- 是否对 reader 做 size/checksum 校验？

## 16. 练习

1. 设计一个 manifest，描述 3 个 shard、每个 shard 的 size、checksum 和样本数。
2. 解释为什么对象存储上的 rename 不能当成本地 POSIX rename 使用。
3. 给一个 1TB tar shard 数据集，选择 shard size 和 worker 切分策略。
4. 看到 MDS CPU 高、OST 带宽低，应如何改 DataLoader 或目录布局？
5. 为一个节点本地 cache 设计 key、容量水位和清理规则。
