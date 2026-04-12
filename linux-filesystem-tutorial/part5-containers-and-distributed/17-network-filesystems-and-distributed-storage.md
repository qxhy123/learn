# 第17章：网络文件系统与分布式存储

> 高阶视角下，远端文件系统最难的不是“网络慢”，而是：路径、缓存、锁、元数据、失败边界和提交语义都从单机问题变成了跨节点协商问题。

## 学习目标

完成本章后，你将能够：

1. 区分本地文件系统语义、网络文件系统语义和对象存储语义
2. 理解 close-to-open、属性缓存、delegation/lease、锁服务为什么重要
3. 说明 NFS、CephFS、FUSE 网关、对象存储挂载为什么都叫“文件”，但边界完全不同
4. 认识为什么 POSIX 语义在分布式环境里昂贵且常被部分折中
5. 用远端失败模型解释“本地正常，迁远端后出问题”的根因
6. 理解元数据服务器、split-brain 成本和“数据库为何常常绕开共享文件系统”的原因

---

## 正文内容

### 17.1 远端语义从一开始就不是本地语义的平移

本地文件系统很多默认直觉来自：

- 单机内核掌握全局真相
- 元数据和缓存协调范围有限
- 写入顺序只要和本机 I/O 栈协调

一旦跨网络，这些前提都不再免费成立：

- 不同客户端可同时读写
- 每个客户端都有自己的缓存
- 元数据与锁状态需要跨节点一致
- 失败不再只有崩溃，还有网络分区、超时和脑裂边界

所以“远端 ext4”这个想法本身就不成立。

### 17.2 close-to-open 解决的是“什么时候更可能看到新内容”

NFS 中常见的 close-to-open 语义直觉是：

- 某客户端关闭文件后
- 另一个客户端下次打开该文件时
- 更有机会看到较新的状态

注意这不是“全时刻强同步”，而是一种观测边界折中。它的意义在于：

- 降低每次操作都全局同步的成本
- 给应用一个相对合理的可见性边界

但如果程序默认“别人一写完我立刻就必须看见”，close-to-open 可能仍然不够强。

### 17.3 属性缓存与目录缓存为什么会制造“不同客户端看到不同现实”

远端客户端常缓存：

- `stat` 类属性
- 目录项结果
- 部分数据页

这意味着不同客户端会在一段时间里看到不同的元数据世界。于是会出现：

- A 看到目录里有新文件，B 暂时还没看到
- A 的 `mtime` 已更新，B 的属性缓存还没过期
- 程序依赖“立刻可见”的小技巧在本地能工作，在远端失效

这不是 bug，而是缓存、一致性和可扩展性的权衡结果。

### 17.4 delegation / lease 为什么重要

为了降低频繁同步成本，一些远端文件系统会授予客户端某种 delegation 或 lease，让客户端在短时间内更大胆地缓存和操作对象。但一旦其他客户端介入，服务端又要撤销或协调这些授权。

这说明远端文件系统的一致性常常不是“每次都强同步”，而是“在可撤销授权和缓存收益之间动态平衡”。

### 17.5 锁服务为什么是分布式文件系统的硬骨头

本地锁很多时候依赖单机内核对象状态。远端场景下，锁意味着：

- 谁持有锁的真相要跨节点维护
- 客户端崩溃后锁如何清理
- 网络分区时要不要继续允许一侧写
- 锁服务自身故障怎么办

所以远端文件锁往往既贵又复杂，也经常是应用迁移后最先暴露问题的地方。

### 17.6 POSIX 在远端为什么昂贵

POSIX 的很多直觉在单机上很自然：

- rename 原子
- 目录语义明确
- 锁和权限边界清晰
- 元数据变化即时可见

但分布式环境中，要把这些语义都做强，成本极高。因此很多系统会：

- 只实现接口兼容，不实现完全相同的边界语义
- 在缓存可见性、锁一致性、目录更新时序上做折中
- 鼓励应用改变协议，而不是假设远端等同本地

### 17.7 先把三种“一致性直觉”分开

远端文件系统讨论里，最容易混淆的是把不同强度的一致性说成同一个词。至少要先区分：

- **会话/客户端局部直觉**：我自己刚写完，再读到自己的更新
- **close-to-open 一类可见性直觉**：我关闭后，别人下次打开时更可能看见新状态
- **单副本强一致直觉**：任何客户端在任何时刻都像在操作同一个本地 inode

这三者的成本完全不同。很多系统能比较好地做到前两者，但第三种在跨节点、多缓存、可扩展元数据管理下代价极高。

如果应用没有先说明自己到底需要哪一种，就很容易在测试环境“看起来没问题”，到生产流量或故障场景才暴露。

### 17.8 元数据面、数据面、控制面经常不是一回事

单机文件系统里，很多人会下意识认为“文件系统”就是一个整体。分布式里通常会拆成多层：

- **元数据面**：目录树、inode 属性、权限、配额、锁状态、命名空间
- **数据面**：真正承载文件内容的对象、chunk、stripe 或块
- **控制面**：客户端会话、授权、回收、故障转移、健康检查

这会直接带来几个工程后果：

- 元数据热点和数据吞吐热点可能根本不是一个瓶颈
- “目录操作慢”与“顺序吞吐低”往往要找不同组件
- 某些系统扩展数据面容易，但扩展强元数据语义很贵
- 调优时必须先判断你在打哪一层，而不是笼统地说“远端存储慢”

### 17.9 失败矩阵比接口表更重要

远端文件系统真正难的是失败，不是 API 名字。至少要分别考虑：

| 失败事件 | 典型问题 |
|----------|----------|
| 客户端崩溃 | lease/锁如何回收，未刷新的脏数据如何处理 |
| 服务端重启 | session、delegation、锁状态、挂起 I/O 如何恢复 |
| 网络分区/高丢包 | 谁还能写，谁必须冻结，是否可能脑裂 |
| 元数据节点切换 | 目录/锁/rename 中途是否需要重放或回退 |
| 时钟偏差 | 依赖 mtime/atime/租约超时的逻辑会不会误判 |

这张矩阵一旦没想清，应用再漂亮的接口抽象也只是纸面工程。

### 17.10 FUSE 网关和对象存储挂载为什么更要小心

把对象存储通过 FUSE 网关或某种适配层“挂成文件系统”，并不代表它突然拥有了本地 POSIX 语义。很多时候你得到的是：

- 文件接口长得像 POSIX
- 但 rename、目录更新、mtime、锁、原子性、持久化边界并不一样

这类系统常适合：

- 归档
- 大对象读多写少
- 简化访问方式

但并不天然适合强依赖本地语义的小文件、频繁 rename、复杂锁协议场景。

### 17.11 CephFS、NFS、对象存储挂载的差别不在“是不是远端”

这三类系统都可能是远端，但工程定位不同：

- **NFS**：传统网络文件接口，强调兼容和可用折中
- **CephFS**：分布式文件系统，更强调多节点扩展和一致性协议
- **对象存储挂载/FUSE 网关**：接口适配层，常把对象语义伪装成文件语义

所以“都是远端文件系统”这句话信息量太低，真正重要的是：你依赖的语义是哪一类。

可以把它们先做一个粗粒度对照：

| 方案 | 最擅长什么 | 最容易踩什么坑 |
|------|------------|----------------|
| NFS | 兼容传统共享目录与现有工具链 | 属性缓存、close-to-open、锁/可见性错觉 |
| CephFS | 更大规模共享与分布式治理 | 元数据路径复杂，运维与故障域更重 |
| 对象存储挂载/FUSE | 统一访问入口、归档、大对象读多写少 | 小文件、rename、锁、目录语义、延迟尾巴 |

### 17.12 一个迁移排查框架：本地正常，远端异常

当应用从本地迁到远端后出问题，可以先问：

1. 是否依赖强 rename 语义？
2. 是否假设 `stat` / mtime 立刻同步？
3. 是否依赖本地文件锁行为？
4. 是否假设 `close` 后所有客户端立刻看到同一状态？
5. 是否把对象存储挂载误当成完整 POSIX 文件系统？
6. 失败后由谁恢复：客户端重试、服务端重放，还是应用自己的 WAL/校验协议？

很多“偶发异常”“难复现 bug”恰恰来自这些隐式假设。

### 17.13 元数据服务器、缓存撤销与“远端目录树并不便宜”

很多分布式文件系统真正昂贵的，不是传数据本身，而是维护名字空间和元数据真相。

典型要处理的问题包括：

- 谁负责目录树、inode 属性、权限和锁的权威状态
- 客户端缓存了目录项、属性或 delegation 之后，何时必须撤销
- 当目录热点、rename 风暴、海量小文件出现时，元数据路径是否会先于数据路径成为瓶颈

这也是为什么：

- 一些系统需要单独的 metadata server（MDS）层
- 大目录、小文件、频繁 rename 在远端场景里格外昂贵
- “带宽够大”并不能保证远端文件系统就适合元数据密集业务

### 17.14 split-brain、网络分区与“你愿意牺牲哪边”

单机文件系统的失败模型主要是崩溃和设备错误；分布式文件系统还要面对：

- 网络分区
- 脑裂
- 客户端长时间失联后重连
- 锁状态、lease、delegation 的回收与重建

高阶设计里最难的问题之一不是“怎么让它工作”，而是：

- 分区时优先保可用性还是保强一致？
- 客户端超时后，谁有权宣布它的锁失效？
- 重连后，哪些缓存必须作废，哪些写入必须重放或拒绝？

所以远端文件系统很多语义折中，背后其实都是 CAP 式代价在文件接口里的投影。

### 17.15 为什么数据库、WAL 系统和构建缓存常常谨慎对待共享文件系统

很多数据库、日志系统、构建缓存或消息系统，对共享文件系统通常采取更谨慎甚至规避的态度，原因不是“它们不懂文件系统”，而是它们非常清楚自己依赖什么：

- 精确的刷盘语义
- 稳定的锁与故障恢复边界
- 可预测的尾延迟
- 对 rename、mtime、目录项、缓存失效的严格假设

一旦这些假设在远端场景下变弱，系统就需要：

- 自己补 WAL / manifest / checksum / fencing
- 降低共享目录扫描和元数据风暴
- 避免把“能 mount 上”误当成“适合承载强语义数据库负载”

所以更成熟的判断方式不是“这个共享文件系统能不能用”，而是：

**应用到底依赖了哪些语义，而这些语义是底层真的提供，还是你过去默认免费获得的。**

### 17.16 一个按语义而不是按品牌做选择的表

| 你最看重什么 | 更值得优先考虑什么 | 最该警惕什么 |
|--------------|--------------------|--------------|
| 多客户端共享、传统目录工作流 | NFS / SMB / CephFS 这类真正的远端文件系统 | 把本地可见性和锁语义直接照搬 |
| 大规模对象、归档、读多写少 | 对象存储或对象网关 | 把挂载接口误判成完整 POSIX |
| 高性能单机强语义事务 | 本地文件系统 + 应用自己的 WAL/恢复协议 | 以为远端共享一定更省事 |
| 容器镜像/平台层分发 | overlayfs、snapshotter、对象存储分发链路 | 把镜像层语义与运行时写层语义混为一谈 |

先写出自己真正依赖的语义，再选系统，通常比先比品牌名更可靠。

---

### 17.17 NFS 协议内部：RPC、属性缓存与 close-to-open 实现

**NFS 客户端的内核结构**：

```c
/* include/linux/nfs_fs.h — NFS inode 扩展结构 */
struct nfs_inode {
    /* VFS inode（必须是第一个字段，用于 container_of 转换）*/
    struct inode        vfs_inode;

    /* NFS 文件句柄（服务端 inode 的不透明标识符）*/
    struct nfs_fh       fh;             /* 文件句柄（最长 64 字节）*/

    /* 属性缓存 */
    unsigned long       read_cache_jiffies; /* 上次从服务器刷新属性的时间 */
    unsigned long       attrtimeo;      /* 属性缓存超时（动态调整：actimeo）*/
    unsigned long       attrtimeo_timestamp; /* 超时时间戳 */
    struct timespec64   atime;          /* 缓存的访问时间 */
    struct timespec64   mtime;          /* 缓存的修改时间 */
    struct timespec64   ctime;          /* 缓存的状态改变时间 */
    __u64               change_attr;    /* NFSv4 change attribute（单调递增版本号）*/
    loff_t              cur_size;       /* 缓存的文件大小 */

    /* 目录缓存 */
    unsigned long       cache_validity;  /* 缓存有效性标志（NFS_INO_INVALID_DATA 等）*/
    struct nfs_open_context *cache_head; /* 当前 open file 上下文链表 */

    /* page cache 状态 */
    struct radix_tree_root  nfs_page_tree; /* 待提交的写请求树 */
    unsigned long       ncommit;        /* 待提交的写请求数 */
    atomic_long_t       nrequests;      /* 进行中的读写请求数 */

    /* 锁状态（NFSv4）*/
    struct list_head    open_files;     /* 此 inode 的所有 open 上下文 */
    struct rw_semaphore rwsem;          /* 保护 nfs4 stateid 等 */
};

/* NFS 挂载参数（影响属性缓存行为）*/
/* acregmin=N (默认 3s):  普通文件属性缓存最短时间 */
/* acregmax=N (默认 60s): 普通文件属性缓存最长时间 */
/* acdirmin=N (默认 30s): 目录属性缓存最短时间 */
/* acdirmax=N (默认 60s): 目录属性缓存最长时间 */
/* actimeo=N:             统一设置 acregmin/max 和 acdirmin/max */
/* noac:                  禁用属性缓存（总是从服务器获取，极慢但总是最新）*/
```

**属性缓存的动态调整算法（Solaris 发明，Linux 沿用）**：

```
属性缓存超时的自适应算法（exponential backoff）：

初始：attrtimeo = acregmin（默认 3 秒）

每次访问时：
  if (文件在 attrtimeo 内未被修改):
      attrtimeo = min(attrtimeo * 2, acregmax)  ← 指数增加（最长 60s）
  else:
      attrtimeo = acregmin                       ← 重置为最短

效果：
  - 频繁修改的文件 → 缓存时间短（3-6s），较快看到更新
  - 长期稳定的文件 → 缓存时间长（最长 60s），减少 GETATTR RPC 次数

观察 NFS 属性缓存命中率：
```

```bash
# 查看 NFS 统计（挂载点的 RPC 调用次数）
nfsstat -c           # 客户端统计
# 或
cat /proc/net/rpc/nfs
# net 0 0 0 0                   ← 网络层统计
# rpc 12345 0 0                 ← RPC 调用总数、重传数、认证错误数
# proc4 58 0 3456 2345 ...      ← NFSv4 各 operation 调用次数

# 查看特定挂载点的详细统计
cat /proc/self/mountstats | grep -A50 "device server:/path"
# device server:/export mounted on /mnt/nfs with fstype nfs4 statvers=1.1
# opts: rw,vers=4.2,rsize=1048576,wsize=1048576,namlen=255,acregmin=3,acregmax=60
# age: 3600 seconds
# impl_id: name='',domain='',date='0,0'
# caps:   caps=0x3ffdf,wtmult=512,dtsize=32768,bsize=0,namlen=255
# nfsv4:  bm0=0x7ffffbff,bm1=0x40f9be3e,...
# sec: flavor=unix,pseudoflavor=0
# events: 1234 5678 90 ...     ← 各类缓存事件（inode revalidate 等）
# bytes:  12345678 87654321 0 0 ...  ← 读/写/直接读/直接写 字节数
# RPC iostats version: 1.0 p/v: 100003/4 (nfs)
# ops[0]  GETATTR 4500  0 0 ...  ← GETATTR: 次数、超时、重传、RTT(ms)...
# ops[1]  SETATTR 12    0 0 ...
# ops[8]  READ    789   0 0 0 0 3456789 0 0 234    ← 每个 op 的延迟分布

# 用 mountstats 工具格式化输出
mountstats /mnt/nfs
# ...
# NFS byte counts:
#   Applications read 123456789 bytes via read(2)
#   Applications wrote 12345678 bytes via write(2)
#   Applications read 0 bytes via O_DIRECT read(2)
#
# RPC statistics:
#   736 RPC requests sent, 736 RPC requests completed (100.0% successful)
#   Average bytes sent per RPC: 212
#   GETATTR: 245 ops (33%), avg RTT: 1.2ms  ← 高比例 GETATTR 说明属性缓存失效频繁
#   READ:    456 ops (62%), avg RTT: 3.4ms
```

**close-to-open 一致性的内核实现**：

```bash
# close-to-open 的 NFS 实现（fs/nfs/file.c）：
# nfs_file_release()（close 系统调用的文件系统钩子）：
#   → nfs_wb_all()：把所有脏页写回服务器（刷写本地 page cache）
#   → nfs_commit_all()：等待写入被服务器 commit（若是 unstable write）
#
# nfs_open()（open 系统调用的文件系统钩子）：
#   → nfs_revalidate_inode()：向服务器发 GETATTR，刷新属性缓存
#   → 若 change_attr 变化 → 使本地 page cache 失效（丢弃旧数据）

# 演示 close-to-open 效果
# 在服务端修改文件
ssh nfs-server "echo 'new content' > /export/test.txt"

# 客户端立刻读（属性缓存可能命中旧值）
cat /mnt/nfs/test.txt     # 可能看到旧内容

# 关闭并重新打开（触发 close-to-open 一致性）
exec 3< /mnt/nfs/test.txt
exec 3>&-   # 关闭
cat /mnt/nfs/test.txt     # 现在应该看到新内容

# 用 strace 确认
strace -e trace=network -p $$ &
cat /mnt/nfs/test.txt   # 观察 GETATTR RPC 是否被发送
```

---

### 17.18 CephFS 架构：MDS、OSD 与客户端缓存

CephFS 是 Ceph 存储系统的文件系统层，理解其架构有助于理解分布式文件系统的复杂性：

```
CephFS 架构：

客户端（libcephfs / ceph-fuse / kcephfs）
       │
       ├── 元数据操作 ──→ MDS 集群（Metadata Server）
       │                  ├── 维护目录树、inode、权限、锁
       │                  ├── 动态分片（目录可以分布到多个 MDS）
       │                  ├── 基于 RADOS 持久化元数据（metadata pool）
       │                  └── 向客户端发放 capability（读/写/缓存授权）
       │
       └── 数据操作 ───→ OSD 集群（Object Storage Daemon）
                         ├── 文件数据分片为 objects（默认 4MB/object）
                         ├── 每个 object 按 CRUSH 算法分布到多个 OSD
                         ├── 每个 OSD 通常对应一块物理磁盘
                         └── 直接与客户端通信（bypass MDS）

capability 机制（Ceph 的 delegation 实现）：
  MDS 向客户端发放 capability：
  - Fc (file cache)  → 允许客户端缓存文件内容
  - Fw (file write)  → 允许客户端本地脏数据
  - Fr (file read)   → 允许客户端读数据
  - Fx (file excl)   → 排他锁
  - Fs (file shared) → 共享锁（读锁）

  当其他客户端需要不兼容的 capability 时：
  → MDS 发 revoke 消息撤销已有 capability
  → 持有 capability 的客户端必须在超时内：
      1. 刷写本地脏数据到 OSD
      2. 返回 capability 给 MDS
  → MDS 再把 capability 授予新请求者
```

**CephFS 客户端观察**：

```bash
# 查看 CephFS 客户端状态（在挂载节点上）
ceph tell mds.* session ls  # 所有活跃客户端会话

# 查看 MDS 缓存状态
ceph tell mds.0 cache status
# {
#   "capacity": 4294967296,      ← MDS 内存缓存容量
#   "num_inodes": 123456,        ← 缓存的 inode 数
#   "num_dentries": 234567,      ← 缓存的 dentry 数
#   "heap": 2345678900,          ← 堆内存使用量
# }

# 观察 MDS 的 capability 撤销事件（可能影响应用延迟）
ceph tell mds.0 dump_tree /
# 查看目录树和 capability 分布

# 挂载 CephFS（内核客户端）
mount -t ceph mon1:6789,mon2:6789,mon3:6789:/ /mnt/cephfs \
    -o name=admin,secret=AQDxxxxxx...==

# 挂载选项
# rbytes=1        ← 报告真实目录大小（而非快速估算）
# nocrc           ← 禁用校验和（提高写性能，降低可靠性）
# readdir_max_entries=N ← 每次 readdir 最大条目数

# 通过 /sys/kernel/debug/ceph 观察内核客户端状态
ls /sys/kernel/debug/ceph/
# 每个挂载点一个目录（包含 mds_sessions、caps、osd_requests 等文件）
cat /sys/kernel/debug/ceph/*/caps
# total 12345, implemented 11111, issued 9876
# ↑ 客户端持有的 capability 总数（过多可能导致 MDS revoke 延迟）
```

---

### 17.19 NFS 挂载参数调优与常见问题排查

```bash
# 查看当前 NFS 挂载参数
mount | grep nfs
cat /proc/mounts | grep nfs4

# NFS 关键挂载参数对性能的影响：

# 1. rsize/wsize：读写块大小（影响每次 RPC 传输量）
mount -t nfs4 server:/path /mnt -o rsize=1048576,wsize=1048576
# 默认通常 1MB（较老内核可能只有 32KB），大 rsize/wsize 减少 RPC 次数

# 2. soft vs hard 挂载（决定 NFS 故障时应用的行为）
mount -t nfs4 server:/path /mnt -o soft,timeo=100,retrans=3
# soft: 超时后返回 EIO 给应用（应用能感知错误，避免永久挂起）
# hard（默认）: 永久重试，应用进程会一直阻塞（在 D 状态）
# timeo=100: 超时 10 秒（单位：0.1秒）
# retrans=3: 重试 3 次

# 3. async vs sync 挂载（写入语义）
mount -t nfs4 server:/path /mnt -o async  # 默认：异步（write 立即返回，后台 commit）
mount -t nfs4 server:/path /mnt -o sync   # 每次 write 都同步到服务器（极慢但安全）

# 4. 属性缓存调整
mount -t nfs4 server:/path /mnt -o actimeo=0   # 禁用属性缓存（总是 GETATTR）
mount -t nfs4 server:/path /mnt -o acregmax=5  # 文件属性最长缓存 5 秒（比默认 60s 更激进）

# 常见问题排查：

# 问题 1: df 挂起（NFS 服务器不可达）
# → D 状态进程
ps aux | grep " D "  # 找到阻塞在 NFS 的进程
# → 用 soft 挂载或 nfsidmap + 超时保护

# 问题 2: stale file handle (ESTALE)
# → 服务端文件被删除/重建，但客户端还持有旧 fh
stat /mnt/nfs/file  # 返回 ESTALE
# → umount 并重新挂载（或 mount -o remount）

# 问题 3: 权限问题（nobody:nobody）
# → uid/gid 映射问题
ls -la /mnt/nfs/file  # 显示 nobody:nobody
# → 检查 /etc/idmapd.conf 和服务端的 /etc/exports
# → 确认客户端和服务端的 Domain = 一致

# 问题 4: 写入速度慢（COMMIT RPC 频繁）
nfsstat -c | grep commit  # 统计 COMMIT 次数
# → 大量 COMMIT 说明 write 是 unstable write，每次都等服务端 commit
# → 在服务端启用 write cache 或调整应用写入批次大小

# 调试 NFS 性能的完整工具链
rpcdebug -m nfs -s all    # 开启 NFS 调试（输出到 dmesg）
mountstats /mnt/nfs       # 查看每个 operation 的延迟统计
nfsiostat 1 5 /mnt/nfs    # 实时 NFS I/O 统计（类似 iostat）
# ops/s   kB/s    kB/op   retrans  avg RTT (ms)  avg exe (ms)
# Read:  234.5  1234.5    5.3      0.0           3.4           3.6
# Write:  45.6   456.7   10.0      0.0           8.9          12.3
```

---

## 本章小结

| 主题 | 结论 |
|------|------|
| 远端语义 | 不是本地语义的简单平移 |
| close-to-open | 是可见性边界的常见折中 |
| 属性缓存 | 会让不同客户端暂时活在不同元数据世界里 |
| delegation / lease | 用授权和撤销平衡缓存收益与一致性 |
| 锁服务 | 是分布式文件系统最难的组成之一 |
| 失败矩阵 | 比接口兼容表更能决定方案是否可用 |
| 对象存储挂载 | 接口像文件，不等于语义像本地 POSIX |
| metadata server / split-brain | 解释了为什么远端目录、锁和恢复成本常比数据吞吐更关键 |
| 数据库等强语义应用 | 往往需要自己补协议，而不是盲信共享文件接口 |

---

## 练习题

1. 为什么远端文件系统的难点不只是“多一段网络延迟”？
2. close-to-open 解决了什么，又没有解决什么？
3. 属性缓存为什么会让不同客户端看到不同现实？
4. 为什么对象存储挂载看起来像文件系统，却不能被直接当成本地文件系统替代？
5. 应用从本地迁到远端后，最容易暴露哪几类隐式语义假设？
6. 为什么“能否容忍客户端/服务端重启后的恢复方式”常比吞吐更影响方案选择？
7. 为什么 metadata server、目录热点和 rename 风暴常常比“网络带宽”更能决定远端文件系统上限？
8. 数据库或 WAL 系统为什么经常需要比普通应用更严格地审视共享文件系统语义？
