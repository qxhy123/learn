# 第8章：ext4 布局与日志机制

> **本章导读**：ext4 不是”带日志的 ext2”，也不是一句”成熟稳定”就能讲完。理解 ext4，要同时看 extent、延迟分配、目录索引、JBD2 日志、写屏障、恢复语义和它刻意没有做的事情。

**前置知识**：第7章（块、超级块与空间分配）
**预计学习时间**：60 分钟

---

## 学习目标

完成本章后，你将能够：

1. 说明 ext4 的核心布局对象和它们的工程目的
2. 理解 extent、delayed allocation、mballoc、HTree 目录索引的价值
3. 区分 `data=ordered`、`data=writeback`、`data=journal` 的语义差异
4. 解释 JBD2 transaction、commit、checkpoint、barrier/flush 的直觉
5. 识别 ext4 在快照、校验、自修复、跨设备池化等方面的边界
6. 认识 unwritten extent、`fallocate`、e2fsck 和挂载参数如何改变你对 ext4 的工程感受

---

## 正文内容

### 8.1 ext4 的目标不是“最强”，而是“稳健默认”

ext4 的工程定位通常是：

- 向后兼容 ext 系生态
- 在普通服务器和桌面场景中表现稳定
- 元数据恢复能力足够强
- 性能和复杂度保持平衡
- 与 Linux 工具链、启动流程、运维经验高度兼容

这很重要，因为 ext4 的很多设计选择都不是“理论最优”，而是“在通用场景下风险最低、迁移成本最低、行为最可预期”。

### 8.2 ext4 的基本布局地图

理解 ext4 时，最值得抓住这些对象：

| 结构 | 作用 | 为什么重要 |
|------|------|------------|
| superblock | 全局参数、特性位、状态 | 决定如何解释整个文件系统 |
| block group | 局部化管理块和 inode | 提升分配局部性与管理效率 |
| group descriptor | 描述块组内位图、inode 表等位置 | 帮助找到局部元数据 |
| block bitmap | 记录哪些块空闲/已用 | 空间分配入口 |
| inode bitmap | 记录哪些 inode 空闲/已用 | 文件创建入口 |
| inode table | 保存 inode 元数据 | 对象元数据核心 |
| extent tree | 描述文件逻辑范围到物理块范围的映射 | 大文件和连续分配关键 |
| journal | 记录关键元数据更新事务 | 崩溃恢复关键 |

如果只记住“ext4 有 inode 和 block”，你仍无法解释它的性能和恢复语义。

### 8.3 extent 为什么替代传统块映射

早期文件系统常用直接块、间接块、双重间接块等方式描述文件内容位置。对大文件来说，这会带来大量指针和元数据开销。

extent 的核心思想是：用一个区间描述连续块。

```text
逻辑块 0..4095 -> 物理块 100000..104095
```

这带来几个收益：

- 大文件映射更紧凑
- 顺序读写更容易保持局部性
- 元数据树更浅，查找更少
- 预分配和延迟分配更容易表达

但 extent 不是魔法。如果磁盘空闲空间已经高度碎片化，extent 只能描述碎片，不能凭空制造连续空间。

### 8.4 delayed allocation 的收益与风险

延迟分配（delayed allocation）的直觉是：写入刚进入 page cache 时，先不要急着决定最终落在哪些物理块；等到回写时，系统能看到更大的写入范围，再做更好的分配决策。

收益：

- 更容易分配连续空间
- 减少碎片
- 合并小写入
- 降低短生命周期临时文件的无意义分配成本

风险：

- `write()` 返回时，物理块可能还没真正分配
- 崩溃时，应用以为“写过”的数据可能还停留在 cache 层
- 空间不足可能在回写阶段才暴露

这就是为什么 ext4 的性能优化和持久化语义必须一起看。

### 8.5 mballoc 与局部性策略

mballoc（multi-block allocator）的核心目标是一次性为较大范围分配连续空间，而不是每写一点就随便找一块。

它会考虑：

- 当前文件是否顺序增长
- 目标块组是否有足够连续空闲区间
- 是否应当把相关数据放在同一局部区域
- 是否需要为未来增长保留余地

所以 ext4 的分配器不只是“找空位”，而是在预测工作负载和维护未来空间形态。

### 8.6 HTree 目录索引解决什么问题

如果一个目录里有大量文件，线性扫描目录项会变得很慢。ext4 常用 HTree 一类哈希索引结构来改善大目录查找。

这说明目录也不是简单“文件名列表”。当目录变大时，它本身就是一种需要索引、分裂和维护的一类数据结构。

工程含义：

- 小目录和大目录的性能曲线不同
- 海量小文件会把压力放在目录索引、inode 分配和 dcache 上
- “磁盘吞吐高”不代表目录操作就快

### 8.7 JBD2：日志层不是 ext4 的附属小功能

ext4 的日志通常由 JBD2 处理。它把一批元数据更新组织成事务（transaction），再提交到日志区域。可以用这个简化流程理解：

```text
收集元数据修改 -> 写入 journal -> commit -> checkpoint 到主文件系统位置
```

几个关键概念：

- **transaction**：一批相关更新
- **commit**：事务在日志中达到可恢复的提交点
- **checkpoint**：把已提交事务的修改写回主位置，之后日志空间可复用
- **replay**：崩溃后根据日志重放已提交但未 checkpoint 完的事务

这让恢复不必全盘扫描所有结构，而是优先处理日志中最近的事务。

### 8.8 三种 journaling mode 的真实差异

ext4 的日志语义不能只说“有 journal”。关键在于数据是否进日志，以及数据和元数据顺序如何约束。

| 模式 | 直觉 | 优点 | 风险/代价 |
|------|------|------|-----------|
| `data=ordered` | 元数据提交前，相关数据通常先写出 | 常见默认折中，减少新元数据指向旧数据 | 不等于所有最新数据都不丢 |
| `data=writeback` | 元数据和数据顺序更宽松 | 性能可能更好 | 崩溃后更可能出现内容级旧数据现象 |
| `data=journal` | 数据和元数据都写入日志 | 语义更强 | 写放大更高，吞吐可能受影响 |

这也是为什么“我用 ext4，所以有日志，所以不会丢数据”是错的。日志模式影响的是恢复语义边界，而不是给应用一个万能事务。

### 8.9 barrier、flush 与设备缓存

即便 JBD2 正确提交了事务，底层设备也可能有自己的缓存和重排。文件系统需要依赖 barrier / flush 类机制来约束关键写入顺序。

这就是 `fsync` 昂贵的原因之一：它可能触发的不只是内存状态变化，还包括把关键写入推到设备稳定边界。

工程上要记住：

- 文件系统日志只能控制文件系统层看到的顺序
- 设备缓存和控制器实现会影响真实持久化边界
- 禁用或错误实现 flush/barrier 可能让一致性假设失效

### 8.10 orphan 处理与 unlink 生命周期

ext4 需要处理“目录项已删除，但 inode 还在使用”的对象。例如：

- 文件被 `unlink`
- 进程仍持有 fd
- 崩溃发生在回收过程之前

文件系统需要在恢复时知道哪些 inode 需要继续清理或回收。你不必记住所有实现细节，但要理解：删除文件不是单点动作，而是一套对象生命周期协议。

### 8.11 fast commit 的直觉

较新的 ext4 版本中存在 fast commit 一类优化思路：某些元数据更新可以记录更紧凑的增量，而不是总是走完整事务路径。它的目标是减少小型元数据更新的日志成本。

这里不需要展开实现细节，但它说明：

- ext4 仍在围绕“日志成本”持续优化
- 小文件、目录、元数据密集负载是重要场景
- 日志机制不是静态不变的一块功能，而是性能和恢复之间的持续工程折中

### 8.12 ext4 的挂载参数为什么会改变语义感受

ext4 的表现会受挂载参数影响，例如：

- 日志模式
- commit interval
- barrier/flush 相关设置
- atime 更新策略
- discard 策略

这些参数会改变性能、写放大、恢复窗口和延迟特征。不要把“ext4 的行为”理解成一个固定常量；它是文件系统实现、内核版本、挂载参数和设备行为共同作用的结果。

### 8.13 ext4 的边界在哪里

ext4 很强，但它不是为所有高级特性而生。它不以这些能力作为核心卖点：

- 原生跨设备池化
- 端到端数据校验和自修复
- 原生快照/子卷管理
- 分布式一致性
- 面向对象存储的语义适配

这些需求常常会把你带向 xfs、btrfs、zfs、CephFS、对象存储或应用层日志系统。

### 8.14 怎么判断一个 ext4 问题该从哪里查

面对 ext4 相关问题，可以先问：

1. 是空间分配问题，还是目录索引/元数据问题？
2. 是 page cache 语义误判，还是 journal 语义误判？
3. 是否涉及 `fsync(file)` 与 `fsync(dir)` 的边界？
4. 是否被挂载参数影响？
5. 是否是设备 flush/barrier 或虚拟化层引入的问题？

这比单纯说“ext4 稳定”更接近工程现实。

### 8.15 unwritten extent、`fallocate` 与“预分配并不等于已经有了真实数据”

当应用做预分配时，ext4 往往会使用 **unwritten extent** 一类机制：空间先被预留，但并不意味着对应逻辑范围已经拥有可读的真实业务数据。

这背后有几个很容易混淆的点：

- `fallocate()` 可能让文件“看起来已经占到空间”，但并不代表这些范围已被正常写入业务内容
- 预分配能降低后续空间不足和碎片化风险，却不能替代应用自己的提交协议
- 稀疏文件、预分配文件、真正写入完成的文件，在 extent 语义上不是同一种状态

工程上这很关键，因为很多系统会把：

- “空间已经留好了”
- “内容已经写好了”
- “崩溃后一定能恢复到这个状态”

误当成同一件事。它们其实分别属于分配、数据写入、持久化三个不同问题。

### 8.16 journal 真正保护什么，又明确不保护什么

初学者常把“ext4 有 journal”直接理解成“应用写入从此很安全”。更准确的说法是：

- journal 主要保护的是文件系统结构更新的可恢复性
- 它帮助元数据在崩溃后保持可解释，而不是替应用提供跨文件事务
- 即使在 `data=ordered` 下，也不能把“目录项已经可见”与“应用需要的所有新数据都永久可靠”画等号

换句话说，journal 解决的是：

- inode、目录项、分配位图这类结构如何避免在崩溃后自相矛盾

它没有直接解决：

- 多个文件之间的业务一致性
- 用户态缓存到内核缓存之间的提交语义
- 校验和、版本号、manifest 这类业务层恢复信息

所以高阶工程里常见的做法不是“信 ext4 就够了”，而是：

- 底层依赖 ext4 保证结构层恢复
- 应用层再用 WAL、版本文件、checksum 或 manifest 协议补逻辑恢复

### 8.17 e2fsck、metadata checksum 与 ext4 的恢复边界

ext4 的恢复并不只有“开机 replay journal”这一件事。还要理解：

- journal replay 更偏近期已提交事务的恢复
- e2fsck 处理的是更广义的结构检查和修复
- metadata checksum 等机制能帮助发现部分损坏，但发现不等于自动修复所有语义问题

这意味着 ext4 的恢复链条大致是：

1. 尽量通过 journal replay 快速回到结构一致状态
2. 在需要时通过 e2fsck 做更重的离线检查和修复
3. 对业务内容损坏、误写、应用级事务丢失，仍需要应用或上层协议自己负责

很多团队的问题不是“没 journal”，而是从未演练过：

- replay 能恢复什么
- fsck 会花多长时间
- 哪些数据结构错误会变成业务停机窗口

### 8.18 ext4、xfs、btrfs 的高阶选型差异，别只看“谁更现代”

如果把 ext4 放回工程对比里，更有价值的不是问“它是不是最先进”，而是问它和其他常见本地文件系统各在哪些方向取舍不同：

| 系统 | 更强的地方 | 更需要额外理解的地方 |
|------|------------|----------------------|
| ext4 | 默认行为稳健、工具链成熟、恢复经验丰富 | 不提供更强的快照/校验/池化能力，应用需自己补更多协议 |
| xfs | 并发元数据、大文件和扩展性场景通常表现更好 | 后台行为、调优和恢复心智要求更高 |
| btrfs | CoW、快照、校验、子卷等能力更丰富 | 写放大、碎片、运维和恢复复杂度更高 |

高阶理解不是背结论，而是知道：

- 你需要 ext4 提供什么
- 你准备用应用或平台补 ext4 没做的什么
- 你是否愿意为更强特性承担更高复杂度

### 8.19 JBD2 内部数据结构：transaction 和 journal block 的磁盘格式

JBD2（Journaling Block Device 2）是 ext4 的日志层，独立于 ext4 但专为其设计。理解其磁盘格式有助于理解 journal replay 的工作原理。

**journal superblock（日志超级块，存在 journal 设备或 journal 文件的第 1 块）**：

```c
/* include/linux/jbd2.h */
typedef struct journal_superblock_s {
    journal_header_t s_header;      /* magic 0xC03B3998 + blocktype 4 */
    
    /* 静态信息（格式化时写入，运行时不变）*/
    __be32  s_blocksize;            /* journal 块大小（通常等于文件系统块大小）*/
    __be32  s_maxlen;               /* journal 的总块数 */
    __be32  s_first;                /* 第一个日志数据块号 */
    
    /* 动态信息（挂载时更新）*/
    __be32  s_sequence;             /* 最期待的事务序列号（commit 时递增）*/
    __be32  s_start;                /* 最旧的未 checkpoint 事务的起始位置 */
    __be32  s_errno;                /* 日志中记录的错误码 */
    
    /* 特性位 */
    __be32  s_feature_compat;       /* 兼容特性 */
    __be32  s_feature_incompat;     /* 不兼容特性（如 64BIT, CSUM_V3）*/
    __be32  s_feature_ro_compat;
    __u8    s_uuid[16];             /* journal UUID（ext4 用于匹配 superblock）*/
    
    __be32  s_nr_users;             /* 共享此 journal 的文件系统数 */
    __be32  s_dynsuper;             /* 动态超级块用的 journal 块号 */
    __be32  s_max_transaction;      /* 单个事务最多包含的元数据块数 */
    __be32  s_max_trans_data;       /* 单个事务最多包含的数据块数 */
    __u8    s_checksum_type;        /* checksum 算法（CRC32C）*/
    __u8    s_padding2[3];
    __be32  s_padding[42];
    __be32  s_checksum;             /* journal superblock 自身 checksum */
    __u8    s_users[16*48];         /* 使用此 journal 的文件系统 UUID 列表 */
} journal_superblock_t;
```

**journal 磁盘格式（journal 文件的内容布局）**：

```
journal 文件内容（每块 4096 字节）：
  块 0：  journal superblock（上述结构体）
  块 1..：循环缓冲区，交替存放：
    ┌─────────────────────────────────────────────┐
    │ descriptor block（事务开始标记）              │
    │   magic = 0xC03B3998，blocktype = 1          │
    │   包含一张表：哪些块被此事务修改 + 序列号     │
    ├─────────────────────────────────────────────┤
    │ data block 1（被修改的元数据块的完整副本）    │
    │ data block 2                                 │
    │ ...                                          │
    ├─────────────────────────────────────────────┤
    │ commit block（事务提交标记）                  │
    │   magic = 0xC03B3998，blocktype = 2          │
    │   包含整个事务的 checksum                    │
    └─────────────────────────────────────────────┘
```

**replay 逻辑**：挂载时扫描 journal，对每个 `commit block`：
1. 验证 commit block 的 checksum
2. 把对应 descriptor block 中列出的每个块，从 journal 中的副本写回主文件系统
3. 所有已 commit 事务 replay 完成后，journal 标记为 clean

---

### 8.20 transaction 生命周期：从修改发起到 checkpoint

JBD2 的事务不是显式创建的——内核代码通过 `jbd2_journal_start()` 隐式加入或创建当前活跃事务。

```
事务状态机：
  T_RUNNING          ← 当前活跃事务，接受新的 handle
     │ (超时 / 块数超限 / jbd2_journal_stop)
     ▼
  T_LOCKED           ← 不再接受新 handle，等待已有 handle 完成
     │ (所有 handle 完成)
     ▼
  T_FLUSH            ← 刷新 data=ordered 的数据页
     │ (data 写出完成)
     ▼
  T_COMMIT           ← 把事务内容写入 journal 循环缓冲区
     │ (journal I/O 完成)
     ▼
  T_COMMIT_DFLUSH    ← 发送 flush 命令确保设备缓存到稳定存储
     │ (flush 完成)
     ▼
  T_COMMIT_JFLUSH    ← 写 commit block 到 journal
     │ (commit block 落盘)
     ▼
  T_FINISHED         ← 事务已提交，等待 checkpoint
     │ (checkpoint 把修改写回主文件系统)
     ▼
  T_NONE             ← 事务彻底完成，journal 空间可回收
```

**内核关键函数调用链**（文件：`fs/jbd2/transaction.c`，`fs/jbd2/commit.c`）：

```
ext4_write_inode()  或  ext4_mark_iloc_dirty()
   → jbd2_journal_start(journal, nblocks)      /* 获取 handle，加入当前事务 */
   → do_get_write_access(handle, bh)           /* 通知 journal 将修改这个 buffer */
   → 实际修改 buffer_head                      /* 修改元数据 */
   → jbd2_journal_dirty_metadata(handle, bh)   /* 标记 buffer 为 dirty（属于此事务）*/
   → jbd2_journal_stop(handle)                 /* 释放 handle 引用 */
        ↓ 如果引用计数降到 0
   → jbd2_journal_commit_transaction(journal)  /* 异步提交（由 kjournald2 线程执行）*/
        → journal_write_metadata_buffer()       /* 把每个修改的 buffer 写入 journal */
        → jbd2_journal_write_revoke_records()   /* 写 revoke 块（防止重播已删除块）*/
        → jbd2_journal_submit_commit_record()   /* 写 commit block */
        → jbd2_journal_wait_updates()           /* 等待 I/O 完成 */
        → journal_end_buffer_io_sync()          /* 确认 journal 落盘 */
```

**kjournald2 线程**：

```bash
# 观察 jbd2 内核线程（每个 ext4 文件系统一个）
ps aux | grep jbd2
# root       234  0.0  0.0     0     0 ?  S  00:00   0:00 [jbd2/sda1-8]
# root       235  0.0  0.0     0     0 ?  S  00:00   0:00 [jbd2/sda2-8]
# 命名格式：jbd2/<设备名>-<journal commit interval in seconds>
```

---

### 8.21 data=ordered 的实现细节：write_inode_now 与 writeback 顺序

`data=ordered` 是 ext4 的默认模式，其实现保证"新元数据提交前，相关数据页先写出"。具体机制：

**一次文件写入在 data=ordered 下的完整路径**：

```
1. write(fd, buf, len)
   → 数据进入 page cache，对应 page 标记为 dirty
   → 创建 journal handle，修改 inode size、extent tree（元数据）
   → 把 dirty data page 记录在事务的 t_iobuf_list（数据有序列表）
   → jbd2_journal_stop(handle)

2. 事务准备提交（T_FLUSH 状态）
   → 遍历 t_iobuf_list，对每个关联的 page 调用 writepage()
   → 等待这些 data page 的 I/O 完成（sync_writeback）
   → 此步骤确保：元数据（描述新数据的 extent）写入 journal 之前，
     数据本身先落盘

3. 数据 I/O 完成后
   → 把元数据写入 journal（descriptor block + metadata blocks）
   → 写 commit block
   → 发送 flush（或 FUA write）确保 journal 内容到达稳定存储

4. journal checkpoint（后台异步）
   → 把 journal 中的元数据副本写回主文件系统位置
   → 释放 journal 空间
```

**为什么 data=ordered 不能完全防止数据丢失**：

```
崩溃场景（data=ordered 仍无法防止）：
write(fd, new_data, 4096)   ← data page dirty，挂入 t_iobuf_list
  ↓
事务提交：data page 写出完成
  ↓
inode size 更新写入 journal ✓
  ↓
崩溃！commit block 未来得及写
  ↓
重启：replay 找不到已提交的 commit，旧元数据恢复
  ↓
结果：文件内容可能部分更新，但 inode size 仍是旧值
      → "长度对了但内容可能旧"的情况反过来变成
      → "内容可能新了但 inode 不知道"

结论：data=ordered 防止的是"新 inode 指向旧数据内容"（类型安全），
      不防止"最近一次 write 的内容完全丢失"（需要 fsync）。
```

**三种 journal 模式的内核实现差异一览**：

```bash
# 查看当前 ext4 分区的 journal 模式
tune2fs -l /dev/sda1 | grep "Default mount options"
# Default mount options: user_xattr acl

# 查看实际挂载参数
mount | grep ext4
# /dev/sda1 on / type ext4 (rw,relatime,errors=remount-ro)

# 修改 journal 模式（需重新挂载）
mount -o remount,data=writeback /mountpoint    # 性能最好，一致性最弱
mount -o remount,data=ordered /mountpoint      # 默认，平衡
mount -o remount,data=journal /mountpoint      # 最强一致性，最高写放大

# data=journal 的内存开销
# 每个被修改的数据块都需要在内存中维护两份：
# 1. 正在提交的事务的副本（journal 写入中）
# 2. 下一个事务正在修改的副本（new transaction buffer）
# 这就是 "double buffering" 的来源，内存需求较高
```

---

### 8.22 barrier、FUA write 与设备缓存的最后一公里

即使 JBD2 正确地按序写入 journal，底层设备的 write cache 仍可能使顺序承诺失效。

**设备缓存的问题**：

```
JBD2 认为的顺序：
  data page → journal metadata → commit block

设备缓存实际执行的顺序（可能被重排）：
  commit block → journal metadata → data page
  ↑ 如果断电在第一步后，看到 commit 但实际数据还没真正落盘
```

**三种强制顺序的机制**：

```bash
# 机制 1：FLUSH cache 命令（全局序列化点）
# → 发给设备 FLUSH CACHE 命令，设备必须把 write cache 全部刷到稳定存储
# → 昂贵：所有待处理写入都要完成，相当于全局 barrier
# → ext4 在 commit block 写入前发 FLUSH，确保 data 和 metadata 已落盘

# 机制 2：FUA（Force Unit Access）写
# → 单个写命令带 FUA 标志，设备必须保证此命令写入稳定存储再报 complete
# → 比 FLUSH 更细粒度，只强制特定写命令，不影响 write cache 中的其他数据
# → ext4 使用 FUA 写 commit block（确保 commit block 本身不在 write cache 里）

# 机制 3：write barrier（内核层面的序列化）
# → 内核 block layer 保证 barrier 之前的 write 先完成，再执行 barrier 之后的 write
# → 配合设备 FLUSH 或 FUA 实现
```

**观察 barrier 的开销**：

```bash
# 用 blktrace 观察 FLUSH 和 FUA 命令
blktrace -d /dev/sda -o - | blkparse -i - | grep -E "F|U"
# 8,0    3      1     0.000000000   234  F   N [kjournald2]   ← FLUSH 命令
# 8,0    3      2     0.001234567   234  FU  W [kjournald2]   ← FUA write

# 用 iostat 观察 FLUSH 频率（对应 journal commit 频率）
iostat -x 1 | grep sda
# 如果 w/s 高但 wkB/s 低，可能是大量小型 FLUSH 命令

# 禁用 barrier（危险！仅在虚拟机或 UPS 保护的场景，且需评估风险）
mount -o remount,barrier=0 /mountpoint  # 提升性能，但断电可能导致 journal 不一致
# 在虚拟化场景中，barrier=0 常见，因为 hypervisor 保证写入顺序

# 查看当前 barrier 状态
cat /sys/block/sda/queue/write_cache   # "write back"（有缓存）或 "write through"
cat /sys/block/sda/queue/fua           # 1 表示设备支持 FUA
```

**journal commit interval 调优**：

```bash
# 查看 commit interval（默认 5 秒：每 5 秒强制提交一次事务）
cat /proc/sys/vm/dirty_expire_centisecs  # 与 journal commit 相关但不完全相同
tune2fs -l /dev/sda1 | grep -i commit    # 暂无直接字段

# 修改 commit interval（挂载参数）
mount -o remount,commit=1 /mountpoint    # 每 1 秒提交（更高一致性，更多 FLUSH）
mount -o remount,commit=30 /mountpoint   # 每 30 秒提交（更高性能，更长恢复窗口）

# 对数据库工作负载（大量 fsync），commit interval 影响不大（fsync 强制立即提交）
# 对批量写入工作负载（少 fsync），较大 commit interval 可以合并更多 I/O
```

---

### 8.23 fast commit 与 journal 的性能演进

ext4 从 Linux 5.10 引入 fast commit 特性（由 Ext4 fast commit 项目），减少小型元数据操作的 journal 开销。

**传统 full commit 的成本**：

```
一次 rename() 的传统 journal commit：
  1. 写入 descriptor block（256 字节，但要占一个完整 journal 块 = 4096 字节）
  2. 写入旧父目录的完整块副本（4096 字节）
  3. 写入新父目录的完整块副本（4096 字节）
  4. 写入 inode 的完整块副本（4096 字节）
  5. 写入 commit block（4096 字节）
  总计：5 × 4096 = 20480 字节写入（只为了记录一个 rename 操作！）
```

**fast commit 的优化**：

```
一次 rename() 的 fast commit：
  1. 写入 fast commit header（标识这是 fast commit）
  2. 写入操作的增量日志（仅记录"从哪里 remove，add 到哪里"的操作描述）
  总计：~200-300 字节（减少 ~98% 的 journal 写入）

fast commit 适用的操作：
  ✓ rename（目录项移动）
  ✓ link/unlink（目录项增删）
  ✓ inode mtime/size 更新
  ✗ 数据块分配（仍需 full commit）
  ✗ 涉及多个块组的操作（可能无法用增量表达）
```

**启用 fast commit**：

```bash
# 检查内核是否支持 fast commit
grep "fast_commit" /sys/fs/ext4/*/features 2>/dev/null

# mkfs 时启用
mkfs.ext4 -O fast_commit /dev/sdX

# 对已有文件系统启用（需要内核 >= 5.10）
tune2fs -O fast_commit /dev/sdX

# 验证是否已启用
dumpe2fs /dev/sdX 2>/dev/null | grep "fast_commit"
# Filesystem features: ... fast_commit ...
```

---

## 实践观察

可以通过以下命令观察当前根分区文件系统类型与挂载参数：

```bash
findmnt /
findmnt -o TARGET,SOURCE,FSTYPE,OPTIONS /
cat /proc/mounts | head
```

这些命令不会让你看到所有内部结构，但能帮助你建立“当前系统具体挂了什么、带了什么参数”的意识。

---

## 本章小结

| 主题 | 结论 |
|------|------|
| ext4 | 是成熟稳健的通用本地文件系统，不是万能存储系统 |
| extent | 用区间描述连续块映射，降低大文件元数据成本 |
| delayed allocation | 改善分配质量，但让写入返回与真正分配之间出现距离 |
| JBD2 | 用 transaction / commit / checkpoint 支撑日志恢复 |
| journaling mode | 决定数据和元数据的顺序语义边界 |
| barrier / flush | 把文件系统顺序约束推向设备稳定边界 |
| 边界 | 快照、端到端校验、自修复、分布式语义不是 ext4 的核心卖点 |
| unwritten extent / `fallocate` | 说明“空间已预留”和“内容已可靠提交”不是一回事 |
| e2fsck / checksum | 恢复不仅是 journal replay，还包括更重的结构检查与修复 |

---

## 练习题

### 基础题

**8.1** extent 相比逐块记录的核心优势是什么？说明它如何降低大文件的元数据成本，以及在磁盘空闲空间高度碎片化场景下的局限性。

**8.2** `data=ordered`、`data=writeback`、`data=journal` 三种日志模式分别在数据与元数据写入顺序约束上有什么差异？各自对崩溃后内容恢复语义有什么影响？

### 中级题

**8.3** delayed allocation 为什么既提升性能，又增加持久化语义理解难度？描述一次 `write()` 返回到物理块真正分配之间可能发生的崩溃场景，以及对上层应用的影响。

**8.4** JBD2 的 transaction、commit、checkpoint 分别解决什么问题？画出一次完整元数据更新的流程，并说明崩溃时 journal replay 如何恢复一致状态。

### 提高题

**8.5** 为什么”ext4 有 journal”不能替代应用层可靠写入协议？区分 `fallocate()`/unwritten extent/`fsync()` 后三种状态分别保证了什么，并设计一个需要同时依赖 ext4 journal 语义和应用层 WAL 的场景，说明各层分别保护哪个不变量。

---

## 练习答案

**8.1** extent 用”起点+长度”描述连续块区间，一个 extent 可映射数千个连续块。优势：①元数据树更浅（最多 3 层），大文件查找更快；②顺序读写预读效率更高；③延迟分配和预分配更容易表达大连续区间；④减少元数据更新次数。局限：磁盘空闲空间已碎片化时，extent 只能描述已有碎片分布，无法凭空创造连续区间；极度碎片化场景下 extent tree 退化为大量短区间，元数据树加深，优势减弱，最终退化接近逐块记录。

**8.2** data=ordered：元数据写入 journal commit 之前，相关 dirty 数据页先 writeback 到磁盘；崩溃后元数据一致，数据可能丢失最近写入，但不会出现新元数据指向旧数据内容的情况。data=writeback：元数据和数据顺序宽松，数据 writeback 可晚于元数据 commit；崩溃后新 inode/extent 可能指向尚未更新的旧数据块，出现内容回退。data=journal：数据和元数据均写入 journal，崩溃后均可 replay；一致性语义最强，但写放大最高（每块数据写两次），适合对一致性极其敏感的场景。

**8.3** 性能收益：write() 返回时数据进入 page cache 但物理块未分配，延迟到回写时分配器能看到更大范围，做出更优的连续分配决策，合并小写入，减少短命文件的无意义分配。崩溃场景：write() 返回（应用认为成功）→ 内核准备回写 → 崩溃 → 物理块从未分配 → 数据永久丢失，而应用未收到任何错误。后果：应用不能以 write() 成功作为数据持久化的证明；需要 fsync() 确保数据和元数据均已落盘。

**8.4** transaction：将一批相关元数据修改组织为逻辑原子单元；commit：将 transaction 序列化写入 journal 区域并追加 commit record，之后即使崩溃也能通过 replay 恢复；checkpoint：将已 commit 事务的修改写回磁盘主位置，释放 journal 空间。流程：应用修改元数据 → 加入当前 transaction → transaction 满/超时 → JBD2 写 journal → 写 commit record → 后台 checkpoint 写主区域 → journal 空间回收。崩溃恢复：挂载时扫描 journal，找所有已 commit 未 checkpoint 的事务，按序 replay 到主区域，恢复一致状态；未 commit 的事务直接丢弃。

**8.5** journal 只保护文件系统元数据结构（inode、目录项、分配位图）的一致性，保证结构可解释；不保护多文件的业务事务一致性，不保证 write() 内容持久，不提供 checksum 或版本控制。三种状态：`fallocate()` 后为”空间已预留但内容未承诺（unwritten extent）”；`write()` 后为”内容在内存中但可能未持久”；`fsync()` 后为”内容已持久到设备稳定存储”。场景：数据库写 WAL 日志文件——ext4 journal 保证 WAL 文件的 inode 和目录项在崩溃后结构完整（能找到文件、大小信息正确）；应用层 WAL 保证事务逻辑顺序（按序重放）和业务数据完整性（checksum 校验）。两层分工：ext4 保证”能找到文件且结构不自相矛盾”，WAL 保证”业务数据正确且可按事务粒度恢复”。

---

## 延伸阅读

1. **Linux 内核文档**. `Documentation/filesystems/ext4/` — ext4 磁盘布局与元数据结构完整规范
2. **Mathur, Cao, et al.** *The new ext4 filesystem: current status and future plans* (OLS 2007) — extent tree 与延迟分配设计原文
3. **Tweedie**. *Journaling the Linux ext2fs filesystem* (LinuxExpo 1998) — JBD 日志机制的设计背景
4. **man 8 tune2fs / man 8 dumpe2fs** — 查看和修改 ext4 超级块参数及块组信息
5. **man 8 e2fsck** — ext4 离线检查与恢复工具，了解 `-b` 备份超级块用法

---

[← 上一章：块、超级块与空间分配](./07-blocks-superblocks-and-allocation.md)

[下一章：VFS、挂载与文件系统家族 →](./09-vfs-mount-and-filesystem-family.md)

[返回目录](../README.md)
