# 第10章：崩溃一致性与恢复

> **本章导读**：文件系统最难回答的问题不是”正常时怎么写”，而是”任意一个写入点突然断电、内核崩溃、设备重排、缓存未刷时，重启后还能解释成什么状态”。

**前置知识**：第8章（ext4 布局与日志机制）
**预计学习时间**：55 分钟

---

## 学习目标

完成本章后，你将能够：

1. 区分结构一致性、名字持久化、数据持久性和应用事务一致性
2. 理解写入重排、page cache、设备缓存、日志提交和 checkpoint 的关系
3. 解释 `write`、`fsync(file)`、`rename`、`fsync(dir)` 分别补哪个语义缺口
4. 认识 fsck、journal replay、copy-on-write、应用 WAL 各自工作的层次
5. 用故障矩阵分析一个文件更新协议是否真的可靠
6. 把多文件提交、torn write 和故障注入测试纳入一致性设计

---

## 正文内容

### 10.1 先定义失败模型

讨论一致性前，必须先问：失败发生在哪里？

| 失败类型 | 典型影响 |
|----------|----------|
| 进程崩溃 | 用户态 buffer 丢失，已进入内核的数据可能还在 |
| 内核崩溃 | page cache、未提交事务、运行时状态都可能丢失 |
| 断电 | 设备缓存、控制器重排、未 flush 数据都可能丢失 |
| 设备错误 | 某些块写失败或读回错误 |
| 网络/远端故障 | 远端文件系统语义可能出现超时、重试、分裂视图 |

“可靠写入”如果不说明失败模型，就是一句空话。

### 10.2 一致性不是一个词，而是四个层次

至少要分清四层：

1. **结构一致性**：inode、目录项、位图、extent tree、superblock 不互相矛盾
2. **名字持久化**：目录里是否稳定存在预期名字
3. **数据持久性**：文件内容是否稳定变成预期字节
4. **应用事务一致性**：业务是否能判断某次更新完整提交

文件系统日志主要帮助第一层和部分第二层；应用往往真正需要第四层。中间差距需要 `fsync`、协议、WAL、校验和恢复逻辑补上。

### 10.3 一次“保存文件”到底更新了什么

一个看似简单的保存动作，可能更新：

- 新数据页
- extent 映射
- inode size
- mtime / ctime
- block bitmap
- inode bitmap
- 目录项
- journal transaction
- 父目录元数据

这些更新不可能天然同时发生。崩溃一致性的核心问题就是：**如果只完成了其中一部分，重启后应该解释成什么状态？**

### 10.4 `write()` 成功只说明什么

`write()` 成功通常说明：数据已经被内核接受，或者至少拷入了某个内核路径。它不自动说明：

- 物理块已经分配
- 数据已经写到设备
- 设备缓存已经 flush
- 目录项已经持久化
- 应用事务已经提交

因此把 `write()` 成功当成“可靠落盘”，是很多持久化 bug 的根源。

### 10.5 `fsync(file)` 补的是文件层缺口

`fsync(file)` 试图推动该文件的数据和必要元数据到稳定存储边界。它主要补的是：

- 文件内容页的回写
- inode size 等必要元数据
- 文件系统层与设备层之间的提交/flush 路径

但它不天然覆盖所有父目录项语义。这就是为什么“写完临时文件并 `fsync` 它”还不等于“新名字已经可靠出现在目录里”。

### 10.6 `rename()` 补的是可见性/原子替换缺口

`rename()` 常用于可靠更新，因为它能让目录项切换更接近原子替换：

```text
旧版本仍可见 -> 新版本可见
```

理想情况下，读者不会看到“半个文件名”或“目标路径既不是旧版本也不是新版本”的中间状态。

但注意：`rename()` 的原子性偏向**名字可见性**，不等于：

- 新文件数据已经持久化
- 父目录项一定在断电后保留
- 并发写入冲突自动解决

### 10.7 `fsync(dir)` 补的是目录项持久化缺口

父目录本身也是一个对象。创建、删除、重命名文件都会改变目录内容。如果你希望“新名字关系”在崩溃后稳定存在，就要考虑父目录 `fsync`。

典型可靠更新协议：

```text
write temp
fsync(temp)
rename(temp, target)
fsync(parent directory)
```

这四步各自补不同语义缺口。少一步，不一定平时出错，但可能在断电测试里暴露。

### 10.8 日志事务和 checkpoint 不要混

日志文件系统里，事务提交到 journal 并不一定等于所有修改已经写回主位置。可以粗略理解为：

1. 修改进入事务
2. 事务写入 journal
3. 事务 commit，恢复时可重放
4. checkpoint 把修改写回主文件系统区域
5. journal 空间可以复用

重启后，系统可能通过 replay 把已提交但未 checkpoint 完的事务补到主结构中。

这解释了为什么 journal 能缩短恢复路径：恢复不必从全盘所有结构开始，而是优先看最近提交事务。

### 10.9 `data=ordered` 仍不是应用事务

即便在 `data=ordered` 这类更保守的模式下，也不能把文件系统日志当成应用事务。原因是：

- 它不知道你的业务记录边界
- 它不知道多个文件之间的提交关系
- 它不知道哪个 manifest 与哪个数据文件版本匹配
- 它不替你写 checksum、版本号和恢复逻辑

数据库和存储引擎为什么还要 WAL？因为应用事务边界通常比文件系统元数据事务更高层。

### 10.10 设备缓存与写屏障是最后一公里

如果设备或虚拟化层在收到写请求后继续缓存、重排、延迟 flush，那么文件系统层以为的顺序可能无法落到稳定介质。

所以一致性链条至少包括：

```text
应用协议 -> 系统调用 -> page cache -> 文件系统日志 -> 块层 -> 设备缓存 -> 稳定介质
```

这条链上任何一层误解语义，都会让“我明明 fsync 了”变成不可靠陈述。

### 10.11 fsck、journal replay、CoW、WAL 的分工

| 机制 | 工作层次 | 解决什么 | 不解决什么 |
|------|----------|----------|------------|
| fsck | 文件系统结构扫描 | 修复结构不一致 | 不知道应用事务语义 |
| journal replay | 文件系统日志恢复 | 重放已提交元数据事务 | 不保证最新业务数据完整 |
| copy-on-write | 写新版本再切换 | 减少原地覆盖中间态 | 可能增加写放大和碎片 |
| 应用 WAL | 应用事务层 | 记录业务提交边界 | 仍需底层持久化配合 |

高阶理解的关键是：不要拿某一层机制去替代另一层责任。

### 10.12 用故障矩阵检查协议

假设你更新配置文件，可以问：

| 崩溃点 | 期望结果 |
|--------|----------|
| 临时文件写一半崩溃 | 旧配置仍可用，临时文件可清理 |
| 临时文件写完但未 fsync | 新配置不一定可靠，恢复逻辑不能默认使用 |
| fsync 临时文件后 rename 前崩溃 | 旧配置仍是提交版本 |
| rename 后、fsync 目录前崩溃 | 取决于目录项是否已持久化，不能盲目假设 |
| fsync 目录后崩溃 | 新名字关系更可靠 |

如果你的程序说“我做了可靠写入”，至少应该能解释这张表。

### 10.13 分布式和容器会让一致性再复杂一层

在容器里，你还要问：

- 目标路径是不是 overlayfs 上层？
- 写入是否触发 copy-up？
- volume 是本地盘、网络盘还是对象存储网关？

在远端文件系统里，你还要问：

- close-to-open 边界是什么？
- 属性缓存多久失效？
- 锁和 rename 语义是否真的等同本地？

所以一致性不是第10章的独立话题，而会贯穿容器、分布式存储和应用设计。

### 10.14 一个工程原则

可靠持久化不是“调用某个神奇 API”，而是设计一条提交协议：

1. 写入可校验的新内容
2. 明确提交点
3. 持久化数据
4. 原子切换名字
5. 持久化目录项
6. 重启后能根据校验和版本号恢复

文件系统给你的是构件，不是完整业务事务。

### 10.15 多文件更新为什么比单文件替换更难

很多教程讲可靠写入时都用“单个配置文件替换”举例，但真实系统更常见的是：

- 数据文件 + 索引文件
- segment 文件 + manifest
- checkpoint 文件 + 元数据目录
- 多个 shard / 分区文件一起推进版本

这时真正困难的地方不是“怎么把每个文件都写好”，而是：

- 哪个对象定义“这一批已经提交”
- 崩溃后读者如何判断该信旧版本还是新版本
- 是否存在“新数据文件已经在，但旧 manifest 仍指向旧版本”这种中间态

高阶系统常用的思路不是让所有文件同时 magically 原子，而是：

- 先把新数据写成一组可校验对象
- `fsync` 它们各自的数据边界
- 最后只切换一个较小、可判断的“指针对象”，例如 manifest、版本文件或目录项
- 再把“指针对象”的名字关系持久化

这样崩溃恢复时，应用只需要回答：

- 当前哪个指针对象是可信的
- 指向的那一批数据是否完整可校验

而不是试图从一堆半更新文件里猜“系统原本想提交什么”。

### 10.16 torn write、部分覆盖与“记录级原子性”不要想当然

文件系统语义讨论里，一个常被忽略的陷阱是：即使你已经认真考虑 `fsync` 和 `rename`，也不能自动推出“应用记录一定按你想的粒度原子落盘”。

需要警惕的误判包括：

- 误以为一次 `write()` 对应的整条业务记录天然不可撕裂
- 误以为 4K、页大小、块大小、记录大小天然拥有同样的原子边界
- 误以为“文件存在且长度正确”就代表内容一定自洽

这也是为什么很多高阶系统会给每条记录或每个 segment 增加：

- 长度字段
- checksum
- 版本号 / sequence number
- footer / magic number

因为崩溃恢复里，真正关键的问题往往不是“文件还在不在”，而是：

- 最后半条记录是否应被忽略
- 当前文件尾部是否处在可解释状态
- manifest 指向的对象是否真的内容自洽

文件系统可以帮你维持结构层一致，但“记录级可解释性”通常仍属于应用协议。

### 10.17 crash testing 与故障注入为什么是高阶门槛

只靠逻辑推演很难确信一个协议真的可靠，因为人很容易漏掉某个崩溃点。高阶团队通常会把一致性设计变成实验对象，而不只留在文档里。

常见验证方式包括：

- 在可控环境里反复做中途 kill / reboot / 断电模拟
- 强制把崩溃点放在 `write`、`fsync(file)`、`rename`、`fsync(dir)` 之间
- 检查重启后系统是否能稳定落在“旧版本可用”或“新版本完整”两种结果之一
- 验证错误路径是否真的会让程序拒绝损坏状态，而不是默默接受

这类测试的重要性在于：

- 它能暴露“平时 100 次都没事，但第 101 次断电就坏”的问题
- 它能逼你把恢复判据写清，而不是只写“理论上应该可靠”
- 它能区分“结构能 replay”与“业务能恢复”是不是一回事

如果一个系统声称自己支持可靠写入，却从未做过 crash testing，它大概率只是拥有一套看起来合理的故事。

### 10.18 设计评审时，先把恢复判定表写出来

评审一个依赖文件系统持久化的系统时，与其反复争论”这样大概没问题”，不如先写一张恢复判定表：

| 崩溃点 | 重启后允许出现什么 | 必须拒绝什么 |
|--------|--------------------|--------------|
| 新数据写入中 | 旧版本继续有效 | 把半写入对象当成已提交版本 |
| 数据 `fsync` 前 | 旧版本有效，临时对象可丢 | 默认相信新数据已经稳定 |
| 指针/manifest 切换前 | 旧版本仍是提交版本 | 让新旧对象混杂组成”伪新版本” |
| 指针切换后、目录持久化前 | 结果取决于目录项是否稳定 | 不加检查地假设新版本必然可见 |
| 恢复阶段 | 通过 checksum / version 判定可信状态 | 看到文件存在就直接加载 |

只要这张表写不出来，协议就还没有真正设计完成。

---

### 10.19 torn write 的设备层机制：为什么 4K 对齐不等于原子性

torn write（撕裂写）是一个常被低估的崩溃一致性危险源。在应用层，人们常假设”一次 write() 系统调用的数据要么全写要么全不写”，但这个假设在设备层可能并不成立。

#### 物理写入单位 vs 逻辑块大小

```
块设备的三种写入粒度：

1. 逻辑扇区大小（Logical Sector Size）
   → 块设备向操作系统暴露的寻址单位
   → 历史上 512B，现代设备通常 4096B（512e 或 4Kn）
   → 由 blockdev --getss /dev/sda 或 cat /sys/block/sda/queue/logical_block_size 查看

2. 物理扇区大小（Physical Sector Size）
   → 设备内部的实际写入原子单位
   → 大多数现代 SSD/HDD：4096B
   → 由 cat /sys/block/sda/queue/physical_block_size 查看

3. 擦写块大小（Erase Block Size，SSD 专有）
   → SSD 需要按擦写块为单位清零后才能写入新数据
   → 通常 128KB ~ 2MB，远大于物理扇区
   → 写入小于擦写块的数据时，设备内部做 read-erase-write，增加写放大
```

**torn write 发生条件**：

当文件系统块大小（通常 4096B）恰好等于设备物理扇区大小时，单个文件系统块的写入在设备层是原子的（要么写完要么不写）。但以下情况会导致 torn write：

```
场景 1：512B 逻辑扇区的老式设备
  文件系统块 = 4096B = 8 × 512B 扇区
  设备按扇区写入，断电可能只完成前 4 个扇区
  结果：文件系统块的前 2048B 是新数据，后 2048B 是旧数据
  → torn write

场景 2：4096B 块但跨两个物理扇区边界的写入
  某些 NVMe 设备的写入原子单位是 512B（Physical Sector Size = 512B）
  即使用 4096B 写入，设备内部可能分多次提交
  → 断电可能导致 4096B 内部撕裂

场景 3：RAID 5/6 stripe write
  RAID 5 的 stripe 通常远大于文件系统块
  写入未对齐的小数据需要 read-modify-write 一整个 stripe
  断电在 read-modify-write 中间 → parity 与数据不一致（RAID write hole）

场景 4：日志文件系统的 journal commit block
  JBD2 在 journal 中写 commit block 标记事务完成
  如果 commit block 的 4096B 本身发生 torn write
  → replay 时可能读到部分写入的 commit block
  → JBD2 通过 commit block 的 checksum 检测并拒绝此事务
```

**查看设备的写入原子单位**：

```bash
# 查看物理/逻辑扇区大小
cat /sys/block/sda/queue/logical_block_size    # 逻辑扇区（操作系统视角）
cat /sys/block/sda/queue/physical_block_size   # 物理扇区（设备内部原子写入）
cat /sys/block/sda/queue/minimum_io_size       # 建议最小 I/O 大小
cat /sys/block/sda/queue/optimal_io_size       # 最优 I/O 大小（RAID stripe）
blockdev --getpbsz /dev/sda                    # 物理扇区大小（字节）

# NVMe 的写入原子单位（AWUN: Atomic Write Unit Normal）
nvme id-ns /dev/nvme0n1 -H | grep -i atomic
# Atomic Write Unit Normal (AWUN): 0 (1 logical block)
# Atomic Write Unit Power Fail (AWUPF): 0 (1 logical block)
# → AWUN=0 表示 1 个逻辑块（512B 或 4096B）是原子的

# 查看 SSD 是否支持 Atomic Write（较新 NVMe 特性）
nvme id-ctrl /dev/nvme0 | grep -i “atomic\|awun\|awupf”
```

#### 数据库如何防范 torn write

```
PostgreSQL 的 full_page_write 机制：

正常写入流程（无 torn write 保护）：
  事务修改 page（8KB）
  → WAL 只记录变更（delta）
  → checkpoint 时把 dirty page 写到磁盘
  → 如果写 page 时断电，page 可能半写（torn）
  → redo 日志只有 delta，无法在撕裂 base page 上正确 redo

full_page_write 保护机制：
  checkpoint 后第一次修改某 page 时
  → WAL 中记录完整的 page 快照（8KB full image）
  → redo 时如果发现 torn page，用 WAL 中的完整快照覆盖
  → 之后的 delta redo 基于可信的完整 page

MySQL InnoDB 的 doublewrite buffer 机制：
  page 在写入到真实位置之前，先写入 doublewrite buffer（连续区域）
  → doublewrite buffer 写完成（fsync）
  → 再写真实位置
  → 如果真实位置写到一半断电：重启时检测到 torn page
    → 从 doublewrite buffer 恢复完整 page
    → 再基于 redo log 重放
```

**PostgreSQL full_page_write 的实际效果**：

```bash
# 查看 PostgreSQL full_page_write 设置
psql -c “SHOW full_page_writes;”
# full_page_writes
# -----------------
# on
# (已启用，保护 torn write 场景)

# 禁用 full_page_write（危险，只在有下层保护时才考虑）
# postgresql.conf: full_page_writes = off
# → 仅当底层设备保证原子写入 8KB（如 NVMe with AWUN >= 16）时才安全

# WAL 中 full page write 的体积影响
psql -c “SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), '0/1000000');”
# 启用 full_page_write 后，checkpoint 后期的 WAL 明显更大
```

---

### 10.20 应用层 WAL 实现模式：从 SQLite 到 LevelDB

理解应用 WAL 不只是”知道有日志”，而是要理解具体记录什么、何时提交、如何恢复。

#### WAL 的基本结构

```
WAL 文件（顺序追加，崩溃后从尾部向前扫描）：

  ┌──────────────────────────────────────────────────────┐
  │ LSN=1  BEGIN TXN=100                                 │  ← 事务开始
  │ LSN=2  UPDATE page=42 offset=128 len=64 data=...     │  ← 操作记录
  │ LSN=3  UPDATE page=7  offset=0   len=512 data=...    │
  │ LSN=4  COMMIT TXN=100 checksum=0xABCD1234            │  ← 提交标记+校验
  │ LSN=5  BEGIN TXN=101                                 │
  │ LSN=6  UPDATE page=99 offset=512 len=128 data=...    │
  │ LSN=7  COMMIT TXN=101 checksum=0xDEADBEEF            │
  │ LSN=8  BEGIN TXN=102                                 │
  │ LSN=9  UPDATE page=3  offset=0   len=256 data=...    │
  │                                                      │  ← 崩溃点（TXN=102 未提交）
  └──────────────────────────────────────────────────────┘

恢复逻辑：
  1. 从 WAL 开头向后扫描
  2. 验证每个 COMMIT 的 checksum
  3. 找到 checksum 有效的最后一个 COMMIT（此处为 LSN=7，TXN=101）
  4. 将 TXN=100 和 TXN=101 的操作 redo 到主文件
  5. LSN=8 之后的未提交事务（TXN=102）丢弃
```

#### SQLite WAL 模式实现

SQLite 的 WAL 模式（WAL mode）是一个精心设计的单文件 WAL 实现，值得深入理解。

```
SQLite WAL 模式的核心数据结构：

主数据库文件（database.db）：
  → 稳定已提交的数据（checkpoint 之后的状态）
  → 读者优先读这里

WAL 文件（database.db-wal）：
  → 所有新写入都先追加到这里
  → 多个读者可以并发访问（读 WAL 或读主文件）
  → WAL header（32 字节）+ 一系列 WAL frame（页大小 + 24 字节 frame header）

WAL frame header（24 字节）：
  pgno        4B   // 此 frame 对应的数据库 page 号
  szPage      4B   // page 大小（第一帧有效，后续为 0）
  mxFrame     4B   // 当前写入的最大 frame 号（commit marker 用）
  aDbSize     4B   // commit 时数据库的 page 数（commit 帧有效）
  aFrameCksum 8B   // 此 frame 的 checksum（累积计算）

WAL index（database.db-shm，共享内存）：
  → 内存中的哈希表，page_no → 最新 WAL frame 号
  → 加速读者”找到这个 page 最新版本在哪个 frame”
  → 可以从 WAL 文件重建，不需要持久化
```

**SQLite WAL 的读写并发模型**：

```
WAL 模式的并发优势（相比 journal 模式）：
  读者：读主文件 + WAL 中的最新版本，不阻塞写者
  写者：追加到 WAL，不修改主文件，不阻塞读者
  checkpoint：把 WAL 中已提交的页写回主文件，需要独占访问

journal 模式（对比）：
  写者获取写锁 → 旧数据写 rollback journal → 修改主文件
  读者在写者持锁期间必须等待

SQLite WAL 读事务流程：
  1. 获取 WAL 读锁（共享，允许多个读者并发）
  2. 记录当前 mxFrame（WAL 的最大有效 frame 号）
  3. 读某 page 时：查 WAL index 找此 page 最近一次在 WAL 中的 frame
     - frame 号 <= mxFrame → 从 WAL 中读（最新已提交版本）
     - 不在 WAL → 从主文件读
  4. 释放读锁（不需要提交）

SQLite WAL 写事务流程：
  1. 获取写锁（独占，只允许一个写者）
  2. 将修改的每个 page 追加到 WAL
  3. 最后追加 commit frame（mxFrame 字段反映提交点）
  4. fsync WAL 文件
  5. 更新 WAL index（共享内存）
  6. 释放写锁
```

**命令行验证 SQLite WAL 模式**：

```bash
# 切换到 WAL 模式
sqlite3 mydb.db “PRAGMA journal_mode=WAL;”
# journal_mode
# WAL

ls -la mydb.db*
# mydb.db       ← 主数据库
# mydb.db-shm   ← WAL index（共享内存映射）
# mydb.db-wal   ← WAL 文件

# 写入数据观察 WAL 增长
sqlite3 mydb.db “INSERT INTO t VALUES (1, 'test');”
ls -lh mydb.db mydb.db-wal
# mydb.db     → 大小不变（数据还在 WAL）
# mydb.db-wal → 增大（新数据写在这里）

# 触发 checkpoint（将 WAL 内容写回主文件）
sqlite3 mydb.db “PRAGMA wal_checkpoint(FULL);”
# 0|N|N  ← (result=0, wal_frames, checkpointed_frames)

# 查看 WAL 状态
sqlite3 mydb.db “PRAGMA wal_checkpoint;”
sqlite3 mydb.db “PRAGMA journal_mode;”
```

#### LevelDB/RocksDB 的 WAL 与 memtable 设计

```
LevelDB 写入路径：

  write(key, value)
    │
    ├── 1. 追加到 WAL 文件（顺序写，O(1) 持久化）
    │      WAL 记录格式：
    │      ┌─────────────────────────────────────────┐
    │      │ checksum(4B) length(2B) type(1B) data... │
    │      └─────────────────────────────────────────┘
    │      type: kFullType(1) / kFirstType(2) / kMiddleType(3) / kLastType(4)
    │      （大记录可跨多个 32KB block，用 type 标记片段关系）
    │
    ├── 2. 写入 memtable（内存中的 SkipList，O(log n)）
    │
    └── 3. 返回（此时写入已”持久化”到 WAL + 内存 memtable）

  memtable 满时（默认 4MB）：
    → 冻结为 immutable memtable
    → 启动 compaction 线程将 immutable memtable 写到 SSTable（Level-0）
    → 新 WAL 文件开始记录

  崩溃恢复：
    → 扫描 WAL 文件（可能有多个，按序号排列）
    → 验证每条记录的 checksum
    → 跳过 checksum 错误的记录（truncate 到最后有效记录）
    → 将有效记录重放到新 memtable
    → memtable 写入 SSTable 完成后，删除旧 WAL
```

**RocksDB WAL 的可观察性**：

```bash
# 查看 RocksDB WAL 文件
ls -lh /path/to/rocksdb/data/*.log
# 000001.log  ← 当前活跃 WAL（数字是序列号）
# 000002.log  ← 可能是等待 flush 的旧 WAL

# RocksDB 的 WAL 统计（通过 db_stats）
db.GetProperty(“rocksdb.stats”)
# ...
# ** Compaction Stats [default] **
# ...
# WAL:
# Writes: 12345  Syncs: 1234  Write With Wal: 12345  Bytes: 123456789
# WAL syncs: 数字越大说明调用 fsync 越频繁

# 调整 WAL sync 策略（权衡持久性与性能）
# Options::wal_bytes_per_sync = 0（默认，每次 write 同步）
# Options::wal_bytes_per_sync = 1024*1024（1MB 批量 sync，提升吞吐但增加丢失窗口）
```

---

### 10.21 crash testing 工具链：从理论到实验验证

#### dm-flakey：内核级故障注入

`dm-flakey` 是 Linux Device Mapper 的一个目标类型，可以模拟设备在特定时间段内的写入失败或数据损坏，是崩溃一致性测试的重要工具。

```bash
# 创建 dm-flakey 设备（在真实设备或 loop device 上）
# 参数：up_interval down_interval [feature...]
# up_interval：正常工作的秒数
# down_interval：返回错误的秒数

# 准备 loop device
dd if=/dev/zero of=/tmp/test.img bs=1M count=100
LOOP=$(losetup -f --show /tmp/test.img)
# LOOP=/dev/loop0

# 获取设备大小（扇区数）
SECTORS=$(blockdev --getsz $LOOP)

# 创建 dm-flakey：每 30 秒正常工作，随后 5 秒模拟写失败
echo “0 $SECTORS flakey $LOOP 0 30 5” | dmsetup create flakey-test

# 在 flakey 设备上创建文件系统并挂载
mkfs.ext4 /dev/mapper/flakey-test
mount /dev/mapper/flakey-test /mnt/test

# 运行负载测试（在 down_interval 期间会遇到 I/O 错误）
for i in $(seq 1 1000); do
    echo “data $i” > /mnt/test/file_$i
    sync
done &

# 等待几个 up/down 周期后，模拟突然断电：强制卸载并检查
sleep 70
umount -l /mnt/test
e2fsck -n /dev/mapper/flakey-test  # -n 表示只检查不修复

# 清理
dmsetup remove flakey-test
losetup -d $LOOP
```

**dm-flakey 的高级 feature**：

```bash
# feature 1: drop_writes（直接丢弃写请求，不报错）
# 模拟”写入成功但数据没有落盘”（write cache 场景）
echo “0 $SECTORS flakey $LOOP 0 10 5 1 drop_writes” | dmsetup create flakey-drop

# feature 2: error_writes（写入返回 EIO 错误）
# 默认行为

# feature 3: corrupt_bio_byte N percent direction
# 随机损坏写入数据的第 N 字节（percent% 概率，direction: w=写/r=读）
echo “0 $SECTORS flakey $LOOP 0 30 5 3 corrupt_bio_byte 10 50 w 0” | \
    dmsetup create flakey-corrupt
```

#### ALICE：崩溃一致性系统级测试框架

ALICE（Application-Level Intelligent Crash Explorer）是 OSDI 2014 论文的配套工具，系统化地探索崩溃点空间。

```
ALICE 工作原理：

1. strace 记录：录制应用的完整系统调用序列
   strace -e trace=file,write,read,fsync,rename,open,close,unlink \
          -o syscalls.log ./application_under_test

2. 构建系统调用图：
   → 提取 write/pwrite 操作
   → 记录每次 write 的 fd、offset、length
   → 追踪 fsync/fdatasync/sync 调用
   → 记录 rename、unlink 等元数据操作

3. 生成崩溃状态：
   → 对每个可能的崩溃点（write 之间的间隙）
   → 枚举哪些写入已到磁盘、哪些未到
   → 考虑 write 内部的 partial write（torn write）
   → 每种组合产生一个”崩溃状态”

4. 验证崩溃后应用行为：
   → 将每个崩溃状态应用到文件系统
   → 启动应用程序的恢复/检查逻辑
   → 检查应用是否报告一致或数据丢失

典型发现（原论文）：
   测试 11 个主流应用（LevelDB、HSQLDB、Git、SQLite 等）
   发现 60 个崩溃一致性 bug
   其中许多在正常测试中从未触发
```

**简化的崩溃模拟脚本**：

```bash
#!/bin/bash
# 手动崩溃点注入测试框架（简化版）

TEST_DIR=/mnt/crash-test
PROTOCOL_SCRIPT=”$1”  # 要测试的协议脚本

crash_and_check() {
    local crash_point=”$1”  # 在哪一步崩溃

    echo “=== 测试崩溃点: $crash_point ===”

    # 1. 准备干净的文件系统状态
    umount $TEST_DIR 2>/dev/null
    mkfs.ext4 -q /dev/mapper/flakey-test
    mount /dev/mapper/flakey-test $TEST_DIR

    # 2. 建立”旧版本”基准状态
    echo “old config v1” > $TEST_DIR/config.dat
    fsync $TEST_DIR/config.dat
    sync

    # 3. 注入崩溃点（通过 flakey 设备）
    case “$crash_point” in
        “after_write”)
            # 写完临时文件后崩溃（在 fsync 之前）
            echo “new config v2” > $TEST_DIR/config.tmp
            # 不调用 fsync，模拟崩溃
            echo b > /proc/sysrq-trigger  # 强制重启（真实测试用）
            ;;
        “after_fsync”)
            echo “new config v2” > $TEST_DIR/config.tmp
            sync $TEST_DIR/config.tmp
            # rename 之前崩溃
            echo b > /proc/sysrq-trigger
            ;;
        “after_rename”)
            echo “new config v2” > $TEST_DIR/config.tmp
            sync $TEST_DIR/config.tmp
            mv $TEST_DIR/config.tmp $TEST_DIR/config.dat
            # fsync(dir) 之前崩溃
            echo b > /proc/sysrq-trigger
            ;;
    esac

    # 4. 模拟重启后检查（对 loop device 场景不真实重启，而是重新挂载）
    umount $TEST_DIR
    mount /dev/mapper/flakey-test $TEST_DIR

    # 5. 检查恢复状态
    echo “重启后状态：”
    cat $TEST_DIR/config.dat 2>/dev/null || echo “(文件不存在)”
    ls $TEST_DIR/config.tmp 2>/dev/null && echo “WARNING: 临时文件仍然存在”
}

# 运行所有崩溃点测试
for point in “after_write” “after_fsync” “after_rename”; do
    crash_and_check “$point”
done
```

#### ext4 自带的错误注入（debugfs）

```bash
# 使用 debugfs 注入磁盘错误（需要 CONFIG_EXT4_DEBUG）
debugfs -w /dev/sda1 << 'EOF'
# 损坏特定 inode 的 extent tree（用于测试 e2fsck 恢复能力）
set_inode_field <12345> i_blocks_hi 0xDEAD
EOF

# 验证 e2fsck 能检测并修复
e2fsck -f /dev/sda1
# e2fsck 1.46: checking /dev/sda1
# Pass 1: Checking inodes, blocks, and sizes
# Inode 12345 has illegal block(s). FIXED.

# 使用 debugfs 查看 journal 内容
debugfs -R “logdump -a” /dev/sda1 | head -50
# Journal starts at block 1, transaction 42
# Found expected sequence 42, type 1 (descriptor block) at block 1
# Dumping descriptor block, sequence 42, at block 1:
#   FS block 8192 logged at journal block 2 (flags 0x0)
# ...
```

---

### 10.22 `fdatasync` vs `fsync`：在持久化成本上做精确切割

```
fsync(fd)：
  → 刷新文件数据（dirty page）+ 所有元数据（inode: size, mtime, ctime）
  → 对于追加写的日志文件，mtime 也要落盘，多一次元数据写

fdatasync(fd)：
  → 刷新文件数据 + 仅刷新影响数据可读性的元数据（如文件大小）
  → 不刷新 mtime/ctime/atime 等”时间类元数据”
  → 比 fsync 少一次元数据 I/O（如果文件大小未变化）

实际差异（何时 fdatasync 比 fsync 快）：
  追加写场景：fdatasync 需要更新 i_size，仍需元数据 I/O，差距不大
  覆写场景（文件大小不变）：fdatasync 不更新 mtime → 省去一次 inode 写
  对于高频 fsync 的数据库日志：使用 fdatasync 可减少 10-20% inode 写 I/O

O_DSYNC 标志（更细粒度的 fdatasync 语义）：
  open(“wal.log”, O_WRONLY|O_DSYNC)
  → 每次 write() 完成后自动做 fdatasync 语义的刷新
  → 适合 WAL 文件：每次写入立即持久化，无需手动调用 fdatasync
```

**内核实现差异**：

```bash
# 用 strace + perf 观察 fsync vs fdatasync 的系统调用差异
strace -T -e trace=fsync,fdatasync dd if=/dev/zero of=/tmp/test bs=4K count=1000 \
    oflag=dsync 2>&1 | tail -10
# write(1, “”, 4096) = 4096 <0.000023>       ← O_DSYNC 的每次写
# ... 每次写后内核自动做 fdatasync 语义

# 比较 fsync 和 fdatasync 的实际延迟
python3 -c “
import os, time

f = open('/tmp/fsync_test', 'w+b')
f.write(b'x' * 4096)

t0 = time.monotonic()
for _ in range(100):
    f.write(b'x' * 4096)
    os.fsync(f.fileno())
t1 = time.monotonic()
print(f'fsync: {(t1-t0)*10:.1f} ms per call')

f.seek(0)
t0 = time.monotonic()
for _ in range(100):
    f.write(b'x' * 4096)
    os.fdatasync(f.fileno())
t1 = time.monotonic()
print(f'fdatasync: {(t1-t0)*10:.1f} ms per call')
f.close()
“
# fsync:     15.2 ms per call   （含 mtime 更新）
# fdatasync: 13.8 ms per call   （跳过 mtime，略快）
```

---

## 本章小结

| 主题 | 结论 |
|------|------|
| 失败模型 | 不说明失败点，就无法讨论可靠性 |
| 一致性层次 | 结构、名字、数据、应用事务不是一回事 |
| `write` | 只说明内核接受写入，不等于稳定落盘 |
| `fsync(file)` | 补文件数据与必要元数据持久化缺口 |
| `rename` | 补名字切换的原子可见性缺口 |
| `fsync(dir)` | 补目录项持久化缺口 |
| journal | 保护文件系统结构事务，不替代应用 WAL |
| 设备缓存 | 是持久化语义的最后一公里 |
| 多文件提交 | 通常需要 manifest/指针对象定义真正的提交版本 |
| torn write | 说明记录级可解释性通常仍要由应用自己补 |
| crash testing | 是验证协议而不是验证故事的关键手段 |

---

## 练习题

### 基础题

**10.1** 结构一致性、名字持久化、数据持久性、应用事务一致性有什么区别？说明文件系统 journal 主要覆盖哪层，应用 WAL 主要覆盖哪层。

**10.2** 为什么 `write()` 成功不能说明数据已经可靠持久化？描述 `write()`→`fsync(file)`→`rename()`→`fsync(dir)` 四步各自补哪个语义缺口。

### 中级题

**10.3** 为什么文件系统 journal 不能替代数据库 WAL？从”文件系统保证什么、WAL 保证什么”两个角度分析，并说明 `data=ordered` 模式下仍需要应用层 WAL 的根本原因。

**10.4** 设计一个配置文件可靠更新协议，列出所有关键步骤，并用崩溃矩阵说明每个崩溃点的期望恢复结果与禁止出现的状态。

### 提高题

**10.5** 如果系统需要原子推进 3 个数据文件和 1 个 manifest，真正的提交点应该放在哪里？为该协议设计 crash testing 方案（至少覆盖 5 个崩溃点），并解释为什么”文件长度对了”不能证明最后一条记录完整可用（提示：torn write、checksum、序列号）。

---

## 练习答案

**10.1** 结构一致性：inode、目录项、位图、extent tree 不互相矛盾，文件系统结构可被正确解释；名字持久化：目录中预期名字在崩溃后稳定存在；数据持久性：文件内容以预期字节稳定落盘；应用事务一致性：业务逻辑能判断某次多步更新是否完整提交。journal 主要覆盖结构一致性和部分名字持久化（data=ordered 还部分覆盖数据持久性顺序）；应用 WAL 主要覆盖应用事务一致性，通常也依赖底层提供数据持久性。

**10.2** write() 成功说明数据进入内核路径（page cache 或内核缓冲区），不说明物理块已分配、设备已写入、设备缓存已 flush。四步语义：fsync(file) 将文件 dirty 数据页和必要 inode 元数据推到设备稳定边界；rename() 将目录项原子切换，使新文件名对读者可见而不出现中间态；fsync(dir) 将父目录 dirty 元数据持久化，确保新名字关系在崩溃后仍存在；完整协议才能覆盖”内容可靠 + 名字原子切换 + 名字持久化”三个缺口。

**10.3** 文件系统 journal 保护文件系统元数据结构的原子性（inode/目录项/位图不自相矛盾），不知道业务记录边界、多文件提交关系、版本号和 checksum。data=ordered 仍需 WAL 的原因：①journal 不知道哪两个文件的更新属于同一业务事务；②journal 不记录 checksum 或版本号，无法判断业务记录是否完整；③多个数据文件之间的一致性约束需要应用自己定义提交点（manifest）并在 WAL 中记录。数据库 WAL 记录每个事务的 begin/end 边界、操作日志和 checksum，崩溃后 redo/undo 到一致事务边界，这些是文件系统 journal 从设计上就不提供的能力。

**10.4** 协议步骤：①write(tmpfile, new_content)；②fsync(tmpfile)（数据持久化）；③rename(tmpfile, target)（名字原子切换）；④fsync(parent_dir)（目录项持久化）。崩溃矩阵：
- 步骤①中崩溃：tmpfile 部分写入或不存在 → 旧配置不变，临时文件清理即可 ✓
- 步骤①②之间崩溃：tmpfile 存在但数据未持久 → 旧配置仍有效，临时文件视为无效 ✓
- 步骤②③之间崩溃：tmpfile 数据已持久但 rename 未发生 → 旧配置仍是有效版本 ✓
- 步骤③④之间崩溃：rename 可能已发生，但目录项未必持久 → 结果取决于日志模式；协议必须能接受旧版本仍然有效 ✓，不能假设新版本必然出现
- 步骤④后崩溃：新名字已持久 → 新配置为有效版本 ✓
禁止出现：新旧内容混合的”伪版本”；空文件被当成有效配置。

**10.5** 提交点设计：先对 3 个数据文件各自执行 write+fsync，确保内容持久；然后写 manifest（包含 checksum 和版本号）并 fsync(manifest)；最后如需原子可见，对 manifest 做 rename+fsync(dir)。manifest 切换是唯一的提交点，崩溃后根据 manifest 存在性和版本号判断是否已提交。Crash testing 覆盖点：①数据文件 1 写入中；②数据文件 3 fsync 后 manifest 写入前；③manifest write 后 fsync 前；④manifest fsync 后 rename 前；⑤rename 后 fsync(dir) 前。”文件长度对了”不能证明完整：torn write（扇区内的部分写入）可能使最后几字节未落盘但文件 size 已更新（crash 在写入中间，设备缓存丢失部分数据）；没有 checksum 无法区分完整记录和被截断的记录；没有 sequence number 无法知道写入是否处于预期位置；正确做法：每条记录加 length + checksum，回放时逐条验证，发现 checksum 错误则截断到上一条有效记录。

---

## 延伸阅读

1. **Pillai, et al.** *All File Systems Are Not Created Equal: On the Complexity of Crafting Crash-Consistent Applications* (OSDI 2014) — 分析主流文件系统崩溃一致性行为差异
2. **Arpaci-Dusseau**. *Operating Systems: Three Easy Pieces*, Ch.42 — 崩溃一致性与日志
3. **man 2 fsync / man 2 fdatasync** — fsync 与 fdatasync 语义区别（fdatasync 不更新 mtime，但更快）
4. **LWN.net**. *Ensuring data reaches disk* — `fsync`、`rename`、barrier 完整语义分析
5. **SQLite 文档**. *How SQLite Works* — WAL 模式崩溃一致性设计参考实例

---

[← 上一章：VFS、挂载与文件系统家族](./09-vfs-mount-and-filesystem-family.md)

[下一章：页缓存、脏页与回写 →](../part4-cache-and-io/11-page-cache-and-read-write-path.md)

[返回目录](../README.md)
