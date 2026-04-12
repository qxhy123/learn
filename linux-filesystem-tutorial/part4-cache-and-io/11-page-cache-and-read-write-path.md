# 第11章：page cache 与读写路径

> page cache 不是“顺手加的缓存层”，而是 Linux 文件 I/O 的核心执行面；很多性能现象、持久化误解和内存压力问题，本质上都要从它这里解释。

## 学习目标

完成本章后，你将能够：

1. 说明 page cache、address_space、readahead、writeback 之间的关系
2. 理解读取命中、页缺失、脏页生成、回写和回收是如何串起来的
3. 解释 writeback error、dirty throttling、cache reclaim 为什么会改变应用感受
4. 区分 page cache、buffer cache、direct I/O、mmap 在路径上的不同位置
5. 用缓存与回写模型解释“为什么明明没写盘却看起来很快”这类现象
6. 理解 workingset/refault、容器内存限制和观测证据如何改变你对 cache 结论的判断

---

## 正文内容

### 11.1 page cache 是文件内容的运行时工作面

在 Linux 中，普通文件 I/O 很多时候先落到 page cache，而不是立刻直达设备。可以粗略理解为：

- 文件系统对象提供逻辑内容视图
- page cache 提供内存中的页级表示
- 回写线程再把这些页同步到底层存储

因此 page cache 不是“附加优化”，而是文件 I/O 常规路径的一部分。

### 11.2 address_space 把 inode 和缓存页接起来

理解 page cache 时，一个常被忽视的对象是 `address_space`。它大致连接了：

- 某个 inode 对应的缓存页集合
- 页查找、填充、失效、回写等方法
- readahead、writeback 和页回收协作

这意味着文件缓存不是“全局一锅粥”，而是按对象组织的页集合，每个对象通过对应的缓存映射结构管理自己的页视图。

### 11.3 读路径：命中、缺页、预读

一次普通读取大致可能经历：

1. 查 page cache
2. 如果命中，直接返回页内容
3. 如果未命中，触发页填充
4. 文件系统读取底层块/extent
5. 页进入 cache
6. 用户态读到内容

这里最容易被低估的是 **readahead**：如果内核判断你在顺序读，它会提前拉取后续页。因此“第二次更快”有时不是设备变快，而是：

- 已命中 cache
- 甚至第一次读取时就已经预读到更多页

### 11.4 写路径：脏页不是落盘

普通 `write()` 很多时候只是：

- 修改 page cache 页
- 把对应页标成 dirty
- 更新 inode size / mtime 等内核状态
- 让调用者尽快返回

此时数据可能还没有：

- 分配到最终稳定块位点
- 写到设备控制器
- 被 flush 到稳定介质

所以脏页表示“逻辑上新内容已在内存里生效”，不表示“介质上已经持久化”。

### 11.5 dirty throttling 为什么会卡住写线程

如果脏页越来越多，系统不能无限拖延回写，否则会出现：

- 内存被脏页挤满
- 后台回写风暴
- 尾延迟暴涨
- 进程在最糟糕时刻统一阻塞

因此内核会通过 dirty throttling 让部分写入线程主动慢下来。这看起来像“莫名卡顿”，但它本质上是为了避免整个系统在回写失控时一起雪崩。

### 11.6 writeback 不只是“后台慢慢写”

writeback 涉及多个问题：

- 哪些脏页优先回写
- 一次回写多少
- 是周期性刷、压力触发刷，还是显式同步触发刷
- 回写错误如何反馈给后续 `fsync` / `close` / 写线程
- 如何与内存回收协作

高阶理解的关键在于：**写回路径本身也是协议，不是单纯“后台线程帮你做事”**。

### 11.7 writeback error 为什么会延迟暴露

底层写入错误不一定在最早的 `write()` 调用就同步暴露。现实中很常见的是：

- `write()` 看起来成功
- 后台回写失败
- 直到后续 `fsync`、`close` 或另一个同步点才显露错误

这也是为什么一些程序必须认真处理 `fsync` 错误，而不是只看 write 返回值。

### 11.8 page cache 与内存回收会互相牵制

page cache 不是独占内存。它要和这些对象竞争：

- 匿名页
- slab
- dentry / inode cache
- 其他内核缓存

一旦内存紧张：

- clean cache 页更容易被回收
- dirty 页需要先回写或特殊处理
- readahead 命中率可能下降
- 应用观察到的 I/O 延迟会发生结构性变化

所以文件系统性能和内存压力从来不是两个独立问题。

### 11.9 mmap、page cache 与 direct I/O 的边界

普通 buffered I/O 与 `mmap` 通常都和 page cache 有强关系，而 direct I/O 则试图绕过它。于是会出现一些复杂边界：

- 同一文件可能既被 page cache 路径访问，又被 O_DIRECT 路径访问
- `mmap` 修改的脏页与普通 `write()` 脏页最终都要协调写回
- 某些工作负载为了避免双重缓存故意绕开 page cache

这也是为什么不能只问“有没有缓存”，而要问“是哪条 I/O 路径在工作”。

### 11.10 buffer cache 这个词为什么容易误导

旧资料里常会提 buffer cache，但在现代 Linux 语境里，很多文件内容语义重点在 page cache 上。块设备元数据缓冲和文件页缓存并不完全是同一个概念。

如果把二者混为一谈，就很难理解：

- 为什么文件内容和块元数据缓存行为不同
- 为什么 direct I/O 绕过的是哪一层
- 为什么 page cache 命中能掩盖底层设备特征

### 11.11 一个排障框架：cache 路径还是设备路径？

当你怀疑 I/O 有问题时，可以先问：

1. 当前读写命中了 page cache 吗？
2. 是否受到 readahead 影响？
3. 脏页量是否过高？
4. writeback 是否出现拥塞或错误？
5. 当前慢的是系统调用返回、`fsync`、还是后台回写？

如果这五问不分清，几乎所有“磁盘慢/文件系统慢”的结论都可能是错的。

### 11.12 workingset、refault 与“为什么扫一遍冷数据后热点全丢了”

很多人对 page cache 的直觉是“内存够大，热点自然会留住”。现实里更微妙，因为缓存管理不只看当前是否命中，还要持续判断：

- 哪些页只是一次性扫过
- 哪些页是真正反复会再访问的 working set
- 哪些页刚被挤掉，很快又 refault 回来

这带来几个工程现象：

- 一次大目录扫描、全量备份、日志归档，就可能把原本热的数据从 cache 里挤掉
- 系统表面上没有明显设备故障，但业务延迟会在冷扫描后陡增
- “内存总量看起来还可以”并不代表缓存策略就一定对当前负载友好

所以高阶分析里，cache 问题不只是“够不够大”，而是：

- 当前 working set 有多大
- 一次性流量是否在污染 cache
- 热点页是否在短时间内反复被驱逐又读回

### 11.13 memory cgroup、容器限制与为什么同一程序在容器里更容易抖

同一套程序在宿主机上表现正常，进容器后却突然更容易：

- cache 命中下降
- dirty throttling 提前发生
- 回写压力更早显现
- 读写延迟尾巴变长

常见原因不是“容器把文件系统改坏了”，而是：

- 可用内存变小，page cache 回旋空间更少
- 写入与回写压力被限制在更小的资源边界里
- 同一宿主上的其他 workload 也在竞争全局 cache、I/O 和 reclaim 行为

这说明 page cache 结论必须带上环境上下文：

- 裸机和容器不是同一种 cache 世界
- 单进程和多租户也不是同一种回写压力模型
- 同样的 benchmark，在 memory limit 不同的容器里可能测到完全不同的“文件系统性格”

### 11.14 命中、可见性、持久化是三回事

page cache 很容易制造一种错觉：只要本机再次读到新内容，就仿佛一切都完成了。实际上至少要分清三件事：

1. **命中**：本机后续读取是否命中了 cache
2. **可见性**：其他线程、进程、容器，甚至其他客户端何时看到新状态
3. **持久化**：这些状态是否已经跨过 writeback、flush、设备稳定边界

因此下面三句话不能混成一句：

- “我马上又读到了，所以写入成功”
- “别的进程也看见了，所以已经提交”
- “已经 `fsync` 过，所以远端客户端一定也同步看见”

它们分别属于 cache、本地对象可见性、持久化/远端语义三条不同的问题线。

### 11.15 观察 page cache，不要只盯一个数字

高阶排障里，page cache 最怕的不是没有指标，而是只看单个指标就下结论。更可靠的证据链通常至少要同时看：

- `vmstat`：分页、回收、等待和系统整体节奏
- `/proc/meminfo`：`Cached`、`Dirty`、`Writeback` 这些状态量
- `iostat` / `pidstat`：设备层与进程层有没有同步跟着恶化
- `perf`：热点是在页缺失、拷贝、锁、文件系统路径，还是设备等待
- 压力指标：系统是不是已经在内存或 I/O 压力下持续停顿

如果只看：

- 命中率
- 吞吐
- 某一次平均延迟

你很容易把：

- cache 污染
- reclaim 抖动
- writeback 堵塞
- 设备尾延迟

误当成同一种问题。

---

### 11.16 address_space 与 folio：page cache 的内核数据结构

#### struct address_space（`include/linux/fs.h`）

`struct address_space` 是 VFS 层连接 inode 与其缓存页集合的核心对象。每个文件 inode 有一个关联的 `address_space`（通过 `inode->i_mapping`）。

```c
struct address_space {
    struct inode        *host;              /* 关联的 inode（NULL 表示匿名映射）*/
    struct xarray        i_pages;           /* 页缓存 radix tree（现代版本用 XArray）*/
    struct rw_semaphore  invalidate_lock;   /* 保护 invalidate/truncate vs 读路径 */
    gfp_t                gfp_mask;          /* 分配新页使用的 GFP 标志 */
    atomic_t             i_mmap_writable;   /* 可写 mmap 映射数（影响 O_TRUNC 等操作）*/
    struct rb_root_cached i_mmap;           /* 此文件的所有 VMA 区域（mmap 追踪）*/
    unsigned long        nrpages;           /* 当前缓存的总页数 */
    unsigned long        nrexceptional;     /* shadow/swap entry 数（回收后的踪迹）*/
    pgoff_t              writeback_index;   /* writeback 从这个偏移开始扫描脏页 */
    const struct address_space_operations *a_ops; /* 操作表 */
    unsigned long        flags;             /* AS_EIO（writeback 错误）等标志 */
    struct klist         private_list;      /* 文件系统私有链表（buffer_head 等）*/
    errseq_t             wb_err;            /* writeback 错误序列号（供 fsync 检查）*/
    spinlock_t           private_lock;
};
```

关键字段说明：

- `i_pages`（XArray）：用稀疏数组/基数树存储缓存页，key 是文件内的页索引（`page_index = file_offset / PAGE_SIZE`），value 是 `struct folio *`（或旧版的 `struct page *`）。
- `wb_err`：writeback 错误通过此字段传播。调用 `fsync()` 时，VFS 通过比较错误序列号判断自上次 `fsync` 以来是否有新写入错误。
- `nrpages`：当前在 page cache 中的页数（不含 shadow entry），`/proc/meminfo` 的 `Cached` 字段部分来自这里。
- `a_ops`：操作表（见下）。

#### struct address_space_operations（操作表）

```c
struct address_space_operations {
    int   (*writepage)(struct page *, struct writeback_control *);
        /* 把单个脏页写回底层设备（单页回写入口）*/
    int   (*read_folio)(struct file *, struct folio *);
        /* 从底层设备读入一页（page cache miss 时调用）*/
    int   (*writepages)(struct address_space *, struct writeback_control *);
        /* 批量回写脏页（优先于 writepage，ext4/xfs 用此实现多页合并）*/
    bool  (*dirty_folio)(struct address_space *, struct folio *);
        /* 将页标记为脏（ext4: 同时通知 jbd2 记录事务）*/
    void  (*readahead)(struct readahead_control *);
        /* 预读多页（一次性提交 bio 批量读取）*/
    int   (*write_begin)(struct file *, struct address_space *, loff_t,
                         unsigned, struct page **, void **);
        /* write() 路径开始前：分配/锁定目标页，处理 COW/hole 等 */
    int   (*write_end)(struct file *, struct address_space *, loff_t,
                       unsigned, unsigned, struct page *, void *);
        /* write() 路径结束：标记脏页，释放锁 */
    sector_t (*bmap)(struct address_space *, sector_t);
        /* 文件逻辑块号 → 物理块号（swap 和 mmap 用）*/
    void  (*invalidate_folio)(struct folio *, size_t offset, size_t len);
        /* truncate/hole_punch 时使部分或全部页失效 */
    bool  (*release_folio)(struct folio *, gfp_t);
        /* 内存回收时询问文件系统是否可以释放此页 */
    void  (*free_folio)(struct folio *);
        /* 页彻底释放时清理文件系统私有数据 */
    ssize_t (*direct_IO)(struct kiocb *, struct iov_iter *);
        /* O_DIRECT 路径：绕过 page cache 的 I/O 实现 */
    int   (*swap_activate)(struct swap_info_struct *, struct file *,
                           sector_t *);
        /* 允许文件作为 swap 设备 */
};
```

#### struct folio（Linux 5.16+，替代 struct page 的新抽象）

```c
/* folio 是一个"可能跨多个连续 page 的缓存单元" */
/* 对于普通 4K page：folio = 一个 page */
/* 对于 huge page / THP：folio = 多个连续 page */

struct folio {
    /* -- 继承自 struct page -- */
    unsigned long flags;            /* PG_dirty, PG_writeback, PG_locked, PG_uptodate 等 */
    struct list_head lru;           /* LRU 链表节点 */
    struct address_space *mapping;  /* 关联的 address_space */
    pgoff_t index;                  /* 在文件中的页索引（byte_offset / PAGE_SIZE）*/
    atomic_t _refcount;             /* 引用计数 */
    atomic_t _mapcount;             /* 被页表映射的次数（>= 0 表示有进程映射它）*/
    /* ... */

    /* -- folio 特有 -- */
    union {
        struct {
            unsigned long _flags_1;
            unsigned long _head;     /* 标记这是 folio head（非 tail page）*/
        };
    };
};

/* 关键 page flags（PG_xxx 宏）：
   PG_dirty      - 页被修改（需要回写）
   PG_writeback  - 正在被回写（I/O 进行中）
   PG_locked     - 被某个进程锁定（保护并发访问）
   PG_uptodate   - 页内容与磁盘一致（读完成后设置）
   PG_referenced - 被 LRU 标记为近期访问
   PG_active     - 在 active LRU 链表中
   PG_swapbacked - 是匿名页（backed by swap，不是文件页）
   PG_workingset - workingset 追踪标记（用于 refault 检测）
*/
```

**观察缓存页状态**：

```bash
# 查看当前 page cache 总量
cat /proc/meminfo | grep -E "Cached|Dirty|Writeback|Active|Inactive"
# Cached:         2048576 kB    ← 文件 page cache（含部分 tmpfs）
# Dirty:            12288 kB    ← 脏页量（已写但未落盘）
# Writeback:         1024 kB    ← 正在写回的页
# Active(file):   1024000 kB   ← 活跃文件缓存（近期访问）
# Inactive(file):  512000 kB   ← 不活跃文件缓存（等待回收）

# 查看特定文件在 page cache 中的驻留情况
# 工具 1: fincore（需要 util-linux）
fincore /var/log/syslog
# FILE            PAGES  SIZE    PAGES CACHED  CACHED%
# /var/log/syslog  2048  8.0 MiB  1200          58.6%

# 工具 2: pcstat（Go 工具，https://github.com/tobert/pcstat）
# pcstat /path/to/file

# 工具 3: vmtouch
vmtouch /var/lib/postgresql/data/base/
# Files: 1203
# Directories: 23
# Resident Pages: 128000/256000  500M/1000M  50.0%
# Elapsed: 0.42 seconds

# 强制预热（将整个文件读入 page cache）
vmtouch -t /var/lib/postgresql/data/base/pg_wal/
cat /var/lib/postgresql/data/base/pg_wal/* > /dev/null
```

---

### 11.17 readahead 状态机：file_ra_state 与异步预读

内核为每个 `struct file`（open file description）维护一个 `file_ra_state` 结构，追踪该文件的预读状态：

```c
/* include/linux/fs.h */
struct file_ra_state {
    pgoff_t     start;          /* 当前预读窗口起始页索引 */
    unsigned    size;           /* 当前预读窗口大小（页数）*/
    unsigned    async_size;     /* 异步预读阈值（当读到此处时，触发下一次预读）*/
    unsigned    ra_pages;       /* 最大预读页数（受 /sys/block/sdX/queue/read_ahead_kb 限制）*/
    unsigned    mmap_miss;      /* mmap 访问未命中次数（用于判断 mmap 是否是随机访问）*/
    loff_t      prev_pos;       /* 上次读结束位置（用于检测顺序访问模式）*/
};
```

**预读算法逻辑**：

```
首次读取（cold start）：
  → 发起初始预读窗口（通常 4 页）
  → 同步读所需页 + 异步预读后续页

连续读取（顺序检测）：
  → 每次读取确认当前位置 ≈ prev_pos + last_read_size
  → 确认顺序 → 预读窗口指数增长（4 → 8 → 16 → ... → ra_pages 上限）

随机读取检测：
  → 当前读位置与 prev_pos 偏差大 → mmap_miss++ 或 start = 0 重置
  → 禁用预读（避免无效预读浪费内存和 I/O）

触发时机（异步预读）：
  → 读到"当前窗口起始 + (size - async_size)"位置时
  → 异步提交下一个预读窗口的 bio
  → 应用程序的 read() 感受不到这次 I/O（并发进行）
```

**观察预读效果**：

```bash
# 通过 /sys/block 调整预读大小
cat /sys/block/sda/queue/read_ahead_kb   # 默认通常 128KB
echo 2048 > /sys/block/sda/queue/read_ahead_kb  # 调大预读（顺序读受益）

# 通过 posix_fadvise 提示预读策略
python3 -c "
import os, ctypes
libc = ctypes.CDLL('libc.so.6', use_errno=True)
POSIX_FADV_SEQUENTIAL = 2
POSIX_FADV_RANDOM = 1
POSIX_FADV_WILLNEED = 3
POSIX_FADV_DONTNEED = 4

fd = os.open('/var/log/syslog', os.O_RDONLY)
# 告诉内核即将顺序读取（激进预读）
libc.posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)
# 读取后告诉内核不再需要缓存
# libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
os.close(fd)
"

# 观察实际 I/O 与预读的关系（iostat + vmstat 联动）
# Terminal 1: 监控 I/O
iostat -x 1 /dev/sda &

# Terminal 2: 测试顺序读 vs 随机读
dd if=/var/log/large.log of=/dev/null bs=4K count=1000 iflag=direct  # 顺序读，观察 r/s
python3 -c "
import random, os
f = open('/var/log/large.log', 'rb')
size = os.path.getsize('/var/log/large.log')
for _ in range(1000):
    f.seek(random.randint(0, size - 4096))
    f.read(4096)   # 随机读，r/s 高但预读命中率低
"
```

---

### 11.18 writeback 基础设施：bdi_writeback 与脏页控制参数

#### writeback 架构

```
脏页产生：write() → page cache 中对应页设置 PG_dirty → inode 加入 writeback_index

writeback 触发来源（三种）：
  1. 周期性刷（flusher 线程每 5 秒扫描一次）
  2. 压力触发（脏页比例超过 dirty_background_ratio）
  3. 显式同步（fsync/sync/syncfs 调用）

关键内核线程：
  每个 backing device（块设备）对应一个 bdi_writeback 结构
  每个 bdi_writeback 有一个 wb 工作线程（kworker/uXX:X-flush-MAJ:MIN）
```

```bash
# 观察 writeback 线程
ps aux | grep "flush-"
# root  234  0.0  0.0  0  0 ?  S  00:00  0:02 [kworker/u4:1-flush-8:1]
# root  235  0.0  0.0  0  0 ?  S  00:00  0:01 [kworker/u4:2-flush-8:16]
# 命名格式：flush-MAJ:MIN（对应设备 major:minor 号）
```

#### 关键 /proc/sys/vm 参数

```bash
# 查看所有 vm 相关参数
ls /proc/sys/vm/ | sort | head -20
sysctl -a | grep vm.dirty

# 参数 1: vm.dirty_background_ratio（默认 10）
# 脏页占总内存的百分比超过此值时，后台刷写开始
cat /proc/sys/vm/dirty_background_ratio   # 默认 10（即 10%）
# → 总内存 8GB，脏页超过 800MB 时后台刷写启动

# 参数 2: vm.dirty_ratio（默认 20）
# 脏页占总内存超过此值时，写入进程被迫阻塞（dirty throttling）
cat /proc/sys/vm/dirty_ratio   # 默认 20（即 20%）
# → 总内存 8GB，脏页超过 1.6GB 时写入进程开始被阻塞

# 参数 3: vm.dirty_expire_centisecs（默认 3000 = 30秒）
# 脏页超过多少厘秒后，必须被回写（即使未触发 ratio）
cat /proc/sys/vm/dirty_expire_centisecs   # 3000 = 30 秒

# 参数 4: vm.dirty_writeback_centisecs（默认 500 = 5秒）
# 刷写线程的唤醒周期
cat /proc/sys/vm/dirty_writeback_centisecs  # 500 = 每 5 秒

# 参数 5: vm.vfs_cache_pressure（默认 100）
# 内存回收时 page cache vs 匿名页的压力权重
# 越大 → 越积极回收文件缓存；越小 → 更多保留 page cache
cat /proc/sys/vm/vfs_cache_pressure  # 100

# 参数 6: vm.min_free_kbytes
# 保持最小空闲内存量（防止分配时卡死）
cat /proc/sys/vm/min_free_kbytes

# 数据库场景常用调优（减少延迟抖动）：
sysctl -w vm.dirty_background_ratio=5   # 更早开始后台回写
sysctl -w vm.dirty_ratio=10             # 更早触发写入阻塞（压力曲线更平稳）
sysctl -w vm.dirty_expire_centisecs=1000  # 缩短脏页存活时间
sysctl -w vm.vfs_cache_pressure=50     # 保留更多 page cache
```

#### cgroup 内存限制对 writeback 的影响

```bash
# 查看容器的内存 cgroup 设置
cat /sys/fs/cgroup/memory/docker/<container_id>/memory.limit_in_bytes
cat /sys/fs/cgroup/memory/docker/<container_id>/memory.stat

# memory.stat 中与 page cache 相关的字段：
# cache: 4096000        ← 此 cgroup 的 page cache 用量（字节）
# rss: 102400000        ← 匿名页 + swap
# mapped_file: 1048576  ← 被 mmap 映射的文件页
# pgpgin: 12345         ← 从磁盘读入的页数（自启动起累积）
# pgpgout: 23456        ← 写出到磁盘的页数
# pgfault: 345678       ← page fault 次数（含 minor fault）
# pgmajfault: 123       ← major page fault（必须从磁盘读）

# 观察 cgroup 的内存压力
cat /sys/fs/cgroup/memory/docker/<container_id>/memory.pressure_level
# 或使用 PSI（Pressure Stall Information，Linux 4.20+）
cat /proc/pressure/memory
# some avg10=0.00 avg60=0.00 avg300=0.00 total=0        ← 没有内存压力
# full avg10=1.23 avg60=0.45 avg300=0.12 total=123456   ← 有压力（1.23% of time stalled）
```

---

### 11.19 实践工具：观察 page cache 状态与性能

**基础指标读取**：

```bash
# /proc/meminfo 完整解读
grep -E "Mem|Cache|Dirty|Writeback|Active|Inactive|Mapped|Shmem|Slab|Buff" /proc/meminfo
# MemTotal:       16384000 kB   ← 总物理内存
# MemFree:         2048000 kB   ← 完全空闲
# Buffers:          131072 kB   ← 块设备元数据缓存（非文件页）
# Cached:          8192000 kB   ← 文件 page cache（不含 Buffers）
# SwapCached:            0 kB
# Active:          6144000 kB   ← 活跃 LRU（近期访问，不易被回收）
# Inactive:        3200000 kB   ← 不活跃 LRU（候选回收）
# Active(anon):    2048000 kB   ← 活跃匿名页
# Inactive(anon):   256000 kB
# Active(file):    4096000 kB   ← 活跃文件 page cache
# Inactive(file):  2944000 kB   ← 不活跃文件 page cache
# Dirty:             12288 kB   ← 脏页（已修改，待回写）
# Writeback:          1024 kB   ← 正在回写中的页
# AnonPages:       1843200 kB   ← 非文件页（堆、栈、mmap(MAP_ANON)）
# Mapped:           512000 kB   ← 被进程 mmap 映射的页（是 Cached 的子集）
# Shmem:            102400 kB   ← 共享内存 / tmpfs 用量

# vmstat：系统级页操作速率
vmstat -S M 1 10   # 每秒采样，MB 单位
# r  b   swpd  free  buff  cache  si  so  bi   bo   in   cs  us  sy  id  wa
# 2  0   0     2048  128   8192   0   0   256  128  1024 2048 20  5   74  1
# ↑                                               bi=块读入速率, bo=块写出速率

# 观察脏页/回写的实时变化
watch -n 1 'grep -E "Dirty|Writeback" /proc/meminfo'
```

**page cache 命中分析（eBPF 工具）**：

```bash
# cachestat（来自 BCC/bpftrace）：实时显示 cache 命中/失误
cachestat 1
# HITS   MISSES  DIRTIES  HITRATIO
# 2048   128     64       94.12%    ← 94% 命中率
# 4096   0       128      100.0%
# 1024   512     256      66.67%    ← 命中率下降，可能有冷扫描

# cachestat 原理（eBPF probe 挂载的内核函数）：
# add_to_page_cache_lru → miss（page 从磁盘读入，未命中）
# mark_page_accessed    → hit（page 已在 cache，被访问）
# account_page_dirtied  → dirty（page 被修改）

# cachetop：类似 top 的按进程显示
cachetop 5   # 每 5 秒刷新

# perf：分析 page fault 热点
perf stat -e cache-references,cache-misses,page-faults,major-faults ./workload
# page-faults:        12345  ← minor fault（page 在 cache，只需建页表）
# major-faults:         123  ← major fault（page 不在 cache，需磁盘 I/O）
```

**writeback 延迟分析**：

```bash
# 观察 writeback 事件（通过 ftrace）
echo 1 > /sys/kernel/debug/tracing/events/writeback/writeback_dirty_page/enable
echo 1 > /sys/kernel/debug/tracing/events/writeback/writeback_write_inode/enable
cat /sys/kernel/debug/tracing/trace | head -20
# 每次页变脏和 inode 被回写时的事件

# 实际测量写入延迟与 writeback 的关系
python3 -c "
import os, time, statistics
delays = []
f = open('/tmp/writeback_test', 'wb')
for _ in range(100):
    t0 = time.monotonic()
    f.write(b'x' * 1024 * 1024)  # 1MB write
    t1 = time.monotonic()
    delays.append(t1 - t0)
f.close()
os.unlink('/tmp/writeback_test')
print(f'avg={statistics.mean(delays)*1000:.2f}ms max={max(delays)*1000:.2f}ms')
# avg=0.12ms max=15.34ms  ← max 突刺通常是 dirty throttling 生效
"

# 显式控制回写（不影响应用）
sync  # 同步所有文件系统
echo 3 > /proc/sys/vm/drop_caches  # 回收所有可回收缓存（注意：影响性能）
```

---

## 本章小结

| 主题 | 结论 |
|------|------|
| page cache | 是普通文件 I/O 的核心执行面 |
| address_space | 把对象元数据与缓存页组织连接起来 |
| readahead | 让顺序读表现显著不同于随机读 |
| dirty page | 只代表内存中已修改，不代表已持久化 |
| writeback | 是独立的协议链，不是简单后台线程 |
| dirty throttling | 防止脏页失控和系统整体抖动 |
| 回写错误 | 可能延迟到 `fsync` 等同步点才暴露 |
| workingset / refault | 能解释“为什么扫一遍冷数据后热点突然失效” |
| 容器内存边界 | 会显著改变 cache、reclaim 和回写体验 |
| 命中 / 可见 / 持久 | 是三条不同的问题线，不能混成一句话 |

---

## 练习题

1. 为什么 page cache 不应被理解成“可有可无的优化层”？
2. `address_space` 在文件缓存路径里扮演什么角色？
3. 为什么 `write()` 成功和 writeback 成功不是一回事？
4. dirty throttling 为什么会让写线程看起来“莫名卡住”？
5. 如何判断你测到的是 page cache 性能还是设备性能？
6. 为什么一次全量扫描可能会让原本的热点业务突然变慢？
7. 同一个 workload 在内存受限容器里为什么更容易出现 cache 抖动和尾延迟？
8. 为什么“本机再次读到了新内容”不能自动推出“它已经可靠持久化”？
