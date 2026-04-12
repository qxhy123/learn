# 第13章：空间管理、碎片与 TRIM

> 高阶空间管理不是“看还剩多少 GB”，而是同时理解 inode、extent、预分配、空洞、保留空间、配额、discard 和底层设备回收逻辑。

## 学习目标

完成本章后，你将能够：

1. 区分字节空间、inode 空间、逻辑大小、物理占用这几种不同“空间”
2. 解释内部碎片、外部碎片、空闲空间碎片、extent 碎裂的差异
3. 理解预分配、稀疏文件、延迟分配、reflink/dedupe 对空间观察的影响
4. 认识 quota、reserved blocks、discard / fstrim、SSD GC 的边界
5. 用空间形态而不是单一容量数值来分析 ENOSPC 类问题
6. 区分“磁盘扩了”“分区扩了”“卷扩了”“文件系统真的长大了”这四层差别

---

## 正文内容

### 13.1 先区分“哪种空间不够了”

`No space left on device` 并不只意味着“字节数用完”。可能不够的是：

- 数据块空间
- inode 数量
- 某个项目/用户配额
- 连续 extent 空间质量
- 预留给 root 或系统内部的可用余量

所以真正的第一问不是“磁盘是不是满了”，而是“哪个资源池触顶了”。

### 13.2 逻辑大小、物理占用和文件系统视角为什么会打架

一个文件可能同时有：

- 逻辑大小：应用看到的文件长度
- 物理占用：真实分配的块数
- 文件系统实例层可用空间：`df` 看到的剩余块

这解释了为什么：

- `ls -lh` 很大
- `du -h` 却不大
- `df -h` 又显示整盘快满了

这三种信息关注的层次不同，不能混着下结论。

### 13.3 稀疏文件不是“特殊格式”，而是空间协议

稀疏文件（sparse file）表示逻辑上存在大范围内容，但其中某些区间没有真实分配物理块。其结果是：

- 文件逻辑长度可以很大
- 真正占用磁盘块却很小
- 中间“洞”在读取时表现得像零页或未定义填充值语义

稀疏文件广泛出现在：

- 虚拟机镜像
- 数据库文件
- 预分配场景
- 快照/模板复制

所以空间问题经常不是“文件太大”，而是“这个大文件到底有没有真实占块”。

### 13.4 碎片至少有三种

“碎片”不是一个词能讲完。至少要区分：

- **内部碎片**：块内没用完的尾部空间
- **外部碎片**：空闲空间存在，但被打散得不利于连续分配
- **extent 碎裂**：同一文件被分成很多小区间，映射树复杂、顺序性变差

如果不区分这些碎片类型，就很难解释为什么：

- 小文件海量创建和删除会让目录/元数据很慢
- 大文件顺序写性能会逐渐退化
- 同样剩余很多空间，却越来越难维持连续布局

### 13.5 预分配与延迟分配不是同一件事

- **预分配**：提前保留空间，减少未来失败或碎片
- **延迟分配**：先不急着决定最终物理位置，等写入模式更清楚时再分配

两者都与局部性有关，但方向不同：

- 预分配更像提前占位
- 延迟分配更像延后决策

数据库、日志系统和大文件工作负载经常会显式利用这些机制降低运行期抖动。

### 13.6 quota 和 reserved blocks 为什么会让“明明有空间却不能写”

文件系统并不会把所有剩余块都对所有用户开放。典型约束包括：

- root reserved blocks
- 用户/组/项目 quota
- 某些元数据增长保留空间

这意味着空间可用性不仅是“有没有”，还是“对谁有没有”。在多租户、容器平台和共享主机上，这个区别非常关键。

### 13.7 reflink、dedupe 与快照会制造“空间幻觉”

在支持 CoW / reflink / dedupe 的文件系统里，同一份物理数据可能被多个逻辑对象共享。于是：

- 看起来复制了很多份
- 真实占用并没有线性增长
- 修改其中一份时又会触发新的写放大或分叉

所以“我删掉一个文件为什么空间没立刻回来”未必只和 open file 有关，也可能和快照、reflink、后台引用计数相关。

### 13.8 SSD 上的 discard / TRIM 解决的是什么问题

对 SSD 来说，文件删除并不等于控制器立刻知道这些逻辑块可以更高效地回收。discard / TRIM 的作用是把“这些逻辑区域已无有效内容”的信息告诉设备层。

但要分清：

- 它提升的是设备内部回收和写放大控制
- 它不是文件系统层的“立即回收按钮”
- 在线 discard、定期 `fstrim`、设备固件行为各有性能取舍

### 13.9 ENOSPC 的工程排查顺序

遇到 ENOSPC，可以按下面顺序问：

1. 是 `df -h` 空间不够，还是 `df -i` inode 不够？
2. 是某个 quota 触发，还是 root reserved blocks 在生效？
3. 是 deleted-but-open 占着空间，还是 snapshot/reflink 还在引用？
4. 是逻辑大小大，还是物理占用大？
5. 是设备空间问题，还是布局质量/碎片问题？

只有把空间形态拆开，ENOSPC 才不再像“玄学报错”。

### 13.10 动态扩容时，最容易错的是搞错层次

很多“云盘已经扩容成功，但系统里还是老容量”的问题，本质上不是命令不会用，而是层次没分清。至少要区分：

| 你改了什么 | 还没改什么 |
|------------|------------|
| 底层磁盘/LUN 变大了 | 分区表、PV/LV、文件系统可能还没变 |
| 分区变大了 | 文件系统可能还没使用新增空间 |
| LVM 逻辑卷变大了 | 文件系统仍可能停留在旧大小 |
| 文件系统 grow 完了 | 配额、应用配置、监控阈值未必同步更新 |

所以“扩容”不是一个动作，而是一条链：

```text
设备容量 -> 分区 / PV -> VG / LV -> 文件系统 -> 挂载点视角
```

任何一层没跟上，`df -h` 就可能还是旧结果。

### 13.11 在线扩容最常见的两条路径

实践里最常见的是两种：

1. **分区直接承载文件系统**
   - 扩大底层磁盘
   - 调整分区边界
   - 让 ext4 / xfs 看到更大底层空间
   - 执行 `resize2fs` 或 `xfs_growfs`
2. **LVM 承载文件系统**
   - 扩大底层磁盘或新增 PV
   - `pvresize` 或把新盘加入 VG
   - `lvextend`
   - 再执行文件系统层增长

高阶判断不是背命令，而是知道：

- ext4 常用 `resize2fs`，目标是设备或 LV
- xfs 常用 `xfs_growfs`，目标往往是挂载点
- `lvextend -r` 这种一条命令打包做了不止一层动作

如果你想看完整操作链，可以直接读[附录 A5：分区、mkfs、挂载与动态扩容实操](../appendix/practical-mount-partition-and-resize.md)。

### 13.12 缩容为什么通常比扩容更危险

很多人第一次理解扩容后，会自然地问：“那缩回去是不是同理？”通常不是。

原因包括：

- 文件系统必须先确认已有数据能安全腾挪到更小空间内
- 某些文件系统支持在线增长，但不支持在线缩小
- 一旦把块层或分区层缩得比文件系统真实占用还小，损坏风险极高

所以在工程上常见的经验是：

- 扩容可以设计成标准化 runbook
- 缩容往往需要更保守的离线窗口、备份和恢复预案
- 对 xfs 一类系统，增长常见，缩小则应默认视为高风险操作

这也是为什么“动态扩容”常被教程写进常规实践，而“动态缩容”通常只在高级运维文档里谨慎出现。

---

### 13.13 ext4 块分配器内部：mballoc 与 `struct ext4_group_info`

理解 ext4 的空间管理，需要深入 multi-block allocator（mballoc）的数据结构：

```c
/* fs/ext4/mballoc.h */

/* 每个块组的运行时状态（驻留内存，不在磁盘上）*/
struct ext4_group_info {
    unsigned long   bb_state;           /* 状态标志（EXT4_GROUP_INFO_NEED_INIT_BIT 等）*/
    struct rb_root  bb_free_root;       /* 空闲 extent 的红黑树（按大小和位置索引）*/
    ext4_grpblk_t   bb_first_free;      /* 首个空闲块号（快速找起点）*/
    ext4_grpblk_t   bb_free;           /* 组内空闲块总数 */
    ext4_grpblk_t   bb_fragments;      /* 空闲片段数（碎片化程度指标）*/
    ext4_grpblk_t   bb_largest_free_order;  /* 最大连续空闲块的幂次（优化快速查找）*/
    ext4_group_t    bb_group;          /* 块组编号 */
    struct list_head bb_prealloc_list; /* 预分配区间链表 */
    struct rw_semaphore alloc_sem;     /* 分配锁 */
    ext4_grpblk_t   bb_counters[];    /* 按幂次统计的空闲 extent 数组
                                         bb_counters[k] = 长度 >= 2^k 块的空闲区间数
                                         用于快速判断能否满足请求 */
};

/* 分配请求描述 */
struct ext4_allocation_request {
    struct inode    *inode;         /* 分配给哪个 inode */
    unsigned int    len;            /* 请求分配的块数 */
    ext4_lblk_t     logical;        /* 文件内逻辑块号 */
    ext4_lblk_t     lleft;          /* 左邻居逻辑块（局部性提示）*/
    ext4_lblk_t     lright;         /* 右邻居逻辑块（局部性提示）*/
    ext4_fsblk_t    goal;           /* 理想的物理块位置（块组 + 偏移）*/
    ext4_fsblk_t    pleft;          /* 左邻居物理块（辅助局部性判断）*/
    ext4_fsblk_t    pright;         /* 右邻居物理块 */
    unsigned int    flags;          /* EXT4_MB_HINT_MERGE / EXT4_MB_HINT_RESERVED 等 */
};
```

**mballoc 分配策略**（三级查找）：

```
mballoc 分配流程（ext4_mb_new_blocks → ext4_mb_regular_allocator）：

第 1 级：局部化分配（locality）
  → 优先在 goal 块所在块组内查找
  → 遍历 bb_free_root 红黑树，找符合 len 要求的空闲 extent
  → 若找到且质量好（连续性高），直接分配 → 最快路径

第 2 级：块组间搜索
  → 在目标块组失败时，按轮询或策略顺序搜索其他块组
  → 参考 bb_largest_free_order 快速跳过无法满足的块组
  → 每个块组用 ext4_mb_find_by_goal() 或 ext4_mb_simple_scan_group()

第 3 级：碎片整合搜索
  → 最后手段：允许分配比请求更小的 extent（ext4_mb_scan_group）
  → 结果可能是若干不连续的小 extent，形成更多的 extent tree 节点
  → 这是性能退化和 extent 碎裂的根因

预分配机制（减少频繁小分配的锁竞争）：
  inode 预分配（per-inode prealloc）：
    → ext4_mb_use_inode_pa() → 先从 inode 的 bb_prealloc_list 中取
    → 若预分配有剩余，直接使用，无需查块组位图
    → 文件关闭时归还未用的预分配空间

  局部组预分配（group prealloc）：
    → ext4_mb_use_group_pa() → 从块组级预分配池中取
    → 用于同一进程密集创建多个小文件的场景
```

**观察 ext4 分配器状态**：

```bash
# 查看每个块组的碎片化程度（需 debugfs）
debugfs -R "stats" /dev/sda1 2>/dev/null | grep -E "Block|Group|Fragment"
# Block count: 52428800
# Fragment count: 12345678    ← 碎片总数（越高说明外部碎片越严重）

# 查看具体块组的空闲信息
debugfs -R "group_info 0" /dev/sda1 2>/dev/null
# Group 0: block bitmap at 1025, inode bitmap at 1041, inode table at 1057
#   32637 free blocks, 1 free inodes, 2 used directories
#   Free blocks: 1058-32751, 32753-65535

# 用 e2freefrag 分析空闲空间碎片化程度
e2freefrag /dev/sda1
# Device: /dev/sda1
# Blocksize: 4096 bytes
# Total blocks: 52428800
# Free blocks: 23456789 (44.7%)
#
# Min. free extent: 1 KB
# Max. free extent: 512 MB      ← 最大连续空闲区间
# Avg. free extent: 2048 KB
# Num. free extents: 12345       ← 碎片数
#
# HISTOGRAM OF FREE EXTENT SIZES:
# Extent Size Range    Free extents   Free Blocks    Pct
# 4K...    8K-  :         3456          3456          0.01%
# 8K...   16K-  :         2345          4690          0.02%
# ...
# 512M...  1G-  :            8        131072         0.56%
# 1G...    2G-  :            3        786432         3.35%  ← 大连续区间

# 查看文件的 extent 分布
filefrag -v /var/lib/mysql/ibdata1
# File size of /var/lib/mysql/ibdata1 is 1073741824 (262144 blocks of 4096 bytes)
# ext:     logical_offset:        physical_offset: length:   expected: flags:
#   0:        0..   32767:    2097152..   2129919:  32768:
#   1:    32768..   65535:    2162688..   2195455:  32768:    2129920:
#   2:    65536..  131071:    3145728..   3211263:  65536:    2195456:
# ↑ 3 个 extent（此文件碎片化程度不严重）
# 如果 extent 数达到数百个，说明严重碎片化

# 量化碎片化程度
filefrag /var/log/syslog 2>&1 | grep extents
# /var/log/syslog: 23 extents found    ← 23 个不连续区间，碎片明显
```

---

### 13.14 `fiemap` 与稀疏文件洞的内核实现

`fiemap` ioctl 是比 `FIBMAP` 更现代的逻辑到物理块映射查询接口：

```c
/* include/uapi/linux/fiemap.h */
struct fiemap {
    __u64   fm_start;           /* 查询的文件内起始偏移（字节）*/
    __u64   fm_length;          /* 查询范围长度 */
    __u32   fm_flags;           /* 查询标志 */
    __u32   fm_mapped_extents;  /* 返回的 extent 数量 */
    __u32   fm_extent_count;    /* 调用者分配的 fm_extents 数组容量 */
    __u32   fm_reserved;
    struct fiemap_extent fm_extents[];  /* extent 数组 */
};

struct fiemap_extent {
    __u64   fe_logical;         /* extent 的文件内逻辑起始偏移（字节）*/
    __u64   fe_physical;        /* extent 的物理起始偏移（字节，磁盘地址）*/
    __u64   fe_length;          /* extent 长度（字节）*/
    __u32   fe_flags;           /* 标志（见下）*/
    __u32   fe_reserved[3];
};

/* fe_flags 标志 */
#define FIEMAP_EXTENT_LAST          0x00000001  /* 最后一个 extent */
#define FIEMAP_EXTENT_UNKNOWN       0x00000002  /* 物理位置不确定（延迟分配）*/
#define FIEMAP_EXTENT_DELALLOC      0x00000004  /* 延迟分配：逻辑已分配，物理未分配 */
#define FIEMAP_EXTENT_ENCODED       0x00000008  /* 压缩或加密（不能直接 IO）*/
#define FIEMAP_EXTENT_DATA_ENCRYPTED 0x00000080 /* 数据已加密 */
#define FIEMAP_EXTENT_NOT_ALIGNED   0x00000100  /* 不对齐（对象存储等）*/
#define FIEMAP_EXTENT_DATA_INLINE   0x00000200  /* 数据内联在元数据中 */
#define FIEMAP_EXTENT_DATA_TAIL     0x00000400  /* 尾部打包数据 */
#define FIEMAP_EXTENT_UNWRITTEN     0x00000800  /* 已分配但未写（fallocate 产生）*/
#define FIEMAP_EXTENT_MERGED        0x00001000  /* 多个 extent 合并显示 */
#define FIEMAP_EXTENT_SHARED        0x00002000  /* 与其他文件共享（reflink/dedupe）*/

/* fm_flags */
#define FIEMAP_FLAG_SYNC   0x00000001  /* 先 fsync 再查询 */
#define FIEMAP_FLAG_XATTR  0x00000002  /* 查询 xattr 的 extent 布局 */
#define FIEMAP_FLAG_CACHE  0x00000004  /* 只返回已缓存的 extent（不触发 I/O）*/
```

**Python 程序直接调用 `fiemap` ioctl**：

```python
import fcntl, struct, os, ctypes

def fiemap(fd, start=0, length=None, max_extents=1024):
    """查询文件的 extent 映射"""
    if length is None:
        length = os.fstat(fd).st_size - start

    # struct fiemap + struct fiemap_extent * max_extents
    FIEMAP_SIZE = 32  # sizeof(struct fiemap)
    EXTENT_SIZE = 56  # sizeof(struct fiemap_extent)
    buf_size = FIEMAP_SIZE + EXTENT_SIZE * max_extents

    buf = bytearray(buf_size)

    # 填写 fiemap 头部
    struct.pack_into('QQIIII', buf, 0,
        start,           # fm_start
        length,          # fm_length
        0,               # fm_flags
        0,               # fm_mapped_extents（输出）
        max_extents,     # fm_extent_count
        0)               # fm_reserved

    FS_IOC_FIEMAP = 0xC020660B  # 从 linux/fs.h 计算或 python-fiemap 包
    fcntl.ioctl(fd, FS_IOC_FIEMAP, buf)

    # 解析输出
    _, _, _, n_extents, _, _ = struct.unpack_from('QQIIII', buf, 0)

    extents = []
    for i in range(n_extents):
        offset = FIEMAP_SIZE + i * EXTENT_SIZE
        fe_logical, fe_physical, fe_length, fe_flags = \
            struct.unpack_from('QQQII', buf, offset)[:4]
        extents.append({
            'logical': fe_logical,
            'physical': fe_physical,
            'length': fe_length,
            'flags': fe_flags,
            'unwritten': bool(fe_flags & 0x800),
            'delalloc': bool(fe_flags & 0x004),
            'shared': bool(fe_flags & 0x2000),
        })
    return extents

# 使用示例
fd = os.open('/var/lib/postgresql/data/base/16384/1259', os.O_RDONLY)
extents = fiemap(fd)
os.close(fd)

for ext in extents:
    physical_mb = ext['physical'] / (1024**2)
    length_kb = ext['length'] / 1024
    flags_str = ' '.join(k for k, v in ext.items() if k not in ('logical','physical','length','flags') and v)
    print(f"logical={ext['logical']:10d}  physical={physical_mb:8.2f}MB  "
          f"len={length_kb:6.1f}KB  {flags_str}")
```

**稀疏文件与"洞"的内核表示**：

```bash
# 创建稀疏文件（10GB 但只占几 KB）
truncate -s 10G /tmp/sparse_test
ls -lh /tmp/sparse_test    # 显示 10G
du -sh /tmp/sparse_test    # 显示 ~0（几乎没有物理块）

# 用 SEEK_DATA / SEEK_HOLE 定位数据区和洞（lseek 扩展）
python3 -c "
import os

fd = os.open('/tmp/sparse_test', os.O_RDWR)
# 在偏移 1G 处写入数据（创建一个数据岛）
os.lseek(fd, 1024**3, os.SEEK_SET)
os.write(fd, b'data island in sparse file')

# 在偏移 5G 处写入数据
os.lseek(fd, 5 * 1024**3, os.SEEK_SET)
os.write(fd, b'another island')

SEEK_DATA = 3   # lseek(fd, offset, SEEK_DATA) → 下一个数据区起点
SEEK_HOLE = 4   # lseek(fd, offset, SEEK_HOLE) → 下一个洞起点

# 定位所有数据区和洞
pos = 0
size = 10 * 1024**3
while pos < size:
    try:
        data_start = os.lseek(fd, pos, SEEK_DATA)
        hole_start = os.lseek(fd, data_start, SEEK_HOLE)
        print(f'DATA: [{data_start:12d}, {hole_start:12d}) = {(hole_start-data_start)//1024}KB')
        pos = hole_start
    except OSError:
        break
os.close(fd)
"
# 输出：
# DATA: [ 1073741824,  1073741850) = 0KB   ← 1G 处的 26 字节
# DATA: [ 5368709120,  5368709134) = 0KB   ← 5G 处的 14 字节
```

---

### 13.15 discard / TRIM 内核机制：`blkdev_issue_discard` 与 fstrim

**内核的 discard 请求路径**：

```
discard 的生命周期（以 ext4 delete 为例）：

1. 文件删除：ext4_free_blocks()
   → 把块标记为空闲（block bitmap 清零）
   → 若挂载了 -o discard（在线 discard）：
       ext4_discard_preallocations()
         → sb_issue_discard()
             → blkdev_issue_discard(bdev, sector, nr_sects, GFP_NOFS)
                   → 创建一个 REQ_OP_DISCARD 类型的 bio
                   → submit_bio() → 块层 → 设备驱动
                   → 对 NVMe：转换为 NVMe Dataset Management（DSM）命令
                   → 对 SATA SSD：转换为 ATA TRIM 命令
                   → 设备固件记录这些 LBA 可被 GC 回收

2. 定期 fstrim（推荐方式）：
   fstrim /mountpoint
     → ioctl(fd, FITRIM, &range)
         → file系统 .trim_fs() 方法
               ext4_trim_fs()
                 → 遍历所有块组
                 → 对每个块组的空闲区间调用 ext4_trim_extent()
                     → sb_issue_discard(sb, start, count)
                         → 批量提交 discard bio（比在线 discard 更高效）

在线 discard vs 定期 fstrim 的对比：

| 特性 | 在线 discard (-o discard) | 定期 fstrim |
|------|--------------------------|-------------|
| 触发时机 | 每次删除立刻触发 | 手动或定时（systemd-fstrim.timer）|
| 粒度 | 细粒度，每次小块 | 批量，合并后下发 |
| 对前台延迟的影响 | 删除操作延迟增加（等待设备响应）| 几乎不影响前台 |
| 写放大影响 | 持续产生小 discard 命令 | 集中产生，设备更容易优化 |
| 适合场景 | 对空间回收实时性要求高 | 大多数生产环境推荐 |
| 企业 SSD 建议 | 通常不推荐（增加磨损） | 每周一次 fstrim |
```

**观察 discard/TRIM 行为**：

```bash
# 检查文件系统是否支持 discard（查看挂载参数）
mount | grep discard
# /dev/nvme0n1p1 on / type ext4 (rw,relatime,discard)

# 检查设备是否支持 TRIM
cat /sys/block/nvme0n1/queue/discard_granularity   # 非零说明支持
cat /sys/block/nvme0n1/queue/discard_max_bytes      # 单次最大 discard 大小
cat /sys/block/sda/queue/discard_granularity        # 0 = 不支持（旧 HDD）

# 手动执行 fstrim 并观察效果
df -h /                     # 记录前空间
fstrim -v /                 # -v 打印已释放的字节数
# /: 5.5 GiB (5902737408 bytes) trimmed
df -h /                     # 对比：文件系统空间不变（fstrim 改变的是设备内部）

# 设置 systemd 定时 fstrim（每周一次）
systemctl enable --now fstrim.timer
systemctl list-timers fstrim.timer
# NEXT                          LEFT       LAST  PASSED  UNIT           ACTIVATES
# Mon 2026-04-13 00:00:00 CST   11h left   n/a   n/a     fstrim.timer   fstrim.service

# 用 blktrace 观察 discard 请求
blktrace -d /dev/nvme0n1 -o - | blkparse -i - -f "%T %a %s+%n\n" | grep D
# 0.000000000  D  2097152+8192    ← discard 操作：从 LBA 2097152 开始，长度 8192 块
# 0.001234567  D  3145728+16384

# 观察 NVMe 的 TRIM 命令（nvme-cli）
nvme id-ctrl /dev/nvme0n1 | grep -E "oncs|dsm"
# oncs      : 0x5f  ← bit 2 = 支持 Dataset Management（TRIM）
nvme dsm /dev/nvme0n1 --namespace-id=1 --ad  # 手动发送 DSM（了解即可，勿随意执行）

# SSD 的写放大因子（WAF）观察
nvme smart-log /dev/nvme0n1 | grep -E "host_write|nand_write"
# 计算 WAF = nand_write / host_write（理想值接近 1，越大越坏）
```

---

### 13.16 空间碎片化的量化模型与在线整理

**碎片化的工程量化**：

```bash
# 方法 1: filefrag 计算碎片率
# 对整个文件系统的重要文件做检查
for f in $(find /var/lib/mysql -type f -size +1M); do
    extents=$(filefrag "$f" 2>&1 | awk '/extents found/{print $1}')
    size=$(stat -c%s "$f")
    ideal=$(( (size + 4095) / 4096 ))  # 理想情况下的 extent 数（全连续 = 1）
    echo "$extents extents  ideal=1  file=$f"
done

# 方法 2: e2freefrag 分析空闲空间碎片
e2freefrag /dev/sda1
# 关注 "Avg. free extent" 和 "Num. free extents"
# 若 Avg < 1MB 且 Num 很大 → 严重碎片化

# 方法 3: 通过 inode 的 extent 数量统计（debugfs）
debugfs -R "stat <inode_number>" /dev/sda1
# Extents: 1      ← 好，连续存储
# Extents: 2048   ← 坏，严重碎片化
```

**ext4 在线碎片整理（e4defrag）**：

```bash
# e4defrag 原理：对每个文件，使用 EXT4_IOC_MOVE_EXT ioctl
# 1. 分配一个临时的、连续的 donor extent
# 2. 把原文件数据移到 donor extent
# 3. 原文件 extent 变为连续（原地替换页缓存中的映射）

# 检查碎片化状态（不修改）
e4defrag -c /var/lib/mysql/ibdata1
# <Fragmentation score (before defragmentation)>
# Score = 0 [0-30 low, 31-55 moderate, 56-100 high]
# Result = OK (score = 2)

# 对单个文件整理
e4defrag /var/lib/mysql/ibdata1

# 对整个目录整理（可能耗时较长）
e4defrag /var/lib/mysql/

# EXT4_IOC_MOVE_EXT ioctl（内核接口）
# 用于实现在线整理：把文件的某个 extent 移到物理位置更好的地方
# 应用程序可以调用此 ioctl 实现自定义的碎片整理策略

# 观察 e4defrag 的效果（比较整理前后的 filefrag 输出）
filefrag -v /var/lib/mysql/ibdata1 | tail -3  # 整理前：Extents: 1234
e4defrag /var/lib/mysql/ibdata1
filefrag -v /var/lib/mysql/ibdata1 | tail -3  # 整理后：Extents: 3
```

**XFS 的空间管理特性**：

```bash
# XFS 用 B+树管理空闲空间（两棵：按起始块号索引 + 按大小索引）
# 查看 XFS 空间使用情况
xfs_info /mountpoint
# meta-data=/dev/sda1   isize=512    agcount=4,    agsize=6553600 blks
#          =            sectsz=512   attr=2, projid32bit=1
# data     =            bsize=4096   blocks=26214400, imaxpct=25
#          =            sunit=0      swidth=0 blks
# naming   =version 2  bsize=4096   ascii-ci=0, ftype=1
# log      =internal   bsize=4096   blocks=14400, version=2
#          =            sectsz=512   sunit=0 blks, lazy-count=1
# realtime =none        extsz=4096   blocks=0, rtextents=0

# XFS 有 4 个分配组（AG），每个 AG 独立管理空间
# → 多线程写入时，不同文件分布到不同 AG，减少锁竞争

# XFS 碎片整理（xfs_fsr：文件系统重排器）
xfs_fsr /mountpoint  # 整理整个挂载点
xfs_fsr -v /var/lib/postgres/data/base/16384/1259  # 整理单个文件

# XFS 实时空间（realtime subvolume）— 大文件顺序写特化
# mkfs.xfs -r rtdev=/dev/nvme1n1 /dev/nvme0n1  # 数据和实时卷分离
```

---

## 本章小结

| 主题 | 结论 |
|------|------|
| 空间 | 至少要区分字节、inode、逻辑大小、物理占用 |
| 稀疏文件 | 逻辑大不等于物理占用大 |
| 碎片 | 有内部、外部、extent 碎裂等不同层次 |
| 预分配 / 延迟分配 | 都影响空间形态，但工作方式不同 |
| quota / reserved blocks | 决定“谁还能继续写” |
| discard / TRIM | 服务于设备回收，不是文件系统万能清理 |
| 动态扩容 | 必须分清设备、分区/LVM、文件系统三层是否都已增长 |
| 缩容 | 通常比扩容更危险，也更依赖离线与备份策略 |

---

## 练习题

1. 为什么 `df`、`du`、`ls` 看到的“空间”可能不一致？
2. 内部碎片、外部碎片、extent 碎裂有什么区别？
3. 预分配和延迟分配分别解决什么问题？
4. 为什么 quota 和 reserved blocks 会让“有空间却写不进去”？
5. discard / TRIM 解决的是文件系统问题、设备问题，还是两者边界问题？
6. 为什么“底层磁盘已经扩大”并不自动等于 `df -h` 会立刻变大？
7. 为什么很多系统把在线扩容当常规操作，却把缩容视为高风险变更？
