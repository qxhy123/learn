# 第7章：块、超级块与空间分配

> **本章导读**：真正的文件系统布局不是”把文件切成块放到盘上”，而是在逻辑地址、物理介质、元数据索引、空闲空间、恢复成本和未来增长之间做持续权衡。本章梳理扇区/块/extent 的层次关系、超级块与块组的职责，以及稀疏文件、预分配和局部性策略背后的工程逻辑。

**前置知识**：第4章（inode/dentry 对象模型）
**预计学习时间**：50 分钟

---

## 学习目标

完成本章后，你将能够：

1. 区分扇区、块、页、extent、逻辑块号与物理块号
2. 说明超级块、块组、inode 表、位图、extent tree 在布局中的职责
3. 理解文件系统为什么要围绕局部性、对齐、未来增长和恢复做分配策略
4. 解释稀疏文件、预分配、未写 extent、碎片和空闲空间碎片的关系
5. 从性能和一致性角度理解“空间分配”为什么不是一个简单的找空块问题

---

## 正文内容

### 7.1 先把几个“块”分清楚

很多教程一上来就说“文件系统按块管理磁盘”，这句话没错，但太粗。真实系统里至少要分清这些粒度：

| 层次 | 典型含义 | 容易误解的点 |
|------|----------|--------------|
| 设备扇区 | 块设备暴露的最小或逻辑寻址单位 | 不一定等于文件系统块 |
| 物理介质粒度 | SSD erase block、RAID stripe、磁盘物理扇区等 | 文件系统可能看不到完整细节 |
| 文件系统块 | 文件系统分配与地址映射的基本单位 | 常和页大小相关，但不是概念上同一个东西 |
| 内存页 | page cache 管理文件内容的常见粒度 | 缓存页不等于磁盘块 |
| extent | 一段连续块区间的描述 | 不是数据本身，而是映射描述 |

例如一次写入可能从用户态 buffer 进入 page cache，再由文件系统分配逻辑块，最后被块层映射到设备请求。你看到的是 `write()`，内核看到的是跨多个层次的状态转换。

### 7.2 逻辑块号和物理位置不是同一回事

一个文件内部可以按“第 0 块、第 1 块、第 2 块”来理解，但这些逻辑块未必连续落在设备上。文件系统需要维护映射关系：

```text
文件逻辑范围 0..127  ->  设备块范围 A..A+127
文件逻辑范围 128..255 ->  设备块范围 B..B+127
```

这就是 extent 的直觉：用“起点 + 长度”描述连续区间，而不是为每个块都存一个指针。

这个映射关系会被很多因素影响：

- 文件是否顺序增长
- 当时空闲空间是否连续
- 是否使用延迟分配
- 是否预分配了空间
- 底层设备是否有对齐要求
- 文件系统是否尝试把相关文件放近

因此“文件大小相同”不代表布局相同，也不代表性能相同。

### 7.3 超级块记录的是“如何解释这个世界”

超级块（superblock）不是某个文件的元数据，而是整个文件系统实例的全局描述。它通常记录：

- 块大小
- 总块数、空闲块数
- inode 总数、空闲 inode 数
- 文件系统状态
- 特性位与兼容性标记
- 关键元数据结构位置
- 日志相关信息
- 校验、挂载次数、最近检查时间等管理信息

为什么它如此关键？因为没有超级块，内核甚至不知道如何解释后面的字节。超级块像“地图图例”：它告诉你这张地图上的符号、比例和入口在哪里。

工程上，很多文件系统会考虑超级块副本、元数据校验或恢复策略。原因很简单：全局解释入口坏了，局部数据还在也很难安全使用。

### 7.4 块组为什么不是“随便分区”

像 ext 系文件系统常用块组（block group）一类组织方式，把大文件系统切成多个局部管理区域。这样做的目标不是“看起来整齐”，而是：

- 让 inode、位图、数据块尽量在局部区域内协同
- 降低查找和分配时的全局扫描成本
- 改善目录与其文件之间的空间局部性
- 让 fsck / 恢复 / 统计可以更局部化
- 减少所有元数据都集中在少数区域造成的热点

可以把它想象成城市分区：如果所有居民、仓库、路牌和登记处都集中在一个点，任何操作都会挤在那里。块组把管理工作分散到了多个局部区域。

### 7.5 inode 表不是“文件内容区”

inode 表保存对象元数据，不保存普通文件的完整内容。它记录类似：

- 文件类型
- 权限、所有者、时间戳
- 大小
- 链接计数
- 数据块 / extent 映射入口
- 某些扩展属性或索引入口

这意味着一次文件访问通常要跨越多层结构：

```text
路径组件 -> 目录项 -> inode -> extent tree -> 数据块
```

如果你只把“文件”理解成“数据块”，就无法解释：

- 为什么 `stat` 不读完整内容也能知道大小和权限
- 为什么重命名常常不需要搬动数据块
- 为什么硬链接能让多个名字共享同一对象
- 为什么 inode 耗尽时磁盘字节空间还可能很多

### 7.6 空闲空间管理不只是“找到空块”

空闲空间可以用位图、空闲链表、extent tree 或更复杂结构追踪。真正的问题不是“有没有空块”，而是：

1. 有没有足够连续的空间？
2. 分配后会不会破坏未来增长？
3. 相关文件是否应当放近？
4. 分配元数据本身需要多少更新？
5. 崩溃后这些更新能不能恢复到一致状态？

举例来说，如果一个日志文件将持续增长，文件系统可能希望给它更连续的空间；如果一个目录下会有大量小文件，文件系统可能更关注 inode 与目录索引局部性。

### 7.7 预分配、稀疏文件与未写 extent

空间并不只有“已分配且有真实数据”一种状态。常见状态包括：

- **未分配洞（hole）**：逻辑上文件很大，中间部分没有真实物理块
- **已预分配但未写**：提前占住空间，避免未来写入时碎片化或失败
- **已写数据 extent**：物理块中已有有效内容
- **元数据保留空间**：为了未来操作或恢复保留的内部余量

这解释了为什么：

- `ls -lh` 与 `du -h` 可能不同
- 数据库会使用预分配减少运行期分配抖动
- 某些文件看起来很大，但实际占用很小

### 7.8 局部性为什么是性能的核心

局部性至少有三类：

1. **数据局部性**：同一文件的数据块尽量连续
2. **元数据局部性**：inode、目录项、索引结构尽量不分散
3. **访问局部性**：相关文件、目录和数据在工作负载中经常一起出现

没有局部性会带来：

- 更多随机 I/O
- 更复杂的 extent tree
- 更低的 readahead 效果
- 更高的元数据查找成本
- 更差的恢复扫描局部性

所以优秀的分配器不是“见缝插针”，而是在保留未来空间形态。

### 7.9 底层设备会反过来约束布局

文件系统并不是悬空设计的。底层设备会反过来影响布局策略：

- SSD 关注擦写块、写放大、TRIM / discard
- RAID 关注 stripe 对齐
- SMR 磁盘关注顺序写约束
- NVMe 关注并发队列和延迟特征
- 虚拟块设备可能还有宿主机层的再映射

如果文件系统分配策略与设备特性冲突，结果可能是：

- 写放大增加
- 尾延迟变差
- 空间回收滞后
- 基准测试与生产表现不一致

### 7.10 布局与崩溃恢复其实是一回事

空间分配本身也要被恢复机制保护。因为一次分配可能同时更新：

- 空闲位图
- inode 映射
- extent tree
- 文件大小
- 日志事务

如果中途崩溃，系统必须判断：

- 这个块到底算不算已分配？
- inode 是否应该指向它？
- 文件大小是否已经扩大？
- 目录项是否已经可见？

所以布局设计从来不是静态地图，而是恢复协议的一部分。

### 7.11 一个工程排查视角

当你遇到空间或布局问题，可以按下面顺序想：

1. 是字节空间耗尽，还是 inode 耗尽？
2. 是逻辑大小大，还是物理占用大？
3. 是单个大文件碎片化，还是小文件太多？
4. 是文件系统分配问题，还是底层设备对齐/回收问题？
5. 是正常删除，还是 deleted-but-open / snapshot / reflink 导致空间仍被引用？

这些问题会在后续容量、排障和性能章节继续出现。

---

### 7.12 ext4 超级块的关键字段与工具读取

超级块存在于 ext4 分区的第 1024 字节偏移（block group 0 的固定位置），下面是 `struct ext4_super_block`（`fs/ext4/ext4.h`）的核心字段：

```
ext4_super_block 关键字段（字节偏移 / 大小）：
偏移   大小  字段名                    含义
-----  ----  ------------------------  ----------------------------------------
0x00   4B    s_inodes_count            inode 总数
0x04   4B    s_blocks_count_lo         块总数（低 32 位）
0x08   4B    s_r_blocks_count_lo       为 root 预留的块数
0x0C   4B    s_free_blocks_count_lo    空闲块数（低 32 位）
0x10   4B    s_free_inodes_count       空闲 inode 数
0x14   4B    s_first_data_block        第一个数据块号（块大小 1K 时为 1，4K 时为 0）
0x18   4B    s_log_block_size          块大小 = 1024 << s_log_block_size（0→1K, 1→2K, 2→4K）
0x1C   4B    s_log_cluster_size        簇大小（bigalloc 特性用）
0x20   4B    s_blocks_per_group        每块组的块数（通常 8 * 块大小 bit，即 4K块时为32768）
0x24   4B    s_clusters_per_group      每块组的簇数
0x28   4B    s_inodes_per_group        每块组的 inode 数
0x2C   4B    s_mtime                   最近挂载时间
0x30   4B    s_wtime                   最近写入时间
0x34   2B    s_mnt_count               已挂载次数
0x36   2B    s_max_mnt_count           最大挂载次数（超过触发 fsck，-1 表示不检查）
0x38   2B    s_magic                   魔数：0xEF53（验证这是 ext 文件系统）
0x3A   2B    s_state                   文件系统状态：1=clean, 2=errors, 4=orphans
0x3C   2B    s_errors                  错误行为：1=continue, 2=ro-remount, 3=panic
0x3E   2B    s_minor_rev_level         次版本号
0x40   4B    s_lastcheck               最近 fsck 时间
0x44   4B    s_checkinterval           fsck 检查间隔（秒）
...
0x58   4B    s_rev_level               主版本号（0=老式, 1=动态支持扩展特性）
0x5C   2B    s_def_resuid              保留块默认 UID
0x5E   2B    s_def_resgid              保留块默认 GID
...（版本 >= 1 才有以下字段）
0x60   4B    s_first_ino               第一个非保留 inode（通常 11）
0x64   2B    s_inode_size              每个 inode 的磁盘大小（通常 256B）
0x66   2B    s_block_group_nr          本超级块副本所在的块组号
0x68   4B    s_feature_compat          兼容特性位（有此特性旧内核仍可挂载）
0x6C   4B    s_feature_incompat        不兼容特性位（缺少则拒绝挂载）
0x70   4B    s_feature_ro_compat       只读兼容特性位（缺少则只读挂载）
0x74   16B   s_uuid                    文件系统 UUID
...
0xFC   4B    s_checksum                超级块自身 checksum（crc32c）
```

**用工具读取超级块**：

```bash
# dumpe2fs：最完整的超级块信息（不需要 root，但需要读设备权限）
dumpe2fs /dev/sda1 2>/dev/null | head -80

# 典型输出节选：
# Filesystem magic number:  0xEF53
# Filesystem revision #:    1 (dynamic)
# Filesystem features:      has_journal ext_attr resize_inode dir_index filetype
#                           extent 64bit flex_bg sparse_super large_file huge_file
#                           dir_nlink extra_isize metadata_csum
# Filesystem state:         clean
# Block size:               4096
# Fragment size:            4096
# Group descriptor size:    64
# Blocks per group:         32768
# Inodes per group:         8192
# Inode blocks per group:   512
# Flex block group size:    16
# Reserved GDT blocks:      987

# tune2fs：修改超级块参数（危险！仅了解用途）
tune2fs -l /dev/sda1   # 与 dumpe2fs 类似，显示超级块参数
tune2fs -i 0 -c 0 /dev/sda1  # 禁用 mnt_count 和 interval 触发的 fsck

# 通过 /sys 查看已挂载文件系统的超级块状态（无需 root）
cat /sys/fs/ext4/sda1/errors_count   # 错误计数
cat /sys/fs/ext4/sda1/session_write_kbytes  # 本次挂载写入量（KB）
cat /sys/fs/ext4/sda1/lifetime_write_kbytes # 历史总写入量

# 直接 hexdump 超级块（偏移 1024 字节）
dd if=/dev/sda1 skip=2 bs=512 count=2 2>/dev/null | hexdump -C | head -30
# 在第 56-57 字节（0x38 偏移）找魔数 0xEF53（以小端存储：53 EF）
```

**超级块副本机制**：

```bash
# 查看所有备份超级块的位置
dumpe2fs /dev/sda1 2>/dev/null | grep "superblock"
# Primary superblock at 0, Group descriptors at 1-XXX
# Backup superblock at 32768, Group descriptors at 32769-...
# Backup superblock at 98304, Group descriptors at 98305-...

# 使用备份超级块修复损坏的主超级块
e2fsck -b 32768 /dev/sda1    # 从块 32768 的备份超级块恢复
```

---

### 7.13 块组描述符表：flex_bg 与元数据分布

每个块组有一个对应的 `struct ext4_group_desc`（或 64 位版本 `struct ext4_group_desc_64`）记录该块组内部元数据的位置：

```
ext4_group_desc 关键字段：
字段名                  大小  含义
----------------------  ----  -----------------------------------------------
bg_block_bitmap_lo      4B    块位图所在块号（低 32 位）
bg_inode_bitmap_lo      4B    inode 位图所在块号（低 32 位）
bg_inode_table_lo       4B    inode 表起始块号（低 32 位）
bg_free_blocks_count_lo 2B    本块组空闲块数
bg_free_inodes_count    2B    本块组空闲 inode 数
bg_used_dirs_count      2B    本块组包含的目录数（分配器优化用）
bg_flags                2B    EXT4_BG_INODE_UNINIT（inode 表未初始化，
                               lazy inode table 优化）等
bg_exclude_bitmap_lo    4B    快照排除位图（快照特性用）
bg_block_bitmap_csum_lo 2B    块位图 checksum
bg_inode_bitmap_csum_lo 2B    inode 位图 checksum
bg_itable_unused        2B    inode 表中未使用的 inode 数（lazy init 优化）
bg_checksum             2B    块组描述符自身 checksum
```

**flex_bg 特性**：ext4 默认把相邻的 16 个块组的元数据（块位图、inode 位图、inode 表）合并存放在第一个块组里，形成一个"弹性块组"（flexible block group）。这样元数据更集中，顺序读 inode 表的效率更高，减少随机 I/O。

```bash
# 查看块组详细信息（包含每个块组的元数据位置）
dumpe2fs /dev/sda1 2>/dev/null | grep -A 10 "^Group 0:"
# Group 0: (Blocks 0-32767) csum 0x... [ITABLE_ZEROED]
#   Primary superblock at 0, Group descriptors at 1-6
#   Reserved GDT blocks at 7-993
#   Block bitmap at 994 (+994), csum 0x...
#   Inode bitmap at 1010 (+1010), csum 0x...
#   Inode table at 1026-1537 (+1026)       ← inode 表占用 512 块（8192 inodes × 256B / 4096B）
#   28654 free blocks, 8181 free inodes, 2 directories, 8181 unused inodes

# 计算某个文件的 inode 在哪个块组
# inode_group = (inode_number - 1) / inodes_per_group
# inode_offset_in_group = (inode_number - 1) % inodes_per_group
# inode_block = bg_inode_table + inode_offset * inode_size / block_size

# 例：inode=12345，每组 8192 个 inode，inode_size=256，block_size=4096
# group = (12345 - 1) / 8192 = 1（第 2 个块组）
# offset = (12345 - 1) % 8192 = 4152
# block_in_table = 4152 * 256 / 4096 = 259（第 259 块）
debugfs -R "icheck 12345" /dev/sda1  # 快速验证 inode 位置
```

---

### 7.14 extent tree 节点结构：从根节点到叶节点

ext4 用 extent B-tree 描述文件的逻辑块到物理块的映射。理解树的节点结构，能帮你读懂 `debugfs stat` 的输出。

**节点类型**：

```c
/* 树头（存在于 inode 的 i_block 字段，或内部节点的块开头）*/
struct ext4_extent_header {
    __le16  eh_magic;       /* 0xF30A（验证）*/
    __le16  eh_entries;     /* 当前节点的有效 entry 数 */
    __le16  eh_max;         /* 当前节点最多能存 entry 数 */
    __le16  eh_depth;       /* 树深度（0 = 叶节点，叶节点直接有数据 extent）*/
    __le32  eh_generation;  /* 版本号（FIEMAP 等工具用）*/
};

/* 叶节点的 extent（直接描述数据块范围）*/
struct ext4_extent {
    __le32  ee_block;       /* 文件内的逻辑起始块号 */
    __le16  ee_len;         /* 连续块数（最高位为 1 时表示 unwritten extent）*/
    __le16  ee_start_hi;    /* 物理起始块号（高 16 位）*/
    __le32  ee_start_lo;    /* 物理起始块号（低 32 位）*/
};
/* 物理块号 = (ee_start_hi << 32) | ee_start_lo */
/* unwritten extent：ee_len 最高位为 1，实际长度 = ee_len & 0x7FFF */

/* 内部节点的索引 entry（指向下一层节点）*/
struct ext4_extent_idx {
    __le32  ei_block;       /* 此索引覆盖的逻辑块起始号 */
    __le32  ei_leaf_lo;     /* 子节点所在块号（低 32 位）*/
    __le16  ei_leaf_hi;     /* 子节点所在块号（高 16 位）*/
    __u16   ei_unused;
};
```

**inode 内联 extent tree**：

inode 的 `i_block[15]`（60 字节）存放 extent tree 根节点：
- `ext4_extent_header`（12 字节）：4 个字段
- 最多 4 个 `ext4_extent`（每个 12 字节）：4 × 12 = 48 字节

对于有连续磁盘布局的小文件（通常 < 4 个 extent），整个映射内联在 inode 里，无需额外块。

**用 debugfs 查看 extent tree**：

```bash
debugfs -R "dump_extents <inode_number>" /dev/sda1

# 例：查看 inode 16777 的 extent 分布
debugfs -R "dump_extents <16777>" /dev/sda1
# Level Entries Physical  Logical  Length Flags
#     0   1/4   131072/0     0    16384         ← 逻辑块 0 → 物理块 131072，长 16384 块（64MB）
# → 深度 0（单层叶节点），4 个 entry 中用了 1 个，直接内联在 inode 里

# 另一个碎片化的文件：
debugfs -R "dump_extents <12345>" /dev/sda1
#     0   4/4   40960       0       128          ← 第 1 个 extent：128 块
#     0   4/4   57344     128       64            ← 第 2 个 extent：64 块
#     0   4/4   49152     192       256           ← 第 3 个 extent：256 块
#     0   4/4   73728     448       512           ← 第 4 个 extent：512 块
# → 4 个 extent，填满了 inode 的内联槽位（需要多 extent 说明有碎片）

# 使用 filefrag 查看碎片化程度（用户态工具，不需要 root）
filefrag -v /path/to/file
# File size of /path/to/file is 2097152 (512 blocks of 4096 bytes)
# ext:     logical_offset:        physical_offset: length:   expected: flags:
#   0:        0..     511:      32768..     33279:    512:             last,eof
# → 只有 1 个 extent（连续），碎片化程度低
```

**unwritten extent 的含义**：

```bash
# fallocate 创建预分配文件，观察 unwritten extent
fallocate -l 100M /tmp/prealloc_test
debugfs -R "dump_extents <$(stat -c %i /tmp/prealloc_test)>" /dev/sda1
# Level Entries Physical  Logical  Length Flags
#     0   1/4   131072       0    25600  U    ← U 表示 unwritten（预分配但未写入数据）

# 写入数据后，unwritten extent 变为 written
dd if=/dev/urandom of=/tmp/prealloc_test bs=1M count=50
debugfs -R "dump_extents <$(stat -c %i /tmp/prealloc_test)>" /dev/sda1
# 前 50MB 的 extent 变为 written，后 50MB 仍为 unwritten
```

---

### 7.15 mballoc 分配策略：预分配与局部性

ext4 的多块分配器（mballoc，`fs/ext4/mballoc.c`）实现了多层预分配策略，目标是在分配时预测工作负载并保留未来增长空间。

**三级预分配**：

```
1. 目标数据预测（per-file preallocation）
   → 小文件写入时，mballoc 预分配 8 个额外块（inode 的 pa_inode 链表）
   → 后续 write() 优先从预分配的连续空间取块，避免碎片

2. 每 CPU 局部组预分配（locality group preallocation）
   → 小文件（< 256KB）进入当前 CPU 的局部组预分配池
   → 把同时期创建的小文件尽量集中到相邻块区域
   → 目标：相同时间窗口内创建的文件物理位置相近（时间局部性→空间局部性）

3. 块组优先选择策略
   → 大文件：优先在文件 inode 所在块组分配（或相邻块组）
   → 目录下的文件：优先与目录 inode 在同一块组
   → 写日志文件：特殊策略，倾向于分配到专用块组
```

**delayed allocation（延迟分配）与 mballoc 的配合**：

```
write() → 数据进入 page cache，标记 dirty，但逻辑块尚未分配
  ↓（回写时触发）
writepages() → 收集一批 dirty page → 计算逻辑块范围 → 调用 mballoc
  ↓
mballoc 看到完整写入范围后，做出更优的连续分配决策
  → 比"每次 write() 立即分配"减少 30-50% 的碎片（对顺序写工作负载）
```

**观察 mballoc 预分配统计**：

```bash
# ext4 各分配器统计（需要 debugfs 挂载）
mount -t debugfs none /sys/kernel/debug  # 通常已挂载

cat /sys/kernel/debug/ext4/sda1/mb_groups
# group: 0, len: 32768 free: 28654, fragmentation: 3, prealloc: 1024
# → prealloc 字段：当前块组有多少块被预分配但尚未写入

# 观察局部组预分配的效果（比较顺序写 vs 随机写后的碎片化）
# 顺序写：单个文件
dd if=/dev/urandom of=/tmp/seq_write bs=1M count=512 oflag=direct
filefrag /tmp/seq_write  # 通常 1-2 个 extent

# 并发写多个小文件（mballoc 局部组预分配发挥作用）
for i in $(seq 1 1000); do
    dd if=/dev/urandom of=/tmp/small_$i bs=4K count=1 2>/dev/null &
done; wait
# 查看这批小文件的物理位置是否聚集
for i in $(seq 1 10); do
    filefrag -v /tmp/small_$i | grep "physical_offset"
done
```

---

### 7.16 用工具观察完整空间分配状态

**空间使用全貌**：

```bash
# 字节空间 vs inode 空间
df -h /mountpoint    # 字节空间（blocks）
df -i /mountpoint    # inode 空间

# 典型场景：字节空间还有，但 inode 耗尽
df -i /var
# Filesystem     Inodes  IUsed  IFree IUse% Mounted on
# /dev/sda1     3276800 3276800     0  100% /var
# → IFree = 0，再 create 文件就报 "No space left on device"（即使 df 显示有块）

# 找到哪个目录消耗了最多 inode
find /var -xdev -printf '%h\n' | sort | uniq -c | sort -rn | head -20

# 空闲 inode 的分布（按块组）
dumpe2fs /dev/sda1 2>/dev/null | grep -E "Group [0-9]+:|Free inodes:" | paste - - | head -20
```

**碎片化分析**：

```bash
# e4defrag：分析整个文件系统或目录的碎片化（不实际整理）
e4defrag -c /mountpoint    # 打印碎片化报告，-c 为 check only
e4defrag -c /path/to/file  # 单文件碎片化分析

# 典型输出：
# Current/best extents: 512/1 ... Fragmentation score: 87
# → 当前 512 个 extent，理想情况下只需 1 个，碎片化评分 87（满分 100）

# 查看最碎片化的文件
find /mountpoint -xdev -type f | xargs filefrag 2>/dev/null | \
    awk '{if ($2 > 10) print $2, $NF}' | sort -rn | head -20

# 磁盘 free space 碎片化分析（空闲块是否连续）
dumpe2fs /dev/sda1 2>/dev/null | grep "^Group" | \
    awk '{match($0, /Free blocks: ([0-9,-]+)/, a); print a[1]}' | \
    awk -F',' '{print NF, $0}' | sort -rn | head -10
# → 每个块组的空闲块是否成片还是散碎
```

**deleted-but-open 与空间回收**：

```bash
# 找到所有 deleted-but-open 文件（名字删除但 fd 仍持有）
lsof +L1 2>/dev/null | awk 'NR==1 || $7=="(deleted)"'
# COMMAND  PID USER FD TYPE DEVICE SIZE/OFF NLINK NAME
# python3  1234 root 3u REG  8,1 104857600 0 /tmp/bigfile (deleted)
# → SIZE/OFF 104857600（100MB）被占用但 df 和 du 无法发现

# 不重启服务的情况下回收空间（截断 deleted-but-open 文件）
PID=1234; FD=3
truncate -s 0 /proc/$PID/fd/$FD  # 立即释放磁盘空间，进程继续运行
# 注意：服务可能在下次 write() 时报错，需评估影响

# 更安全的方式：向服务发 SIGHUP 让其重新打开日志文件
kill -HUP $PID  # 大多数守护进程会重新打开日志 fd
```

---

## 本章小结

| 主题 | 结论 |
|------|------|
| 粒度 | 扇区、页、文件系统块、extent 是不同层次 |
| 超级块 | 定义文件系统实例如何被解释 |
| 块组 | 为局部性、分配和恢复服务 |
| inode 表 | 保存对象元数据与映射入口，不是文件内容本身 |
| 空闲空间 | 分配策略要兼顾连续性、未来增长和恢复 |
| 设备特性 | SSD、RAID、虚拟块设备会反向影响布局设计 |
| 恢复 | 空间分配也是崩溃一致性协议的一部分 |

---

## 练习题

### 基础题

**7.1** 为什么扇区、页、文件系统块和 extent 不能混为一谈？列出每层负责的职责和边界，以及混淆它们会导致哪些错误推断。

**7.2** 超级块为什么像文件系统实例的”解释入口”？描述超级块损坏后的后果，以及 ext 系列文件系统如何为超级块提供保护。

### 中级题

**7.3** 块组为什么能改善局部性和恢复成本？比较”有块组”和”无块组”两种设计在元数据访问延迟、分配策略和 fsck 性能上的具体差异。

**7.4** 稀疏文件、预分配和已写 extent 有什么区别？解释为什么同一个文件的 `ls -lh` 与 `du -h` 输出可能相差悬殊，各自反映的是哪种大小。

### 提高题

**7.5** 为什么说空间分配本身也是崩溃一致性问题？分析一次完整分配操作需要原子更新哪些元数据结构，并比较日志化（ext4 data=ordered）和 COW（btrfs）两种方案如何保证分配结果的崩溃一致性。

---

## 练习答案

**7.1** 扇区是块设备暴露的最小寻址单位（512B 或 4KB 逻辑扇区），文件系统块是分配和地址映射的基本单位（通常 4KB，不等于扇区），内存页是 page cache 管理文件内容的粒度（通常 4KB，但缓存页与磁盘块是不同抽象），extent 是”起点+长度”的连续区间映射描述，不是数据本身。混淆导致的错误：误以为写一个字节只影响一个扇区（实际可能触发整个文件系统块的 read-modify-write）；误以为 extent tree 保存的是数据（实际是映射关系）；误以为 page cache 命中就无设备 I/O（实际仍可能触发元数据 I/O）。

**7.2** 超级块记录块大小、总块数、空闲块数、inode 总数、关键元数据结构位置等，没有它内核无法解释文件系统实例的任何字节。损坏后果：文件系统拒绝挂载，即使数据块完整也极难安全访问。保护机制：ext 系列在多个块组的固定偏移处存有超级块副本（`dumpe2fs` 可查看副本位置），`e2fsck -b <backup_sb>` 可指定副本恢复；ext4 还对超级块和块组描述符使用 metadata checksum 检测静默损坏。

**7.3** 有块组设计：inode 表、数据块位图、数据块集中在局部区域；分配器优先在文件所在目录的块组内为新文件分配 inode 和数据块；fsck 可逐块组独立扫描，理论上可并行化。无块组设计：所有 inode 表集中在磁盘头部，所有文件操作都需跨越整个磁盘；恢复必须全局线性遍历。性能差异：旋转磁盘场景下，块组内的局部分配将 inode 与数据块的寻道距离控制在块组范围内，随机 I/O 显著减少；fsck 耗时在有块组时随文件系统大小线性增长更平缓。

**7.4** 稀疏文件洞（hole）：文件逻辑范围存在但无真实物理块，读返回零，不占磁盘空间；预分配（fallocate FALLOC_FL_KEEP_SIZE）：提前占住连续物理块但可能未写真实数据，`du` 计入但文件未必增长逻辑大小；已写 extent：物理块含有效内容，两种大小一致。`ls -lh` 显示逻辑大小（包含洞），`du -h` 显示实际已分配块数，差异来源：①稀疏文件的洞使逻辑大小远大于物理占用；②预分配使物理占用大于已写内容，而逻辑大小可能仍小（取决于 flag）。

**7.5** 一次分配需要原子更新：①空闲位图（将目标块标为已用）；②inode 的 extent tree（插入新映射条目）；③文件大小字段；④块组描述符的空闲计数；⑤日志事务记录。崩溃在任意步骤间发生的风险：位图已更新但 inode 未更新 → 块泄漏（空间永久丢失直到 fsck）；inode 已更新但位图未更新 → 双重分配风险（两个文件可能指向同一物理块）。ext4 data=ordered 方案：先将元数据更新写入日志，`jbd2_journal_commit_transaction` 完成后才更新磁盘元数据，崩溃后重放日志恢复一致状态。btrfs COW 方案：写入新物理位置，更新树指针后原子 swap，崩溃总能回退到上一个一致快照，无需重放日志。

---

## 延伸阅读

1. **Arpaci-Dusseau**. *Operating Systems: Three Easy Pieces*, Ch.39-42 — 文件系统布局、FFS 局部性设计与 ext4 详解
2. **Linux 内核文档**. `Documentation/filesystems/ext4/` — ext4 磁盘布局与元数据结构规范
3. **Mathur, Cao, et al.** *The new ext4 filesystem: current status and future plans* (OLS 2007) — extent tree 与延迟分配设计原文
4. **man 8 dumpe2fs** — 超级块与块组信息读取，可查看副本位置
5. **man 1 fallocate / man 2 fallocate** — 预分配接口、稀疏文件操作与各 flag 语义

---

[← 上一章：从路径到 inode：查找过程](../part2-core-abstractions/06-from-path-to-inode-how-lookup-works.md)

[下一章：ext4 布局与日志 →](./08-ext4-layout-and-journaling.md)

[返回目录](../README.md)
