# 第4章：inode、dentry 与目录项

> **本章导读**：Linux 文件系统对象模型的关键不只是”名字和对象分离”，而是把名字关系、对象元数据、打开状态、缓存和视图上下文拆成了多层，这种拆分决定了后面几乎所有语义。本章建立 inode / dentry / file 三层对象图，并通过 rename、unlink、alias 等操作验证对象模型的完整性。

**前置知识**：第1-3章（文件系统基础、路径、权限）
**预计学习时间**：50 分钟

---

## 学习目标

完成本章后，你将能够：

1. 区分目录项、dentry、inode、file 各自属于哪一层
2. 解释为什么 dcache 需要缓存“存在”和“不存在”两类结果
3. 理解 alias、hard link、rename、unlink 对对象图的影响
4. 说明 dentry 生命周期、revalidation 与名字缓存为什么重要
5. 为后续 path walk、VFS、open file 语义建立对象基础

---

## 正文内容

### 4.1 inode 解决“对象是什么”

inode 描述的是对象本体，而不是名字。它通常保存：

- 文件类型
- 权限与所有权
- 时间戳
- 大小
- 链接计数
- 数据块 / extent 映射入口
- 扩展属性等额外元数据

这意味着 inode 回答的问题是：

- 这是个什么对象？
- 它的元数据是什么？
- 它的数据或索引入口在哪里？

而不是“它当前叫什么名字”。

### 4.2 目录项解决“这个名字在这个父目录下指向谁”

目录项（directory entry）是**父目录上下文里的名字关系**。这句话很重要，因为：

- 同一个名字在不同目录下不是同一个关系
- 同一个 inode 可以被多个目录项引用
- 重命名首先影响目录项关系，而不是对象本体

所以目录项不是对象元数据，也不是打开状态，而是“名字-父目录-目标对象”这组三元关系的磁盘或逻辑表达。

### 4.3 dentry 是 VFS 的名字缓存节点，不是磁盘目录项复制品

Linux 内核中的 dentry（directory cache entry）不是简单把磁盘目录项搬到内存里，而是 VFS 路径解析中的核心节点对象。它常负责：

- 把“父目录 + 名字”映射到某个结果
- 记录与 inode 的关联
- 参与路径遍历和缓存失效
- 缓存名字不存在的结果（negative dentry）
- 支撑 rename、unlink、mount crossing 等过程中的名字视图

因此 dentry 更接近“路径解析图上的节点”，而不只是目录项缓存。

### 4.4 为什么 negative dentry 必须存在

系统中大量查找其实是失败查找。例如：

- 动态链接器试探候选库路径
- shell 或运行时探测多个配置文件位置
- Web 服务按优先级尝试不同静态资源路径

如果每次失败都去底层目录结构重新查，会非常贵。negative dentry 缓存“这个名字在这个父目录下不存在”的结果，可以显著减少重复失败查找成本。

所以 dcache 的价值不只在于“加速命中”，还在于“加速确认不存在”。

### 4.5 alias 与硬链接说明了什么

如果一个 inode 被多个目录项引用，那么同一个对象会通过多个名字出现。这会带来：

- 多个 dentry 可能关联同一个 inode
- rename 只改其中一条名字关系时，对象本体仍不变
- unlink 某个名字时，对象未必消失

这说明 dentry 和 inode 是多对一关系，而不是一一对应关系。

### 4.6 dentry 生命周期：为什么缓存不是永久真理

名字缓存并不是永远正确。目录内容可能因为：

- rename
- unlink
- create
- mount / unmount
- 远端文件系统 revalidation

发生变化。因此 dentry 还要考虑：

- 是否仍然有效
- 是否需要 revalidate
- 是否应被回收
- 是否仍被某条 path 或 open file 间接依赖

在本地文件系统里，dcache 常更稳定；在 NFS、FUSE 等场景里，revalidation 的重要性更高，因为内核不一定独占名字真相。

### 4.7 为什么 file 必须再单独分一层

就算路径已经解析到 inode，也还不够。因为打开一个文件后，还要保存：

- 当前偏移量
- 打开标志
- 锁、通知、异步 I/O 上下文
- 与具体 file operations 的绑定

这就是 file / open file description 层存在的意义：它解决的是“已经打开之后，运行时如何持续操作它”，而不是“它是什么对象”。

### 4.8 rename 为什么最能暴露对象层次

`rename(a, b)` 常常同时证明这几件事：

- 名字可以变
- inode 可以不变
- 已打开 file 仍可继续工作
- dentry 关系需要更新或失效
- 缓存和目录项都可能变化

如果你的对象模型不能同时解释这五件事，就说明它还不够细。

### 4.9 unlink 为什么不是“删除文件内容”

`unlink` 主要删除目录项关系。只有当：

- 链接计数降到 0
- 且没有 open file description 持有引用

对象才可能真正进入回收路径。

这说明“名字消失”和“对象回收”是两条不同生命周期。理解这一点，是理解 deleted-but-open、日志轮转、临时文件协议的前提。

### 4.10 一个对象图心智模型

可以用下面这个简化模型记忆：

```text
+ 路径字符串
+ 当前视图（cwd/root/mount namespace)
-> dentry（名字解析节点）
-> inode（对象元数据）
-> file（打开后的运行时状态）
-> 数据页 / extent / address_space
```

这个图之所以重要，是因为后面的 VFS、page cache、rename、fsync、overlayfs 都会在这个图上工作，而不是直接“对文件做操作”。

### 4.11 内核对象的真实字段：struct inode、struct dentry、struct file

概念层面的三层分离，在内核里体现为三个核心 C 结构体。理解它们的关键字段，能帮你把高层直觉落到可查的代码证据上。

#### struct inode（`include/linux/fs.h`）

```c
struct inode {
    umode_t         i_mode;         /* 文件类型 + 权限位（S_IFREG / S_IFDIR 等 + rwxrwxrwx）*/
    unsigned short  i_opflags;
    kuid_t          i_uid;          /* 拥有者 UID */
    kgid_t          i_gid;          /* 拥有者 GID */
    unsigned int    i_flags;        /* 挂载/锁相关 flag */

    const struct inode_operations *i_op;   /* 对象级操作表（create/link/rename/lookup 等）*/
    struct super_block *i_sb;              /* 所属文件系统实例 */
    struct address_space *i_mapping;       /* page cache 关联的地址空间 */

    unsigned long   i_ino;          /* inode 编号（在本 superblock 内唯一）*/
    union {
        const unsigned int i_nlink;  /* 硬链接计数 */
    };
    dev_t           i_rdev;         /* 设备号（字符/块设备 inode 用）*/
    loff_t          i_size;         /* 文件逻辑大小（字节）*/
    struct timespec64 i_atime;      /* 最近访问时间 */
    struct timespec64 i_mtime;      /* 内容最近修改时间 */
    struct timespec64 i_ctime;      /* inode 状态最近变化时间 */

    unsigned int    i_blkbits;      /* 块大小（以 bit 为单位的 log2）*/
    blkcnt_t        i_blocks;       /* 已分配块数（512B 为单位）*/

    atomic_t        i_count;        /* 引用计数（open file description 持有时 > 0）*/
    atomic_t        i_writecount;   /* 打开写引用数 */

    const struct file_operations *i_fop;   /* 打开后的操作表（read/write/mmap/fsync 等）*/
    struct file_lock_context *i_flctx;     /* 文件锁上下文 */

    /* 文件系统私有数据（如 ext4_inode_info 通过 container_of 访问）*/
};
```

关键关系：

- `i_ino` 是这个 inode 的"身份"，目录项里存的就是这个号，名字 → inode 的映射在目录数据里，不在 inode 里。
- `i_nlink` 记录有多少目录项指向这个 inode；`unlink` 先减 1，降到 0 且 `i_count == 0` 时 inode 才真正释放。
- `i_count` 是内存引用计数，open file description 持有它时不会被回收；`i_nlink` 是磁盘名字引用计数，两者独立。
- `i_fop` 是运行时 I/O 操作表，打开后的 read/write/fsync 都从这里分发，和 `i_op`（对象级操作，如 rename/link/mkdir）不同。

#### struct dentry（`include/linux/dcache.h`）

```c
struct dentry {
    unsigned int d_flags;           /* DCACHE_NEGATIVE：负 dentry；DCACHE_OP_REVALIDATE：需要 revalidate */
    seqcount_spinlock_t d_seq;      /* RCU path walk 用的序列号 */
    struct hlist_bl_node d_hash;    /* dcache 哈希表链表节点 */
    struct dentry *d_parent;        /* 父 dentry */
    struct qstr d_name;             /* 名字（hash + 字符串）*/
    struct inode *d_inode;          /* 关联的 inode（负 dentry 时为 NULL）*/
    unsigned char d_iname[DNAME_INLINE_LEN]; /* 短名字内联存储，避免额外分配 */

    struct lockref d_lockref;       /* 引用计数 + 自旋锁（合并为一个原子单元）*/
    const struct dentry_operations *d_op;  /* 操作表（compare/revalidate/hash 等）*/
    struct super_block *d_sb;       /* 所属 superblock */
    unsigned long d_time;           /* revalidation 时间戳 */
    void *d_fsdata;                 /* 文件系统私有数据 */

    union {
        struct list_head d_lru;     /* 未使用时的 LRU 链表节点 */
        wait_queue_head_t *d_wait;
    };
    struct list_head d_child;       /* 在父 dentry 的子链表中的节点 */
    struct list_head d_subdirs;     /* 子 dentry 链表头 */
    union {
        struct hlist_node d_alias;  /* inode 的 alias 链表（多个 dentry 指向同一 inode）*/
        struct hlist_bl_node d_in_lookup_hash;
        struct rcu_head d_rcu;
    } d_u;
};
```

关键字段解读：

- `d_inode == NULL` 表示这是一个**负 dentry**，即"这个名字确认不存在"的缓存记录。
- `d_alias` 把所有指向同一 inode 的 dentry 串联成链表，可以有多个（硬链接场景）。
- `d_lru` 在 dentry 没有活跃引用时放入 LRU 链表，等待内存回收；`d_lockref` 的引用计数降到 0 时才会进入 LRU。
- `d_op->d_revalidate`：对于 NFS/FUSE 等文件系统不为 NULL，path walk 命中缓存后还会调用它确认有效性；ext4 通常不需要，直接信任缓存。

#### struct file（open file description，`include/linux/fs.h`）

```c
struct file {
    union {
        struct llist_node   f_llist;
        struct rcu_head     f_rcuhead;
        unsigned int        f_iocb_flags;
    };
    spinlock_t              f_lock;
    fmode_t                 f_mode;         /* 打开模式（FMODE_READ / FMODE_WRITE 等）*/
    atomic_long_t           f_count;        /* 引用计数（dup/fork 共享同一 file 时增加）*/
    struct mutex            f_pos_lock;     /* 保护 f_pos 的锁 */
    loff_t                  f_pos;          /* 当前文件偏移量 */
    unsigned int            f_flags;        /* O_APPEND / O_NONBLOCK / O_DIRECT 等 flag */
    const struct file_operations *f_op;     /* 操作表（从 inode->i_fop 复制而来）*/
    struct address_space    *f_mapping;     /* page cache（通常 == inode->i_mapping）*/
    void                    *private_data;  /* 文件系统/驱动私有数据 */
    struct inode            *f_inode;       /* 关联的 inode */
    struct path             f_path;         /* dentry + vfsmount（路径上下文，用于 AT_FDCWD 等）*/
};
```

- `f_pos` 就是 lseek/read/write 依赖的偏移量，它**属于 file，不属于 inode**。两次 `open()` 得到两个不同的 file，各自有独立的 `f_pos`；`dup()` 共享同一 file，共享同一 `f_pos`。
- `f_flags` 保存打开时的标志，`O_APPEND` 在每次 `write()` 前让内核自动 seek 到文件末尾。
- `f_count` 是这个 file 对象自己的引用计数；`fork()` 后父子进程的 fd 表共同持有同一 file，`f_count` 增加；任何一方关闭时减少，降到 0 时 file 对象才释放（此时 inode 的 `i_count` 也相应减少）。

---

### 4.12 ext4 磁盘 inode 布局：从权限位到 extent tree 入口

VFS 的 `struct inode` 是内存对象，真正落到磁盘上的是 ext4 的 `struct ext4_inode`（`fs/ext4/ext4.h`）。理解两者的对应关系，能帮你把 `stat` 的输出和 `debugfs` 的 `stat <inode>` 连起来看。

```
ext4_inode 磁盘布局（256 字节，部分字段）:
偏移  大小  字段名              含义
----  ----  -----------------  ----------------------------------------
0x00  2B    i_mode             文件类型 + 权限（与 VFS i_mode 直接对应）
0x02  2B    i_uid_lo           UID 低 16 位
0x04  4B    i_size_lo          文件逻辑大小低 32 位（字节）
0x08  4B    i_atime            最近访问时间（Unix 时间戳）
0x0C  4B    i_ctime            inode 状态变化时间
0x10  4B    i_mtime            内容修改时间
0x14  4B    i_dtime            删除时间（被 unlink 后记录，用于 fsck）
0x18  2B    i_gid_lo           GID 低 16 位
0x1A  2B    i_links_count      硬链接计数
0x1C  4B    i_blocks_lo        已分配块数（512B 为单位，兼容老接口）
0x20  4B    i_flags            EXT4_EXTENTS_FL 等标志位
...
0x28  60B   i_block[15]        extent tree 入口 OR 传统三级间接块指针
             （EXT4_EXTENTS_FL 置位时存 extent tree 根节点，否则存传统指针）
...
0x74  2B    i_extra_isize      额外 inode 空间大小（ext4 扩展字段起点）
0x76  2B    i_checksum_hi      inode checksum 高 16 位
0x78  4B    i_ctime_extra      ctime 纳秒精度扩展
0x7C  4B    i_mtime_extra      mtime 纳秒精度扩展
0x80  4B    i_atime_extra      atime 纳秒精度扩展
0x84  4B    i_crtime           文件创建时间（ext4 扩展，非 POSIX 标准）
0x88  4B    i_crtime_extra     创建时间纳秒精度扩展
0x8C  4B    i_version_hi       inode 版本高 32 位
...
```

几个值得关注的细节：

**extent tree 入口（`i_block[15]`，60 字节）**

当 `i_flags` 的 `EXT4_EXTENTS_FL` 位为 1 时，`i_block` 存放的不是传统三级间接块指针，而是 extent B-tree 的根节点（`struct ext4_extent_header` + 最多 4 个 `struct ext4_extent`）。对于小文件，所有 extent 能直接放在 inode 里（内联 extent），无需额外块。

**`i_dtime`（删除时间）**

`unlink` 后不为 0，fsck 用它判断这个 inode 是否已被标记为"将被回收"。如果崩溃恰好在 unlink 后但 inode 回收前，fsck 看到 `i_dtime != 0` 且 `i_links_count == 0`，知道要继续回收流程。

**用 `debugfs` 观察磁盘 inode**

```bash
# 以只读方式打开 ext4 分区（/dev/sdX1 替换为实际设备）
debugfs -R "stat <inode_number>" /dev/sdX1

# 例如：查看 inode 12（通常是 /lost+found）
debugfs -R "stat <12>" /dev/sdX1

# 典型输出（节选）:
# Inode: 12   Type: directory    Mode:  0700   Flags: 0x80000
# Generation: 0    Version: 0x00000000:00000001
# User:     0   Group:     0   Project:     0   Size: 16384
# File ACL: 0
# Links: 2   Blockcount: 32
# Fragment:  Address: 0    Number: 0    Size: 0
# ctime: 0x...  -- Mon Jan 01 00:00:01 2024
# atime: 0x...  -- Mon Jan 01 00:00:01 2024
# mtime: 0x...  -- Mon Jan 01 00:00:01 2024
# crtime: 0x... -- Mon Jan 01 00:00:01 2024
# Size of extra inode fields: 28
# EXTENTS:
# (0): [12288/8]   ← 逻辑块0，物理块12288起，长度8块
```

通过 `stat` 系统调用看到的 `ino`、`nlink`、`size`、`blocks` 字段，都直接来自磁盘 inode 的对应字段（经由内核填入 VFS 内存 inode 再返回用户态）。

---

### 4.13 一次 open() 的全生命周期：从 path walk 到 file 对象创建

理解三层对象如何协作，最直接的方式是跟踪一次 `open("/var/log/syslog", O_RDONLY)` 的完整路径。

```
用户态：open("/var/log/syslog", O_RDONLY)
   │
   ▼ 系统调用入口
sys_openat(AT_FDCWD, "/var/log/syslog", O_RDONLY, 0)
   │
   ▼ do_sys_openat2()
   1. 分配 struct open_flags（解析 flags/mode）
   2. 调用 do_filp_open()
         │
         ▼ path_openat()
         3. 初始化 nameidata nd（记录当前解析位置、namespace、flag）
         4. 调用 link_path_walk("/var/log/syslog", nd)
               │
               ▼ 逐段解析
               "/" → 从 nd.root（当前进程 root）出发
               "var" → 在根 dentry 的子树中查找 "var"
                     → dcache 命中（struct dentry *） 
                     → 检查 inode->i_mode & S_IFDIR（是目录？）
                     → 检查 inode_permission(inode, MAY_EXEC)（有 x 权限？）
               "log" → 继续在 /var 的 dentry 中查找 "log"
                     → 可能遇到 mount point：调用 __follow_mount()
                       切换到新 superblock 的 root dentry
               "syslog" → 查找 "syslog"
                     → dcache 未命中 → 调用 inode->i_op->lookup()
                       即 ext4_lookup() → 读目录块 → 找到 "syslog" → inode 号
                       → 分配新 dentry，关联 inode，插入 dcache 哈希表
         5. 路径解析完成，nd 持有目标 dentry + vfsmount
         6. 调用 do_open(nd, file, op)
               │
               ▼
               7. 分配 struct file 对象
               8. 调用 vfs_open()
                     → file->f_op = inode->i_fop（ext4_file_operations）
                     → file->f_pos = 0
                     → file->f_flags = O_RDONLY
                     → 调用 file->f_op->open()（ext4_file_open，建立 extent 缓存等）
   │
   ▼ 回到 do_sys_openat2()
   9. 在进程 fd 表中分配 fd（整数槽位）
   10. 把 struct file * 填入 fd 表槽位
   │
   ▼ 返回用户态
   open() 返回 fd（整数）
```

这条路径说明：
- **fd** 只是进程 fd 表里的整数槽位，最后才分配。
- **struct file** 在第 7 步才创建，保存运行时状态（偏移、flag、操作表）。
- **struct dentry** 在 path walk 过程中被查找或创建，path walk 结束后 file 持有它的引用。
- **struct inode** 通过 dentry 访问，始终代表对象本体，不随路径解析而复制。

用 `strace` 观察这条路径：

```bash
strace -e trace=openat,newfstatat,read cat /var/log/syslog 2>&1 | head -20

# 典型输出（节选）：
# openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
# openat(AT_FDCWD, "/lib/x86_64-linux-gnu/libc.so.6", O_RDONLY|O_CLOEXEC) = 3
# openat(AT_FDCWD, "/var/log/syslog", O_RDONLY) = 3
# newfstatat(3, "", {st_mode=S_IFREG|0640, st_size=...}, AT_EMPTY_PATH) = 0
# read(3, "Jan ...", 131072) = 65536
```

`openat(AT_FDCWD, "/var/log/syslog", O_RDONLY) = 3` 中：
- `AT_FDCWD` 表示从当前工作目录出发（相对路径）或从进程根出发（绝对路径）。
- 返回值 `3` 就是分配到的 fd。

**观察对象关系的命令**：

```bash
# 找到进程 PID
PID=$(pgrep -f "cat /var/log/syslog")

# 查看该进程的所有 fd
ls -la /proc/$PID/fd/
# lrwx------ 1 root root 64 ... 0 -> /dev/pts/0   （标准输入）
# lrwx------ 1 root root 64 ... 1 -> /dev/pts/0   （标准输出）
# lr-------- 1 root root 64 ... 3 -> /var/log/syslog  （刚打开的文件）

# 查看 file 对象的详细信息（包含 offset）
cat /proc/$PID/fdinfo/3
# pos:    65536        ← 当前文件偏移（read 后更新）
# flags:  0100000      ← O_RDONLY
# mnt_id: 22           ← mount ID

# 查看 inode 信息
stat /proc/$PID/fd/3
# File: /proc/12345/fd/3 -> /var/log/syslog
# Size: 12345678    Blocks: 24112    IO Block: 4096  regular file
# Device: 802h/2050d    Inode: 789012    Links: 1
```

---

### 4.14 dcache 的内存组织与内存压力下的回收

dcache 不是一个简单的哈希表——它是同时支持快速名字查找和 LRU 回收的复合结构。

**查找结构**：全局哈希表 `dentry_hashtable`，key 是 `(parent_dentry, name_hash)`。path walk 的每一段都先用 `d_lookup(parent, &name)` 查这张表，O(1) 命中则直接返回 dentry，无需下穿到底层文件系统。

**回收结构**：每个 super_block 维护一个 `s_dentry_lru` LRU 链表，存放引用计数降到 0 的 dentry（即没有 path walk 和 open file 正在使用）。内核的 memory shrinker 在内存紧张时调用 `prune_dcache_sb()` 从链表尾部批量回收。

**dcache 容量与压力**：

```bash
# 查看当前 dcache 使用量
cat /proc/sys/fs/dentry-state
# 例：220541 171823 45 0 0 0
# 字段含义：total(已分配dentry数) unused(LRU中待回收) age_limit want_pages(shrinker请求)...

# 主动释放 dcache（会导致下次 path walk 重新到底层文件系统查找，性能暂时下降）
echo 2 > /proc/sys/vm/drop_caches  # 释放 dcache + page cache

# 观察释放前后的 path walk 性能差异（用 find 产生大量查找）
time find /usr -name "*.so" > /dev/null   # drop_caches 前：快（命中率高）
echo 2 > /proc/sys/vm/drop_caches
time find /usr -name "*.so" > /dev/null   # drop_caches 后：慢（需要重新从磁盘读目录）
```

**负 dentry 的生命周期**：

创建负 dentry 的时机：lookup 底层文件系统确认某名字不存在后，内核创建 `d_inode == NULL` 的 dentry 放入 dcache。后续对同一名字的查找直接命中此负 dentry，返回 `ENOENT`，无需再次查底层文件系统。

负 dentry 的失效时机：
1. 同名文件被 `creat()`/`mkdir()` 创建 → 负 dentry 转为正 dentry（`d_inode` 被填入）。
2. 内存压力下 shrinker 回收 → 下次查找再次穿透到底层文件系统。
3. 远端文件系统的 `d_op->d_revalidate` 返回 0 → 主动失效。

**dcache 热点与元数据瓶颈**：

在高并发创建/删除场景（如海量小文件工作负载），dcache 哈希表的桶锁（`hb->lock`）成为争抢点。每次 path walk 都要对父 dentry 上读锁（RCU path walk）或写锁（rename/unlink），同一热门目录的写操作会序列化，这就是"目录元数据热点"（directory metadata bottleneck）的根本原因，与磁盘 I/O 无关。

---

### 4.15 alias 链表、引用计数与 inode 回收时机

同一 inode 被多个 dentry 引用（硬链接）时，内核通过 `dentry->d_alias` 链表把它们串联在 inode 上。这个链表决定了 inode 何时才能真正回收。

**inode 的引用计数来源**：

```
inode->i_count 增加来源：
  1. dentry 指向它（path walk 找到时）
  2. struct file 打开它（open() 调用时）
  3. bind mount 把它用作根
  4. overlayfs 持有下层引用
```

**回收条件的精确表述**：

```
inode 可回收当且仅当：
  i_count == 0  （无任何 dentry 或 file 引用）
  AND
  i_nlink == 0  （所有目录项名字都已删除，即 unlink 到 0）

若 i_count == 0 但 i_nlink > 0：
  → inode 可缓存在内存（i_count 降到 0 时放入 inode LRU），
    有新路径查找时可从缓存恢复，无需重新读磁盘

若 i_count > 0 但 i_nlink == 0：
  → deleted-but-open：名字已从目录树消失，但 file 仍持有 inode
    → 磁盘空间未释放，ls 找不到，lsof 能找到
    → 所有 file 关闭后 i_count 降到 0 → 进入真正回收路径
```

**用工具观察 alias 和引用计数**：

```bash
# 创建硬链接
echo "hello" > /tmp/orig
ln /tmp/orig /tmp/alias1
ln /tmp/orig /tmp/alias2

# 查看 inode 号和链接计数
ls -li /tmp/orig /tmp/alias1 /tmp/alias2
# 3145729 -rw-r--r-- 3 user user 6 ... /tmp/alias1
# 3145729 -rw-r--r-- 3 user user 6 ... /tmp/alias2
# 3145729 -rw-r--r-- 3 user user 6 ... /tmp/orig
# ↑ 同一 inode 号（3145729），链接计数 3

# 删除原名，其他名字仍可访问
rm /tmp/orig
stat /tmp/alias1  # inode 不变，nlink 变为 2
cat /tmp/alias1   # 内容仍然可读

# 打开后删除所有名字，观察 deleted-but-open
exec 9< /tmp/alias1  # bash fd 9 打开文件
rm /tmp/alias1 /tmp/alias2  # 删除所有名字
ls /tmp/orig /tmp/alias1 2>&1  # 报错：文件不存在
cat /proc/self/fd/9  # 但仍可通过 fd 读取！
# → 说明：i_nlink == 0，但 i_count > 0（fd 9 持有）
exec 9<&-  # 关闭 fd 9，此后 inode 真正回收
```

---

### 4.16 性能视角：三层对象的开销分布

在实际系统中，文件操作的性能瓶颈往往不在你以为的地方。

| 操作 | 主要开销层 | 典型问题 |
|------|------------|----------|
| 频繁小文件创建 | inode 分配 + 目录项写入 + dcache 插入 | inode 耗尽；目录热点争用 |
| 频繁 path walk（命中 dcache）| dentry 哈希查找 + 权限检查 | 深路径慢；热目录锁争用 |
| 频繁 path walk（未命中 dcache）| 目录块 I/O + inode 读取 | drop_caches 后的"冷启动" |
| 大量 `stat()` | inode 读取（若在内存则快） | inode cache 压力 |
| rename() 热目录 | 两个父目录的 dentry 写锁 | 序列化成单线程 |
| deleted-but-open 堆积 | 无 I/O，但磁盘空间引用未释放 | `df` 显示满，`du` 显示少 |

**关键数字参考**：

- dcache 命中的 path walk 单步：~100ns（L1/L2 缓存范围内）
- dcache 未命中，需要读 ext4 目录块：~100μs（SSD）到 ~10ms（HDD）
- 每个 dentry 的内存占用：~192 字节（64 位系统，不含名字字符串）
- 100 万个 dentry 约占 ~200MB 内存
- `/proc/sys/fs/dentry-state` 的第一字段显示当前分配 dentry 总数

**dentry 内存参数调整**：

```bash
# 查看 inode/dentry 缓存统计
slabtop -s c | head -20  # 按缓存大小排序，可以看到 dentry、inode_cache 占用

# 查看 vfs_cache_pressure（默认 100，越小越倾向保留 dcache）
cat /proc/sys/vm/vfs_cache_pressure
# 调低可以减少 dcache 回收，提升 path walk 命中率
# 但内存压力下可能导致 OOM
```

---

## 例子：为什么 `mv a b` 常常不搬数据

在同一文件系统内，很多 `rename` / `mv` 本质上主要改的是目录项关系：

- 把 `a` 从某个父目录名字映射里拿掉
- 把 `b` 加到另一个或同一个父目录里
- inode 不变
- file 内容不必移动

所以同文件系统内 rename 常常比“复制后删除”轻得多。

---

## 常见误区

- 误以为 inode 保存文件名
- 误以为 dentry 就是磁盘目录项
- 误以为 negative dentry 不重要
- 误以为 rename 一定意味着内容搬动
- 误以为 file 和 inode 是同一层对象

---

## 本章小结

| 主题 | 结论 |
|------|------|
| inode | 解决“对象是什么” |
| 目录项 | 解决“这个父目录下这个名字指向谁” |
| dentry | 解决 VFS 路径解析与名字缓存 |
| negative dentry | 缓存“不存在”的结果 |
| file | 解决“打开后如何持续操作” |
| 对象图 | 路径语义来自多层对象协作，而不是单一文件概念 |

---

## 练习题

### 基础题

**4.1** 为什么 inode 不应该保存文件名？如果 inode 保存了文件名，会导致哪些语义问题（以硬链接和 rename 为例）？

**4.2** dentry 为什么不能被简单理解成”磁盘目录项的内存副本”？列出 dentry 在 VFS 路径解析中额外承担的职责。

### 中级题

**4.3** negative dentry 在哪些工作负载里尤其重要？解释为什么”缓存不存在”和”缓存存在”在性能上同样关键。

**4.4** 以 `rename(“a/x”, “b/y”)` 为例，分析这个操作分别对目录项、dentry、inode、open file 各层产生了什么影响，哪些层变了，哪些层没变。

### 提高题

**4.5** 设计一个测试程序，验证”同一 inode 可以有多个 dentry alias，且 rename 不影响已打开 fd 的可用性”：打开文件 `a`，将其 rename 为 `b`，继续通过原 fd 读写，用 `ls -li` 和 `stat /proc/<pid>/fd/<n>` 对比，解释各字段的变化含义。

---

## 练习答案

**4.1** 若 inode 保存文件名，硬链接（多个名字→同一对象）就需要存多个名字到同一 inode，冲突；rename 只改名字时就必须改 inode，破坏其作为”稳定对象标识”的语义。名字应归属于目录项（父目录上下文中的名字-inode 映射），inode 只表示对象本体。

**4.2** dentry 额外职责：①negative dentry（缓存不存在结果）；②与 mount 系统协作处理挂载穿越；③参与 rename/unlink 时的名字视图更新；④dcache LRU 管理与 revalidation 标记；⑤在 NFS/FUSE 场景下配合 `->d_revalidate` 确认有效性。

**4.3** 工作负载：动态链接器探测候选库路径、shell 对 `PATH` 中每个目录查找命令、Web 服务对每个 URL 尝试多个静态资源路径。”缓存不存在”避免重复下穿到底层文件系统查目录，在失败查找占多数的场景（如对不存在文件名频繁探测）中可降低几个数量级的延迟。

**4.4** 目录项：从 `a/` 的名字映射表中移除 `x`，在 `b/` 的名字映射表中增加 `y → inode_N`；dentry：原 dentry（`a/x`）无效化，新 dentry（`b/y`）建立；inode：inode_N 不变（元数据、数据块、引用计数不变）；open file：已打开的 fd 仍然有效，因为 file → inode 链路未断。

**4.5** 参考代码框架：`fd = open(“a”, O_RDWR); rename(“a”, “b”); write(fd, ...); read(fd, ...)`。预期观察：`ls -li b` 的 inode 号与打开前 `ls -li a` 相同；`/proc/<pid>/fd/<n>` 符号链接显示 `b (deleted)` 或直接为 `b`（取决于内核版本和 dentry 状态）；fd 继续可读写，验证 file → inode 链路独立于名字。

---

## 延伸阅读

1. **Bovet & Cesati**. *Understanding the Linux Kernel*, Ch.12 — VFS 对象模型（inode、dentry、file 详解）
2. **Linux 内核文档**. `Documentation/filesystems/vfs.rst` — VFS 接口官方参考
3. **LWN.net**. *The VFS layer* — dcache 和 dentry 生命周期系列文章
4. **Kerrisk**. *The Linux Programming Interface*, Ch.14 — 文件系统挂载与 VFS

---

[← 上一章：权限、元数据与链接](../part1-basics/03-permissions-metadata-and-links.md)

[下一章：文件描述符与 open file →](./05-file-descriptors-and-open-files.md)

[返回目录](../README.md)
