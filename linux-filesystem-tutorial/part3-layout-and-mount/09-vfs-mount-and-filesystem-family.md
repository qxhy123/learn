# 第9章：VFS、挂载与文件系统家族

> **本章导读**：VFS 的价值不是把所有文件系统变成同一种东西，而是让它们在统一接口下暴露不同语义；挂载的价值不是”显示目录”，而是把多个文件系统实例拼成一个进程可见的名字空间。

**前置知识**：第6章（路径到 inode 查找过程）、第7章（块与超级块）
**预计学习时间**：55 分钟

---

## 学习目标

完成本章后，你将能够：

1. 解释 VFS 的核心对象：superblock、inode、dentry、file、mount/path
2. 区分 file system type、filesystem instance、mount point、mount namespace
3. 理解挂载穿越、bind mount、传播语义和 id 视图为什么会影响路径结果
4. 认识 ext4、xfs、btrfs、zfs、tmpfs、procfs、NFS、FUSE 等家族的工程定位
5. 说明“统一接口”为什么不等于“统一语义”
6. 把“新盘初始化、挂载、`/etc/fstab` 持久化”放回 mount 与 filesystem instance 语义里理解

---

## 正文内容

### 9.1 VFS 统一的是接口，不是语义

VFS（Virtual File System）给上层系统调用提供统一入口：

- `open`
- `read`
- `write`
- `stat`
- `readdir`
- `mmap`
- `rename`
- `link`
- `unlink`

但这些接口背后的文件系统可能完全不同：

- ext4：本地块设备文件系统
- tmpfs：内存文件系统
- procfs/sysfs：内核对象视图
- NFS：网络文件系统
- FUSE：用户态实现的文件系统
- overlayfs：联合挂载视图

所以 VFS 的真实含义是：**统一调用路径，但保留底层实现差异**。

### 9.2 VFS 的对象图

理解 VFS，必须理解这些对象不是同一层：

| 对象 | 直觉 | 典型问题 |
|------|------|----------|
| file_system_type | 某类文件系统驱动/类型 | “ext4 / nfs / tmpfs 这类实现是什么？” |
| superblock | 某个文件系统实例 | “这个挂载实例的全局状态是什么？” |
| inode | 某个对象本体 | “这个文件/目录/设备节点是什么？” |
| dentry | 某个名字解析关系 | “这个父目录下的名字指向谁？” |
| file | 某次打开后的运行时状态 | “这个 fd 的偏移、flag、方法上下文是什么？” |
| mount / vfsmount | 某个实例接入名字空间的位置 | “路径走到这里后进入哪个文件系统实例？” |
| path | dentry + mount 的组合 | “这个名字节点在哪个挂载视图里？” |

特别注意最后一点：同一个 dentry 如果不结合 mount 视图，就不足以表达完整路径语义。

### 9.3 `superblock` 与 `mount` 不是一回事

同一个文件系统实例可以被挂到不同位置；同一个挂载点也可能被后续挂载覆盖。要区分：

- **superblock**：文件系统实例的全局对象
- **mount**：把这个实例接到某个名字空间位置的关系
- **mount point**：路径树里发生视图切换的位置
- **mount namespace**：某组进程看到的挂载树

这就是为什么容器里和宿主机上相同的 `/proc`、`/sys`、`/app` 可能不是同一个视图。

### 9.4 挂载穿越如何改变路径解析

路径解析不是只沿目录树走，还会沿挂载树跳转。例如：

```text
/mnt/data/file
```

如果 `/mnt/data` 是挂载点，路径解析到这里后，后续查找进入的是另一个文件系统实例的根，而不是原目录下的普通子树。

这带来几个工程后果：

- bind mount 可以让同一对象树出现在多个路径
- mount 覆盖可以让原目录内容在视图里暂时不可见
- unmount 后原目录内容可能重新显露
- 容器 volume 可能覆盖镜像里已有路径

### 9.5 bind mount 为什么不是符号链接

bind mount 和软链接都能让你从一个路径到另一个内容，但它们层次不同：

| 机制 | 工作层次 | 关键差异 |
|------|----------|----------|
| 符号链接 | 路径解析中的特殊文件对象 | 内容是目标路径字符串 |
| bind mount | 挂载树层面的视图接入 | 把已有对象树接到新位置 |

bind mount 不需要目标是“路径字符串”；它改变的是挂载视图，而不是创建一个普通链接文件。

### 9.6 mount namespace 与传播语义

mount namespace 让不同进程看到不同挂载树。但如果只说“隔离挂载树”，还不够。还要理解 propagation：

- shared：挂载事件可向同组传播
- private：挂载事件不传播
- slave：接收上游传播但不向上游传播
- unbindable：不能被 bind mount

容器运行时要非常在意这些模式。否则可能出现：

- 宿主机挂载事件意外进入容器
- 容器里的挂载变化影响宿主或其他容器
- volume 行为和预期不一致

### 9.7 idmapped mounts 的直觉

更现代的 Linux 还支持 idmapped mounts 这类能力：同一底层文件在某个挂载视图中可以呈现不同的 uid/gid 映射。

你不必在入门阶段掌握细节，但要理解它代表一个趋势：挂载视图不只是“看到哪棵树”，还可能影响“以什么身份语义看到这棵树”。

这对容器、无特权环境和共享目录很重要。

### 9.8 文件系统家族：不是排行表，而是目标差异

不同文件系统优化目标不同：

| 家族 | 典型目标 | 常见关注点 |
|------|----------|------------|
| ext4 | 通用、成熟、兼容性好 | 稳定默认、日志恢复、工具链成熟 |
| xfs | 大文件、高并发、元数据扩展性 | 大规模数据、并行分配、在线扩展 |
| btrfs | CoW、子卷、快照、校验、send/receive | 管理能力强，复杂度和写放大也要考虑 |
| zfs | 端到端校验、池化、快照、复制、自修复 | 管理模型强，但生态/许可/内核集成方式要注意 |
| tmpfs | 内存文件系统 | 快但非持久，受内存和 swap 影响 |
| procfs/sysfs | 内核对象接口 | 不是磁盘文件，读写可能触发内核行为 |
| NFS | 网络文件接口 | 延迟、缓存、一致性、锁语义 |
| FUSE | 用户态文件系统 | 灵活，但多上下文切换和缓存边界复杂 |

高阶判断不是“谁更高级”，而是“谁的语义和代价匹配我的负载”。

### 9.9 FUSE 的价值与代价

FUSE（Filesystem in Userspace）允许把文件系统逻辑放到用户态进程中。这很适合：

- 对象存储网关
- 加密/解密层
- 归档/压缩视图
- 原型文件系统
- 用户态权限和策略实验

但它的代价也明显：

- 系统调用路径更长
- 用户态守护进程成为关键路径
- page cache 策略和一致性更难理解
- 高 IOPS、小文件、低延迟场景可能明显吃亏

所以 FUSE 是“灵活性换成本”，不是“免费自定义内核文件系统”。

### 9.10 伪文件系统为什么会打破直觉

`procfs`、`sysfs`、`debugfs` 这类文件系统经常看起来像普通目录，但它们背后不是普通磁盘块。

例如：

- 读取某个文件可能触发内核生成内容
- 写某个文件可能修改内核参数
- 文件大小、时间戳、权限语义可能和普通文件不同

这就是“万物皆文件”最容易被误解的地方：统一接口不等于统一存储模型。

### 9.11 VFS 操作表：接口如何落到底层

VFS 不是只保存对象，还把操作分发给底层文件系统实现。可以粗略理解为：

- inode operations：创建、链接、重命名等对象级操作
- file operations：读写、mmap、fsync 等打开文件后的操作
- dentry operations：名字缓存、比较、失效等路径相关操作
- super operations：挂载实例、同步、回收等全局操作

这解释了为什么同一个系统调用在不同文件系统上可能有不同成本和边界。

### 9.12 一个工程判断框架

当你看到一个路径问题时，先问：

1. 这是哪个 mount namespace 里的路径？
2. 路径经过了哪些挂载点？
3. 目标是普通本地文件系统，还是伪文件系统 / FUSE / NFS / overlayfs？
4. 相同路径在宿主、容器、另一个进程里是否指向同一 path？
5. 这个文件系统是否真的支持你依赖的语义？

很多容器、远端存储和权限问题，本质上都是这五问没有问清楚。

### 9.13 从裸盘到挂载点：初始化实操链路其实是在走哪几层

很多人第一次“初始化一块新盘”时，做的动作大致是：

1. 看见一个新块设备
2. 创建分区
3. `mkfs`
4. 挂载到某个目录
5. 写入 `/etc/fstab`

这条链看起来像一串命令，其实每一步都在改变不同层：

| 动作 | 改变的是哪一层 |
|------|----------------|
| 分区 | 块设备上的可寻址区域划分 |
| `mkfs.ext4` / `mkfs.xfs` | 在某个设备或分区上创建文件系统实例 |
| `mount` | 把该实例接入当前名字空间 |
| 写 `fstab` | 描述系统启动时如何重建这条挂载关系 |

只要这四层不分清，就很容易把：

- “已经有分区”
- “已经有文件系统”
- “已经挂载成功”
- “重启后还能自动出现”

误当成同一件事。

### 9.14 `/etc/fstab` 为什么属于”重建挂载关系的协议”

`/etc/fstab` 常被误解成”开机时顺手执行几条 mount 命令的地方”。更准确的理解是：

- 它描述的是系统如何在启动过程中重建一组挂载关系
- 它最好引用稳定标识，例如 UUID 或 LABEL，而不是易漂移的 `/dev/sdX`
- 它和 systemd mount 单元、启动依赖、失败恢复一起决定”路径树最终长什么样”

这也是为什么改 `fstab` 时，真正该检查的不只是语法，而是：

- 目标 UUID/LABEL 是否真对应预期文件系统实例
- 挂载点目录是否已经准备好
- 挂载失败时系统应该阻塞、降级，还是允许 `nofail`

如果你更关心实操路径，可以结合[附录 A5：分区、mkfs、挂载与动态扩容实操](../appendix/practical-mount-partition-and-resize.md)看完整流程。

---

### 9.15 VFS 核心内核结构：super_block、vfsmount 与 mount

VFS 的对象模型不只是概念层——它在内核中对应具体的 C 结构体。理解这些结构体让你能把 `/proc/mounts`、`findmnt`、`strace mount` 的输出与内核行为对应起来。

#### struct super_block（`include/linux/fs.h`）

`struct super_block` 代表一个文件系统实例，挂载时创建，卸载时销毁。

```c
struct super_block {
    struct list_head    s_list;         /* 全局 super_block 链表（sb_lock 保护）*/
    dev_t               s_dev;          /* 所在块设备号（tmpfs/procfs 为虚拟设备号）*/
    unsigned char       s_blocksize_bits;
    unsigned long       s_blocksize;    /* 文件系统块大小（字节）*/
    loff_t              s_maxbytes;     /* 最大文件大小 */
    struct file_system_type *s_type;    /* 指向 file_system_type（ext4/xfs/...）*/
    const struct super_operations *s_op; /* 超级块操作表 */
    const struct dquot_operations *dq_op;
    const struct quotactl_ops *s_qcop;
    const struct export_operations *s_export_op;

    unsigned long       s_flags;        /* SB_RDONLY、SB_NOSUID、SB_NOEXEC 等 */
    unsigned long       s_iflags;       /* SB_I_CGROUPWB 等内部标志 */
    unsigned long       s_magic;        /* 文件系统魔数（ext4: 0xEF53, xfs: 0x58465342）*/

    struct dentry       *s_root;        /* 此文件系统实例的根 dentry */
    struct rw_semaphore s_umount;       /* 保护 umount 与 mount 并发 */
    int                 s_count;        /* 引用计数 */
    atomic_t            s_active;       /* 活跃引用数 */

    const struct xattr_handler **s_xattr;
    const struct fscrypt_operations *s_cop;

    struct hlist_bl_head s_roots;       /* bind mount 共享的根 dentry（anon bind mount）*/
    struct list_head    s_mounts;       /* 此 superblock 上的所有 mount 列表 */
    struct block_device *s_bdev;        /* 关联块设备（内存文件系统为 NULL）*/

    struct backing_dev_info *s_bdi;     /* 回写控制器（脏页限速等）*/
    struct mtd_info     *s_mtd;         /* MTD 设备（Flash）*/
    struct hlist_node   s_instances;    /* 同 file_system_type 的所有 sb 链表 */

    unsigned int        s_max_links;    /* 硬链接计数上限（ext4: 65000）*/

    struct mutex        s_vfs_rename_mutex; /* 跨目录 rename 的全局序列化锁 */
    const char          *s_subtype;     /* 子类型字符串（如 FUSE 子类型）*/
    const struct dentry_operations *s_d_op; /* 默认 dentry 操作表 */

    struct shrinker     s_shrink;       /* 内存压力下回收 dentry/inode 的 shrinker */
    atomic_long_t       s_remove_count; /* 未提交移除计数 */
    int                 s_readonly_remount;

    struct workqueue_struct *s_dio_done_wq; /* Direct I/O 完成工作队列 */
    struct list_head    s_inodes;       /* 此 superblock 的所有内存 inode */
    struct list_head    s_inodes_wb;    /* 待回写的 inode 链表 */
    spinlock_t          s_inode_list_lock;

    void                *s_fs_info;     /* 文件系统私有数据（ext4: ext4_sb_info）*/
    u32                 s_time_gran;    /* 时间戳精度（纳秒）*/
    char                s_id[32];       /* 文件系统实例标识字符串（如 “sda1”）*/
    uuid_t              s_uuid;         /* 文件系统 UUID */
};
```

关键关系说明：

- `s_root`：每个 superblock 有一个根 dentry，`mount()` 把它接入目标路径。
- `s_mounts`：一个 superblock 可对应多个 mount（bind mount 场景）。
- `s_fs_info`：指向文件系统私有数据，ext4 用 `container_of()` 从中取 `ext4_sb_info`（含 journal 句柄、特性位等）。
- `s_shrink`：内存压力时，内核通过此 shrinker 回收属于本 superblock 的 dentry 和 inode 缓存。

---

#### struct vfsmount 与 struct mount（`fs/mount.h`）

历史上 `struct vfsmount` 承担了”挂载实例”的全部信息。Linux 3.x 之后，内部重构成两层：

- `struct vfsmount`：公开接口部分，供路径解析使用
- `struct mount`：完整内部实现，通过 `container_of()` 从 `vfsmount` 中取

```c
/* 公开接口（路径解析可见）*/
struct vfsmount {
    struct dentry   *mnt_root;      /* 此 mount 的根 dentry（在被挂载文件系统的根）*/
    struct super_block *mnt_sb;     /* 指向文件系统实例的 superblock */
    int             mnt_flags;      /* MNT_NOSUID、MNT_NOEXEC、MNT_READONLY 等 */
    struct user_namespace *mnt_userns; /* idmapped mount 用的用户命名空间 */
};

/* 完整实现（fs/mount.h，内核内部可见）*/
struct mount {
    struct hlist_node mnt_hash;             /* mount 哈希表节点 */
    struct mount    *mnt_parent;            /* 父 mount（路径解析时的上游 mount）*/
    struct dentry   *mnt_mountpoint;        /* 挂载点 dentry（在父文件系统中的位置）*/
    struct vfsmount mnt;                    /* 公开接口（内嵌）*/
    union {
        struct rcu_head mnt_rcu;
        struct llist_node mnt_llist;
    };
    struct list_head mnt_mounts;            /* 子 mount 链表 */
    struct list_head mnt_child;             /* 在父 mount 的 mnt_mounts 中的节点 */
    struct list_head mnt_instance;          /* 同一 superblock 的所有 mount 链表节点 */
    const char      *mnt_devname;           /* 设备名字符串（如 “/dev/sda1”）*/
    struct list_head mnt_list;              /* 所属 mount namespace 的 mount 链表 */
    struct list_head mnt_expire;            /* autofs 懒卸载链表 */
    struct list_head mnt_share;             /* shared propagation peer group 链表 */
    struct list_head mnt_slave_list;        /* slave mount 链表 */
    struct list_head mnt_slave;             /* 在 master 的 slave 链表中的节点 */
    struct mount    *mnt_master;            /* slave 的 master mount */
    struct mnt_namespace *mnt_ns;           /* 所属 mount namespace */
    struct mountpoint *mnt_mp;              /* 指向挂载点对象 */
    union {
        struct hlist_node mnt_mp_list;
        struct hlist_node mnt_umount;
    };
    struct list_head mnt_umounting;
    struct fsnotify_mark_connector __rcu *mnt_fsnotify_marks;
    __u32           mnt_fsnotify_mask;
    int             mnt_id;                 /* mount ID（/proc/self/mountinfo 中的 ID）*/
    int             mnt_group_id;           /* propagation group ID（shared peer group）*/
    int             mnt_expiry_mark;        /* autofs 过期标记 */
    struct hlist_head mnt_pins;             /* pin 计数，防止 umount 竞争 */
    struct hlist_head mnt_stuck_children;
};
```

**path = vfsmount + dentry 组合**：

```c
struct path {
    struct vfsmount *mnt;   /* 当前处于哪个 mount 实例 */
    struct dentry   *dentry; /* 当前路径节点 */
};
```

路径解析时，`path.mnt` 和 `path.dentry` 一起变化。遇到挂载点时，内核通过 `__follow_mount()` 将 `mnt` 切换为子 mount 的 `vfsmount`，`dentry` 切换为子 mount 的 `mnt_root`。

**用工具观察 mount 结构**：

```bash
# /proc/self/mountinfo 格式（每字段含义）：
# mnt_id  parent_id  major:minor  root  mountpoint  opts  [optional]  -  fstype  source  super_opts
cat /proc/self/mountinfo | head -5
# 23 0 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw,errors=remount-ro
# ↑  ↑ ↑   ↑ ↑  ↑           ↑        ↑ ↑    ↑        ↑
# id par dev root mntpnt opts propagation  fstype source super_opts

# 字段解析：
# mnt_id=23：mount.mnt_id
# parent_id=0：mount.mnt_parent（0 表示根，无父）
# 8:1：major:minor 设备号
# root=/：此 mount 在源文件系统中的根路径
# mountpoint=/：在 mount namespace 中的挂载路径
# shared:1：propagation group ID（peer group 1）

# 找所有 bind mount（root 字段不是 /）
cat /proc/self/mountinfo | awk '$4 != “/”' | head -10

# 查看 propagation 类型
cat /proc/self/mountinfo | grep -oP '(shared|private|slave|unbindable):\d+' | sort | uniq -c

# 用 findmnt 以树形显示（更易读）
findmnt --list -o TARGET,SOURCE,FSTYPE,PROPAGATION,OPTIONS | head -20
findmnt -o TARGET,SOURCE,FSTYPE,MAJ:MIN,PROPAGATION

# 查看容器内的 mount namespace 与宿主机的差异
ls -la /proc/self/ns/mnt          # 当前进程的 mount namespace inode
ls -la /proc/1/ns/mnt             # PID 1（init）的 mount namespace
# 如果 inode 号相同 → 同一 mount namespace
# 如果不同 → 不同 mount namespace
```

---

### 9.16 VFS 操作表：四张分派表的字段与触发时机

VFS 通过四张函数指针表将系统调用分派给具体文件系统实现。理解这些操作表，就能理解为什么”同一个 read() 在 ext4 和 procfs 上会走完全不同的代码路径”。

#### struct super_operations（超级块操作表）

```c
struct super_operations {
    /* inode 生命周期 */
    struct inode *(*alloc_inode)(struct super_block *sb);
        /* 分配 inode（ext4 分配 ext4_inode_info + 内嵌 VFS inode）*/
    void (*destroy_inode)(struct inode *);
        /* 释放 inode 内存 */
    void (*free_inode)(struct inode *);
        /* RCU 释放回调 */

    /* inode 状态管理 */
    void (*dirty_inode)(struct inode *, int flags);
        /* inode 被标记为 dirty 时调用（ext4 触发 journal 记录）*/
    int  (*write_inode)(struct inode *, struct writeback_control *wbc);
        /* 将 inode 写回磁盘（由 writeback 机制调用）*/
    int  (*drop_inode)(struct inode *);
        /* 引用计数降到 0 时询问是否回收（generic_drop_inode 或文件系统自定义）*/
    void (*evict_inode)(struct inode *);
        /* inode 真正从内存移除时调用（ext4 处理 orphan inode 清理）*/
    void (*put_super)(struct super_block *);
        /* 文件系统卸载时释放私有资源 */

    /* 同步与统计 */
    int  (*sync_fs)(struct super_block *sb, int wait);
        /* fsync/syncfs 时将文件系统元数据同步到磁盘 */
    int  (*freeze_super)(struct super_block *);
        /* 冻结文件系统（快照前操作，停止写入）*/
    int  (*unfreeze_super)(struct super_block *);
    int  (*statfs)(struct dentry *, struct kstatfs *);
        /* 实现 statfs()/df 的数据来源（空间统计）*/

    /* 挂载相关 */
    int  (*remount_fs)(struct super_block *, int *, char *);
        /* mount -o remount 时调用（更新挂载选项）*/
    void (*umount_begin)(struct super_block *);
        /* 强制卸载开始时调用（NFS 用来中断挂起操作）*/

    /* inode 缓存压力 */
    int  (*show_options)(struct seq_file *, struct dentry *);
        /* /proc/mounts 和 /proc/self/mountinfo 的挂载选项字段来源 */
    long (*nr_cached_objects)(struct super_block *, struct shrink_control *);
    long (*free_cached_objects)(struct super_block *, struct shrink_control *);
};
```

**触发示例**：

```bash
# statfs() 触发 super_operations->statfs
df -h /
strace df -h / 2>&1 | grep statfs
# statfs(“/”, {f_type=EXT2_SUPER_MAGIC, f_bsize=4096, f_blocks=..., ...}) = 0

# sync() 触发所有已挂载文件系统的 super_operations->sync_fs
sync
strace sync 2>&1
# syncfs(1) = 0    ← 对 stdout 所在文件系统做同步

# 冻结文件系统（需要 root）
fsfreeze --freeze /mnt/data    # 触发 super_operations->freeze_super
fsfreeze --unfreeze /mnt/data  # 触发 super_operations->unfreeze_super
```

#### struct inode_operations（inode 对象操作表）

```c
struct inode_operations {
    /* 目录操作（需要目录 inode 实现）*/
    struct dentry * (*lookup)(struct inode *, struct dentry *, unsigned int);
        /* dcache 未命中时，在目录里查找名字（ext4_lookup 读目录块）*/
    const char *    (*get_link)(struct dentry *, struct inode *, struct delayed_call *);
        /* 读取符号链接目标（ext4_get_link 读 inline symlink 或数据块）*/
    int             (*permission)(struct user_namespace *, struct inode *, int);
        /* 权限检查（DAC 检查，SELinux hook 从这里挂入）*/
    struct posix_acl * (*get_acl)(struct inode *, int, bool);
    int             (*readlink)(struct dentry *, char __user *, int);
    int             (*create)(struct user_namespace *, struct inode *, struct dentry *,
                              umode_t, bool);
        /* open(O_CREAT) 或 creat() 触发（创建新普通文件）*/
    int             (*link)(struct dentry *, struct inode *, struct dentry *);
        /* link() 系统调用（创建硬链接）*/
    int             (*unlink)(struct inode *, struct dentry *);
        /* unlink() 系统调用（删除目录项，减少 nlink）*/
    int             (*symlink)(struct user_namespace *, struct inode *, struct dentry *,
                               const char *);
        /* symlink() 系统调用（创建符号链接）*/
    int             (*mkdir)(struct user_namespace *, struct inode *, struct dentry *,
                             umode_t);
        /* mkdir() 系统调用 */
    int             (*rmdir)(struct inode *, struct dentry *);
        /* rmdir() 系统调用（目录必须为空）*/
    int             (*mknod)(struct user_namespace *, struct inode *, struct dentry *,
                             umode_t, dev_t);
        /* mknod() 系统调用（创建设备节点/FIFO/socket 文件）*/
    int             (*rename)(struct user_namespace *, struct inode *, struct dentry *,
                              struct inode *, struct dentry *, unsigned int);
        /* rename() 系统调用（跨目录或同目录改名）*/
    int             (*setattr)(struct user_namespace *, struct dentry *, struct iattr *);
        /* chmod/chown/truncate 等属性修改 */
    int             (*getattr)(struct user_namespace *, const struct path *,
                               struct kstat *, u32, unsigned int);
        /* stat()/lstat() 的数据来源 */
    ssize_t         (*listxattr)(struct dentry *, char *, size_t);
    int             (*fiemap)(struct inode *, struct fiemap_extent_info *,
                              u64 start, u64 len);
        /* ioctl(FIEMAP)：获取文件的物理布局（filefrag 的数据来源）*/
};
```

#### struct file_operations（打开文件操作表）

```c
struct file_operations {
    struct module   *owner;
    loff_t          (*llseek)(struct file *, loff_t, int);
        /* lseek() 系统调用 */
    ssize_t         (*read)(struct file *, char __user *, size_t, loff_t *);
    ssize_t         (*write)(struct file *, const char __user *, size_t, loff_t *);
    ssize_t         (*read_iter)(struct kiocb *, struct iov_iter *);
        /* 新式 read 接口（支持 io_uring、sendfile、splice）*/
    ssize_t         (*write_iter)(struct kiocb *, struct iov_iter *);
    int             (*iopoll)(struct kiocb *kiocb, struct io_comp_batch *, unsigned int flags);
        /* io_uring polling 支持 */
    int             (*iterate_shared)(struct file *, struct dir_context *);
        /* getdents64() 的底层实现（readdir）*/
    __poll_t        (*poll)(struct file *, struct poll_table_struct *);
        /* select/poll/epoll 监听的实现 */
    long            (*unlocked_ioctl)(struct file *, unsigned int, unsigned long);
        /* ioctl() 系统调用 */
    long            (*compat_ioctl)(struct file *, unsigned int, unsigned long);
    int             (*mmap)(struct file *, struct vm_area_struct *);
        /* mmap() 系统调用（建立文件与虚拟内存区域的映射）*/
    unsigned long   mmap_supported_flags;
    int             (*open)(struct file *);
        /* open() 完成 path walk 后，对新 file 对象的初始化钩子 */
    int             (*flush)(struct file *, fl_owner_t id);
        /* close() 时调用（注意：不是 fsync！）*/
    int             (*release)(struct inode *, struct file *);
        /* 最后一个引用关闭时调用（file->f_count 降到 0）*/
    int             (*fsync)(struct file *, loff_t, loff_t, int datasync);
        /* fsync()/fdatasync() 系统调用 */
    int             (*fasync)(int, struct file *, int);
        /* SIGIO 异步通知注册 */
    int             (*lock)(struct file *, int, struct file_lock *);
        /* fcntl(F_SETLK) 锁定接口 */
    ssize_t         (*sendpage)(struct file *, struct page *, int, size_t, loff_t *, int);
    unsigned long   (*get_unmapped_area)(struct file *, unsigned long, unsigned long,
                                         unsigned long, unsigned long);
    int             (*flock)(struct file *, int, struct file_lock *);
        /* flock() 系统调用 */
    ssize_t         (*splice_write)(struct pipe_inode_info *, struct file *,
                                    loff_t *, size_t, unsigned int);
    ssize_t         (*splice_read)(struct file *, loff_t *,
                                   struct pipe_inode_info *, size_t, unsigned int);
    int             (*setlease)(struct file *, long, struct file_lock **, void **);
    long            (*fallocate)(struct file *, int mode, loff_t offset, loff_t len);
        /* fallocate() 系统调用（预分配/打洞）*/
    void            (*show_fdinfo)(struct seq_file *m, struct file *f);
        /* /proc/<pid>/fdinfo/<fd> 的额外字段来源 */
    ssize_t         (*copy_file_range)(struct file *, loff_t, struct file *,
                                        loff_t, size_t, unsigned int);
        /* copy_file_range() 系统调用（内核内数据复制）*/
    loff_t          (*remap_file_range)(struct file *file_in, loff_t pos_in,
                                         struct file *file_out, loff_t pos_out,
                                         loff_t len, unsigned int remap_flags);
        /* ioctl(FICLONERANGE) 等 reflink 操作 */
    int             (*fadvise)(struct file *, loff_t, loff_t, int);
        /* posix_fadvise() 系统调用（预读/缓存策略提示）*/
};
```

**观察操作表分派**：

```bash
# 用 strace 验证不同文件系统的同名系统调用行为差异
# tmpfs 的 fsync 是空操作
strace -e trace=fsync python3 -c “
import os
f = open('/tmp/test', 'w')   # tmpfs
f.write('hello')
f.flush()
os.fsync(f.fileno())         # 对 tmpfs 几乎无意义，但系统调用仍会被调用
“ 2>&1 | grep fsync
# fsync(3) = 0    ← 成功返回，但实际不做持久化（tmpfs 无持久存储）

# procfs 的 read 触发内核动态生成内容
strace -e trace=read cat /proc/meminfo 2>&1 | head -5
# read(3, “MemTotal:       ...\n...”, 65536) = N   ← 内核实时生成的字符串
```

---

### 9.17 mount namespace 内核实现与 propagation 工作原理

#### struct mnt_namespace（`fs/mount.h`）

```c
struct mnt_namespace {
    struct ns_common    ns;             /* 命名空间公共头（ioctl 和 /proc/ns/ 用）*/
    struct mount        *root;          /* 此 namespace 的根 mount */
    struct list_head    list;           /* 此 namespace 所有 mount 的链表 */
    spinlock_t          ns_lock;        /* 保护 list */
    struct user_namespace *user_ns;     /* 关联的用户命名空间 */
    struct ucounts      *ucounts;
    u64                 seq;            /* RCU 序列号（加速路径解析中的 mount 检查）*/
    wait_queue_head_t   poll;           /* 监听 mount 事件的等待队列 */
    u64                 event;          /* mount 事件计数（/proc/mounts 检测变化用）*/
    unsigned int        mounts;         /* mount 总数 */
    unsigned int        pending_mounts; /* 正在进行的 mount 操作数 */
};
```

每个进程通过 `task_struct->nsproxy->mnt_ns` 指向自己的 mount namespace。`clone(CLONE_NEWNS)` 或 `unshare(CLONE_NEWNS)` 创建新的 mount namespace（复制当前所有 mount）。

#### mount propagation 实现

propagation 通过 `mnt_share`/`mnt_slave_list`/`mnt_slave`/`mnt_master` 链表维护 peer group 和 master/slave 关系：

```
shared peer group（shared:1）：
  mount A ←→ mount B ←→ mount C  （双向 mnt_share 链表）
  任一成员发生 mount/umount 事件 → 遍历 peer group → 在所有成员中同步

slave 关系：
  master mount D → slave mount E → slave mount F
  D 有 mount 事件 → 通知 E 和 F → E 和 F 的新事件不传回 D

private：mnt_share 为空，mnt_master 为 NULL → 不传播
unbindable：在 private 基础上加标记，禁止作为 bind mount 的源
```

**查看 propagation**：

```bash
# 观察不同 propagation 模式的效果
# 准备：创建两个 mount namespace
unshare --mount bash   # 在新 mount namespace 中启动 bash

# 在原 namespace 挂载（shared 模式）
# 如果两个 namespace 共享 peer group，挂载事件会传播

# 用 findmnt 观察 propagation
findmnt -o TARGET,PROPAGATION /
# TARGET     PROPAGATION
# /          shared

# 使目录变为 private
mount --make-private /mnt/data
findmnt -o TARGET,PROPAGATION /mnt/data
# /mnt/data  private

# 查看 peer group（shared:N 中的 N 即 mnt_group_id）
cat /proc/self/mountinfo | awk '{print $5, $7}' | grep shared | head -10
# /              shared:1
# /sys           shared:7
# /proc          shared:15
```

#### unshare 与容器的 mount namespace 隔离

```bash
# 创建隔离的 mount namespace（类似容器隔离）
unshare --mount --propagation private bash

# 此时的 mount namespace 是原来的副本，但所有 mount 设为 private
# 在此 shell 内的挂载不会传播到原 namespace
mount -t tmpfs tmpfs /tmp/isolated
ls /tmp/isolated  # 在此 namespace 可见

# 退出后在原 shell 检查
exit
ls /tmp/isolated  # 不可见（mount namespace 独立，挂载不传播）

# 查看容器的 mount namespace（通过 nsenter 进入容器 namespace）
CONTAINER_PID=$(docker inspect -f '{{.State.Pid}}' my_container)
nsenter -t $CONTAINER_PID --mount findmnt | head -20
# 看到容器独立的挂载树（与宿主不同）
```

---

### 9.18 FUSE 请求路径：从系统调用到用户态 daemon 的完整生命周期

FUSE 的核心机制是通过 `/dev/fuse` 设备在内核与用户态 daemon 之间传递请求。

#### FUSE 通信机制

```
用户进程                    内核 FUSE 模块                FUSE daemon
    │                           │                            │
    │  read(“file”)             │                            │
    ├──── sys_read ────────────►│                            │
    │                           │ 1. 生成 FUSE_READ 请求     │
    │                           │    写入 fc->pending 队列   │
    │                           │                            │
    │                           │ 2. 唤醒 daemon             │
    │                           │───── wake_up ─────────────►│
    │                           │                            │
    │                           │                            │  read(/dev/fuse)
    │                           │◄──── fuse_dev_read ────────│ 3. 读取请求
    │                           │                            │
    │                           │                            │  处理请求
    │                           │                            │  (查自己的数据源)
    │                           │                            │
    │                           │ 4. daemon 写回结果          │
    │                           │◄──── fuse_dev_write ───────│  write(/dev/fuse)
    │                           │                            │
    │                           │ 5. 唤醒原始系统调用         │
    │◄─── sys_read 返回 ────────│                            │
```

#### FUSE 请求结构

内核通过 `/dev/fuse` 传递如下格式的请求（`struct fuse_in_header`）：

```
FUSE 请求包格式（内核→daemon）：
  fuse_in_header {
    uint32 len;      /* 整个请求长度（header + body）*/
    uint32 opcode;   /* 操作码：FUSE_LOOKUP=1, FUSE_GETATTR=3, FUSE_READ=15,
                                FUSE_WRITE=16, FUSE_READDIR=28, ...  */
    uint64 unique;   /* 请求唯一 ID（daemon 用响应中的 unique 与请求对应）*/
    uint64 nodeid;   /* 目标 inode 的 FUSE nodeid（文件系统内的 inode 号）*/
    uint32 uid;      /* 发起系统调用的进程 UID */
    uint32 gid;
    uint32 pid;
    uint32 padding;
  }
  + 操作特定 body（如 fuse_read_in: offset, size, read_flags 等）

FUSE 响应包格式（daemon→内核）：
  fuse_out_header {
    uint32 len;      /* 整个响应长度 */
    int32  error;    /* 错误码（0 表示成功，负数为 -errno）*/
    uint64 unique;   /* 对应请求的 unique ID */
  }
  + 操作特定 body
```

#### FUSE 缓存层与 direct_io 选项

FUSE 有三种 I/O 模式：

```
1. 普通（缓存）模式（默认）：
   read() → 先查 page cache → 命中则直接返回
   → miss → FUSE_READ 请求 → daemon 处理 → 结果填入 page cache → 返回

   问题：page cache 不知道 daemon 的数据是否已变化
   → 需要 daemon 在 FUSE_GETATTR 中汇报正确的文件大小，让内核失效旧缓存

2. direct_io 模式（绕过 page cache）：
   mount -o direct_io ...
   → 每次 read/write 直接走 FUSE 请求，不经过 page cache
   → 数据始终最新，但无法利用 page cache 加速，IOPS 更贵

3. writeback cache 模式（kernel >= 3.15, libfuse >= 3）：
   open(O_WRONLY) → write → 进 page cache（dirty）→ 内核异步回写触发 FUSE_WRITE
   → 聚合多个小写入为大请求，减少 daemon 调用次数
   → 代价：close() 之前不保证 daemon 看到最新数据
```

**观察 FUSE 请求**：

```bash
# 安装 libfuse 示例（FUSE 文件系统开发参考）
# 用 passthrough_fh（透传 FUSE）观察 /dev/fuse 流量

# 挂载 sshfs（典型 FUSE 文件系统）
sshfs user@host:/remote /mnt/remote -o sshfs_debug,loglevel=debug 2>&1 | head -30
# 可以看到 FUSE 操作序列：
# FUSE_INIT (握手协商版本和 capability)
# FUSE_STATFS
# FUSE_LOOKUP (路径查找)
# FUSE_GETATTR
# FUSE_OPENDIR / FUSE_READDIR
# FUSE_RELEASE

# 用 strace 观察 FUSE daemon 的 /dev/fuse 读写
PID=$(pgrep sshfs)
strace -p $PID -e trace=read,write -o /tmp/fuse_trace.txt &
ls /mnt/remote  # 触发 FUSE 操作
cat /tmp/fuse_trace.txt | head -20
# read(4, “\x98\0\0\0\x01\0\0\0...”, 131072)  ← FUSE_INIT 请求
# write(4, “\x10\0\0\0\0\0\0\0\x01\0\0\0...”, 16)  ← 响应
```

#### FUSE 的多上下文切换开销

```
单次 read() 调用链（FUSE vs ext4 对比）：

ext4：
  用户进程 → sys_read → page cache 命中 → 返回
  用户进程 → sys_read → page cache miss → ext4_readpage → bio 提交 → IRQ 回调 → 返回
  上下文切换次数：0（命中）或 2（miss，内核→块设备→内核，同一内核上下文）

FUSE（page cache miss）：
  用户进程 → sys_read（上下文切换1）
  → FUSE 内核模块写入请求队列
  → 用户进程阻塞
  → FUSE daemon 被唤醒 read(/dev/fuse)（上下文切换2）
  → daemon 处理（可能再做网络 I/O 或磁盘 I/O）
  → daemon write(/dev/fuse)（上下文切换3）
  → 原用户进程被唤醒（上下文切换4）
  上下文切换次数：4

latency 差异（典型值）：
  ext4 page cache 命中：~200ns
  ext4 page cache miss（SSD）：~100μs
  FUSE page cache 命中：~500ns（仍需经过 FUSE 内核模块的少量检查）
  FUSE page cache miss：~500μs-2ms（多了 daemon 往返）
```

---

### 9.19 实操工具：观察 mount namespace 和文件系统状态

**挂载树观察**：

```bash
# 完整挂载树（树形 + 详细选项）
findmnt --tree
findmnt -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL,USE%,OPTIONS --tree

# 只看特定文件系统类型
findmnt -t ext4
findmnt -t tmpfs

# 查看某个文件所在的挂载点
findmnt --target /var/log/app.log
# TARGET          SOURCE     FSTYPE OPTIONS
# /var            /dev/sda2  ext4   rw,relatime

# 查看所有 bind mount
cat /proc/self/mountinfo | awk '{print $4, $5}' | awk '$1 != “/”'
```

**文件系统实例信息**：

```bash
# 查看所有已挂载文件系统的 superblock 信息（通过 /proc/fs/<fstype>/<id>/info）
ls /proc/fs/ext4/
# 每个目录对应一个 ext4 实例，目录名是设备标识

# 查看 ext4 实例的统计（需要 debugfs 挂载）
cat /sys/kernel/debug/ext4/sda1/lifetime_write_kbytes  # 历史写入量
cat /sys/kernel/debug/ext4/sda1/mb_groups | head -5    # 块组状态

# 查看 superblock 上的 inode 缓存情况
cat /proc/slabinfo | grep -E “ext4_inode_cache|inode_cache|dentry”
# ext4_inode_cache 123456 123456  1024  32    8 : ...
# dentry           234567 234567   192  21    1 : ...
```

**跨 mount namespace 操作**：

```bash
# 查看所有进程的 mount namespace
for pid in /proc/[0-9]*/ns/mnt; do
    readlink “$pid” 2>/dev/null
done | sort | uniq -c | sort -rn | head -10
# 输出格式：引用次数 + mount namespace inode
# 引用次数最多的通常是主机 namespace

# 进入某个容器的 mount namespace
CONTAINER_PID=$(docker inspect -f '{{.State.Pid}}' mycontainer)
nsenter --target $CONTAINER_PID --mount -- findmnt

# 查看两个进程是否共享 mount namespace
ls -la /proc/$PID1/ns/mnt /proc/$PID2/ns/mnt
# mnt:[4026531840] 和 mnt:[4026531840] → 相同（同一 namespace）
# mnt:[4026531840] 和 mnt:[4026532196] → 不同（不同 namespace）
```

**文件系统健康检查**：

```bash
# 实时观察文件系统错误（不需要卸载）
dmesg -T | grep -i “ext4\|xfs\|filesystem\|I/O error” | tail -20

# 查看 ext4 错误计数
cat /sys/fs/ext4/sda1/errors_count 2>/dev/null || \
    tune2fs -l /dev/sda1 | grep “Filesystem errors”

# 检查挂载状态（clean/unclean）
tune2fs -l /dev/sda1 | grep “Filesystem state”
# Filesystem state: clean  ← 正常卸载
# Filesystem state: not clean  ← 上次未正常卸载，下次挂载会触发 journal replay

# 查看 xfs 的实时统计
xfs_stats /dev/sdb1  # 需要挂载 xfs 文件系统
```

---

## 本章小结

| 主题 | 结论 |
|------|------|
| VFS | 统一系统调用入口，但不消除底层语义差异 |
| 对象图 | superblock、inode、dentry、file、mount/path 是不同层 |
| 挂载 | 把文件系统实例接入名字空间，并影响路径解析 |
| mount namespace | 让不同进程看到不同挂载树 |
| propagation | 决定挂载事件是否跨视图传播 |
| FUSE | 用灵活性换更长路径和更复杂缓存语义 |
| 文件系统家族 | 没有绝对排行，只有目标和代价匹配 |
| 初始化与挂载 | 分区、`mkfs`、`mount`、`fstab` 分别改变不同层次 |

---

## 练习题

### 基础题

**9.1** 为什么 VFS 统一了系统调用接口，却不等于统一了语义？列举三个调用相同 API 但底层语义不同的文件系统场景。

**9.2** `superblock`、`mount`/`vfsmount`、`path`（dentry + mount）三者有什么区别？说明为什么”同一个 dentry 如果不结合 mount 视图，就不足以表达完整路径语义”。

### 中级题

**9.3** bind mount 和符号链接为什么不是同一层机制？描述 bind mount 后同一对象树出现在两个路径时，路径解析和挂载穿越如何工作。

**9.4** mount propagation（shared/private/slave/unbindable）如何影响容器隔离？举例说明配置错误的 propagation 模式会导致什么安全或隔离问题。

### 提高题

**9.5** 为什么”新盘已经分区”不等于”系统重启后自动在原路径出现”？从分区→mkfs→mount→fstab 四个动作出发，说明每步改变的是哪一层；并比较 FUSE 与内核原生文件系统在系统调用路径上的本质代价差异。

---

## 练习答案

**9.1** VFS 为 open/read/write/stat 等调用提供统一入口，但底层实现差异保留。场景：①`rename()` 在本地 ext4 上原子切换目录项，在 NFS 上语义和原子性依赖远端服务器实现，不同 NFS 服务器行为可能不同；②`fsync()` 在本地文件系统触发设备 flush，在 tmpfs 上本质是无操作，因为 tmpfs 数据本就在内存中；③`readdir()` 在普通目录返回稳定列表，在 procfs 上可能每次读取触发内核动态生成内容，”文件”不对应任何磁盘块。

**9.2** superblock 是文件系统实例的全局对象，记录块大小、inode 总数、特性位等；mount/vfsmount 是把该实例接入某个 mount namespace 路径位置的关系对象，同一 superblock 可对应多个 mount（如 bind mount）；path = dentry + vfsmount 组合，表示”这个名字节点在哪个挂载视图里”。仅凭 dentry 不足的原因：bind mount 将同一 dentry 树接到多个路径，仅有 dentry 无法区分当前路径处于哪个挂载实例的视图中，也无法在 mount namespace 之间正确区分路径归属。

**9.3** 符号链接是文件系统中的普通文件对象，内容为目标路径字符串，path walk 遇到后插入字符串替换步骤重新解析；bind mount 在挂载树层面将已有对象树接到新路径，不创建任何文件对象，改变的是 vfsmount 关系。路径解析行为：bind mount `/original` 到 `/new-path` 后，访问 `/new-path/foo` 时 path walk 到达 `/new-path` 发现挂载点，触发 `__follow_mount` 切换到 bind mount 对应的 vfsmount，后续在同一 dentry 树中查找 `foo`，与访问 `/original/foo` 经过相同的 dentry 和 inode。

**9.4** shared：挂载事件在 peer group 内双向传播，宿主机新挂载可能意外进入容器，或容器内挂载泄露到宿主；private：完全隔离，适合不需要与外部共享挂载事件的场景；slave：容器接收宿主传播但不向上传播，常见于需要宿主 volume 变化进入容器而容器隔离宿主的场景；unbindable：不能被 bind mount，防止意外共享。错误示例：容器内某目录被设为 shared，容器内的 volume mount 事件传播回宿主机 mount namespace，使容器内部挂载在宿主可见，导致隔离失效并可能暴露容器内路径。

**9.5** 四层区别：分区改变块设备可寻址区域（设备层）；mkfs 在分区上写入 superblock/inode 表等元数据，创建文件系统实例（文件系统层）；mount 把实例接入当前 mount namespace（内核名字空间层，重启后消失）；fstab 描述启动时如何重建挂载关系（系统启动协议层）。因此”有分区”≠”有文件系统”≠”已挂载”≠”重启后自动恢复”。FUSE 路径代价：内核 VFS 将请求写入 `/dev/fuse` 队列，用户态 FUSE daemon 读取并处理，处理完将结果写回内核，至少经历两次额外上下文切换（内核→用户态 daemon→内核），比内核原生文件系统多一次往返；daemon 崩溃导致挂载点挂起；page cache 与 FUSE 语义的一致性比本地文件系统复杂。

---

## 延伸阅读

1. **Linux 内核文档**. `Documentation/filesystems/vfs.rst` — VFS 对象接口完整参考
2. **man 8 mount** — 挂载选项、bind mount、propagation 类型（--make-shared 等）说明
3. **LWN.net**. *Mount namespaces and shared subtrees* (2016) — propagation 语义深度解析
4. **man 2 openat2** — `RESOLVE_NO_XDEV` 跨挂载点约束在路径安全中的应用
5. **libfuse 项目文档**. *Architecture overview* — FUSE 系统调用路径与缓存策略

---

[← 上一章：ext4 布局与日志机制](./08-ext4-layout-and-journaling.md)

[下一章：崩溃一致性与恢复 →](./10-crash-consistency-and-recovery.md)

[返回目录](../README.md)
