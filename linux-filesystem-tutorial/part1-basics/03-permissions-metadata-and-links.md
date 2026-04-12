# 第3章：权限、元数据与链接

> **本章导读**：在 Linux 文件系统里，很多关键语义并不来自文件内容本身，而来自元数据：权限、所有权、链接计数、时间戳、特殊位和引用关系共同决定了对象如何被看到、被访问、被回收。本章拆解 DAC 权限模型、时间戳含义、硬链接与软链接的本质区别，以及 deleted-but-open 现象背后的对象生命周期逻辑。

**前置知识**：第1章、第2章（路径与目录基础）
**预计学习时间**：45 分钟

---

## 学习目标

完成本章后，你将能够：

1. 理解传统 DAC 权限模型、`umask`、ACL、特殊权限位的层次关系
2. 区分 atime / mtime / ctime 以及它们各自代表什么变化
3. 说明硬链接、软链接、打开引用和目录项删除之间的关系
4. 解释 deleted-but-open、跨文件系统链接失败、目录 sticky bit 等真实现象
5. 建立“权限、元数据、名字、对象生命周期”是四层不同问题的直觉

---

## 正文内容

### 3.1 权限模型首先是 DAC

Linux 常见基础权限模型是 DAC（discretionary access control），由三组 `rwx` 组成：

- owner
- group
- other

但这只是第一层。真正的访问结果还会受到：

- 当前进程有效 uid/gid
- supplementary groups
- `umask`
- ACL
- capability
- mount 选项
- MAC（如 SELinux / AppArmor）

影响。所以“看 `ls -l` 就知道能不能访问”往往只在最简单场景里成立。

### 3.2 普通文件和目录的权限语义不同

对普通文件：

- `r` 更接近读取内容
- `w` 更接近写入内容
- `x` 更接近把它当程序执行

对目录：

- `r` 允许列目录项
- `w` 允许修改目录项关系（通常还需 `x` 配合）
- `x` 允许穿越/遍历这个目录

因此目录的 `x` 权限往往比普通文件更容易被误解。很多“路径存在但访问失败”的问题，根因并不在目标文件，而在中间目录缺少遍历权限。

### 3.3 `umask` 不是“权限例外”，而是创建协议的一部分

创建文件时，程序通常提供一个模式意图，再被 `umask` 裁剪。也就是说，创建权限不是“程序决定完就结束”，而是：

```text
+ 程序默认模式
- umask 裁剪
= 最终初始权限
```

这意味着：

- 同一程序在不同环境下创建文件权限可能不同
- 部署环境中 `umask` 配置会改变应用行为
- “默认权限太宽/太窄”很多时候不是业务代码 bug，而是运行环境问题

### 3.4 元数据不只是“附属信息”

一个文件对象常见元数据包括：

- inode 编号
- 文件类型
- 权限与所有权
- 大小
- 链接计数
- atime / mtime / ctime
- extent / block 映射入口
- 扩展属性、ACL、label 等附加信息

这些元数据并不比内容“次要”。很多运维和安全问题，本质上就是元数据问题而不是内容问题。

### 3.5 atime、mtime、ctime 经常被搞反

- **atime**：最近访问时间
- **mtime**：内容最近修改时间
- **ctime**：inode 状态最近变化时间

关键点是：`ctime` 不是“创建时间”，而是 inode 状态变化时间。权限变更、链接计数变化、所有者变化，哪怕内容没变，也会更新 `ctime`。

这也是为什么依赖时间戳做同步或缓存判断时，必须明确自己到底依赖的是哪种变化。

### 3.6 硬链接为什么本质上是“加名字”

硬链接的本质不是复制内容，而是多一个目录项指向同一个 inode：

- 内容共享
- 多数元数据共享
- 链接计数增加
- 删除一个名字不等于对象消失

这解释了为什么：

- 硬链接通常不能跨文件系统
- `stat` 能看到相同 inode 编号
- 某个名字删掉后，另一个名字仍然完全可用

### 3.7 软链接为什么是“路径对象”而不是“目标对象”

软链接是一个独立对象，里面保存的是目标路径字符串。它不是目标 inode 本身。

所以软链接：

- 可以跨文件系统
- 目标不存在时会悬空
- 路径解析时需要额外跟随
- 可能引入 symlink race 和安全问题

这也是为什么硬链接和软链接虽然都叫“链接”，但处在完全不同层次。

### 3.8 deleted-but-open 说明了什么

当你执行 `unlink` 时，删掉的是目录项关系，而不是“强制销毁所有对象引用”。只要：

- inode 仍被 open file description 引用
- 或仍有其他目录项/硬链接引用

对象就还活着。

所以 deleted-but-open 不是“奇怪特例”，而是 Linux 对“名字”和“对象生命周期”分层的正常结果。

### 3.9 特殊权限位和 sticky bit 的真实用途

除了普通 `rwx`，你还会遇到：

- setuid
- setgid
- sticky bit

特别是 sticky bit 在共享目录（如 `/tmp`）中很重要：即使大家都能写入目录，也不意味着每个人都能删除别人创建的目录项。它让“共享写入”和“随意删除别人文件”这两件事被拆开。

### 3.10 ACL、capability、SELinux 为什么说明 9 位权限不够

传统 `rwx` 很简洁，但很多现代场景需要更细粒度控制：

- ACL：对特定用户或组赋更细规则
- capability：把部分 root 特权拆分成独立能力
- SELinux/AppArmor：增加 MAC 层的强制策略

所以工程里看到“权限看着没问题但还是失败”，并不稀奇。`ls -l` 只能展示 DAC 的一部分，不是完整安全模型。

### 3.11 一个排障框架：权限问题究竟属于哪层

当你遇到权限问题，可以按层次问：

1. 路径中间目录是否可遍历？
2. DAC 的 owner/group/other 是否允许？
3. `umask` 是否导致创建结果异常？
4. ACL、capability、SELinux/AppArmor 是否拦截？
5. 这是路径问题、对象问题，还是挂载/命名空间问题？

只盯 `chmod 777` 往往是最糟糕的调试方式。

---

### 3.12 inode 权限字段的完整位图

`struct inode` 的 `i_mode` 字段是 16 位无符号整数，同时编码了**文件类型**和**权限**：

```c
/* include/uapi/linux/stat.h */

/* 文件类型掩码（i_mode 的高 4 位）*/
#define S_IFMT   0170000   /* 文件类型掩码 */
#define S_IFSOCK 0140000   /* socket */
#define S_IFLNK  0120000   /* 符号链接 */
#define S_IFREG  0100000   /* 普通文件 */
#define S_IFBLK  0060000   /* 块设备 */
#define S_IFDIR  0040000   /* 目录 */
#define S_IFCHR  0020000   /* 字符设备 */
#define S_IFIFO  0010000   /* FIFO（命名管道）*/

/* 特殊权限位 */
#define S_ISUID  0004000   /* setuid：执行时以文件所有者身份运行 */
#define S_ISGID  0002000   /* setgid：执行时以文件所属组身份运行 */
#define S_ISVTX  0001000   /* sticky bit：只有 owner 能删除目录内文件 */

/* 权限位（9 位）*/
#define S_IRWXU  00700     /* owner: r-x */
#define S_IRUSR  00400     /* owner: r-- */
#define S_IWUSR  00200     /* owner: -w- */
#define S_IXUSR  00100     /* owner: --x */

#define S_IRWXG  00070     /* group: r-x */
#define S_IRGRP  00040     /* group: r-- */
#define S_IWGRP  00020     /* group: -w- */
#define S_IXGRP  00010     /* group: --x */

#define S_IRWXO  00007     /* other: r-x */
#define S_IROTH  00004     /* other: r-- */
#define S_IWOTH  00002     /* other: -w- */
#define S_IXOTH  00001     /* other: --x */

/* 解读示例：
   ls -l 显示 -rwsr-xr-x
   i_mode = 0100755 | S_ISUID = 0104755

   二进制位布局（共 16 位）：
   [ 1000 | 0 | 100 | 111 | 101 | 101 ]
     S_IFREG  SUID  owner grp  oth
              (s)   rwx   r-x  r-x
*/
```

**内核权限检查的实际路径**：

```c
/* fs/namei.c — generic_permission() 是核心函数 */
int generic_permission(struct user_namespace *mnt_userns,
                       struct inode *inode, int mask)
{
    /* 步骤 1：检查进程 uid 是否是文件 owner */
    if (uid_eq(current_fsuid(), inode->i_uid)) {
        /* 用 owner 权限位（rwx @ bits 8-6）*/
        mode = (inode->i_mode >> 6) & 0007;
    }
    /* 步骤 2：检查进程的 group（含 supplementary groups）*/
    else if (in_group_p(inode->i_gid)) {
        /* 用 group 权限位（rwx @ bits 5-3）*/
        mode = (inode->i_mode >> 3) & 0007;
    }
    /* 步骤 3：用 other 权限位 */
    else {
        mode = inode->i_mode & 0007;
    }

    /* 检查所需权限是否被允许 */
    if ((mask & ~mode & (MAY_READ | MAY_WRITE | MAY_EXEC)) == 0)
        return 0;  /* 允许 */

    /* 特殊情况：root（CAP_DAC_OVERRIDE）可以绕过 rwx 检查 */
    if (capable(CAP_DAC_OVERRIDE)) {
        /* 即使没有 r/w 权限，也允许（除非目标是执行且没有任何 x 位）*/
        if (!(mask & MAY_EXEC) || (inode->i_mode & S_IXUGO))
            return 0;
    }

    return -EACCES;
}
```

```bash
# 实际观察权限检查过程
# 用 strace 追踪 faccessat 调用
strace -e trace=faccessat2,openat,newfstatat \
    bash -c 'cat /etc/shadow 2>&1; echo "exit=$?"'

# 输出：
# newfstatat(AT_FDCWD, "/etc/shadow", ...) = 0
# faccessat2(AT_FDCWD, "/etc/shadow", R_OK, AT_EACCESS) = -1 EACCES
# cat: /etc/shadow: Permission denied

# 用 getfattr 查看 inode 上的 ACL xattr（如果存在）
getfattr -n system.posix_acl_access /etc/
# 无输出说明没有 ACL；有 ACL 时会显示二进制 xattr 内容
```

### 3.13 POSIX ACL 的内核实现：xattr 之上的权限扩展

ACL 使用扩展属性存储，内核通过 `struct posix_acl` 在内存中表示：

```c
/* include/linux/posix_acl.h */
struct posix_acl_entry {
    short           e_tag;      /* ACL_USER_OBJ, ACL_GROUP, ACL_MASK 等 */
    unsigned short  e_perm;     /* ACL_READ | ACL_WRITE | ACL_EXECUTE */
    union {
        kuid_t      e_uid;      /* ACL_USER 类型时的 uid */
        kgid_t      e_gid;      /* ACL_GROUP 类型时的 gid */
    };
};

struct posix_acl {
    refcount_t          a_refcount; /* 引用计数 */
    unsigned int        a_count;    /* entry 数量 */
    struct posix_acl_entry a_entries[]; /* 变长 entry 数组 */
};

/* e_tag 的可能值 */
#define ACL_USER_OBJ    0x01    /* 文件所有者（对应传统 owner 权限）*/
#define ACL_USER        0x02    /* 特定 uid 的权限 */
#define ACL_GROUP_OBJ   0x04    /* 文件所属组（对应传统 group 权限）*/
#define ACL_GROUP       0x08    /* 特定 gid 的权限 */
#define ACL_MASK        0x10    /* 有效权限掩码（限制 USER/GROUP/GROUP_OBJ 权限上限）*/
#define ACL_OTHER       0x20    /* 其他用户（对应传统 other 权限）*/
```

**磁盘上的 ACL 格式（xattr `system.posix_acl_access`）**：

```bash
# 创建带 ACL 的文件并观察
touch /tmp/acl_demo.txt
setfacl -m u:1000:rw,g:2000:r,m::rw /tmp/acl_demo.txt

# 查看 ACL
getfacl /tmp/acl_demo.txt
# # file: acl_demo.txt
# # owner: root
# # group: root
# user::rw-          ← ACL_USER_OBJ  (传统 owner 权限)
# user:1000:rw-      ← ACL_USER      (uid=1000 的特殊权限)
# group::r--         ← ACL_GROUP_OBJ (传统 group 权限)
# group:2000:r--     ← ACL_GROUP     (gid=2000 的特殊权限)
# mask::rw-          ← ACL_MASK      (有效权限上限)
# other::r--         ← ACL_OTHER     (传统 other 权限)

# 查看 ACL 的原始 xattr 二进制内容
python3 - << 'EOF'
import os, struct

path = "/tmp/acl_demo.txt"
try:
    raw = os.getxattr(path, "system.posix_acl_access")
    print(f"ACL xattr 长度: {len(raw)} 字节")

    # ACL xattr 格式: 4字节版本头 + N×8字节 entry
    version = struct.unpack_from("<I", raw, 0)[0]
    print(f"版本: {version}")  # 应该是 2

    tag_names = {
        0x01: "USER_OBJ", 0x02: "USER", 0x04: "GROUP_OBJ",
        0x08: "GROUP", 0x10: "MASK", 0x20: "OTHER"
    }

    offset = 4
    while offset < len(raw):
        tag, perm, uid = struct.unpack_from("<HHI", raw, offset)
        name = tag_names.get(tag, f"?{tag:#x}")
        perm_str = ("r" if perm & 4 else "-") + \
                   ("w" if perm & 2 else "-") + \
                   ("x" if perm & 1 else "-")
        if tag in (0x02, 0x08):  # USER 或 GROUP
            print(f"  {name:12s} id={uid:6d} perm={perm_str}")
        else:
            print(f"  {name:12s}          perm={perm_str}")
        offset += 8
except OSError as e:
    print(f"无 ACL xattr: {e}")
EOF
```

**ACL 权限检查算法**：

```
ACL 检查顺序（posix_acl_permission()）:

1. 如果 fsuid == owner → 用 ACL_USER_OBJ entry 的权限
2. 遍历 ACL_USER entries：
     如果 entry.uid == fsuid → 用该 entry 权限 AND ACL_MASK
3. 如果 in_group(group_obj) → 记为"组匹配"
4. 遍历 ACL_GROUP entries：
     如果 in_group(entry.gid) → 记为"组匹配"
5. 如果有任何"组匹配"：
     取最高权限那条 AND ACL_MASK 判断
6. 使用 ACL_OTHER entry 权限

关键：ACL_MASK 限制了 USER/GROUP/GROUP_OBJ 的最大权限上限
     （但不限制 USER_OBJ 和 OTHER）
```

### 3.14 Linux capability 系统：比 root 更精细的权限粒度

`capability` 把"root 特权"拆分成 41 个独立能力（Linux 5.x），每个进程有三组 capability 集合：

```c
/* include/uapi/linux/capability.h */
/* 常用 capability（共约 41 个）*/
#define CAP_CHOWN         0   /* 改变文件 uid/gid（忽略所有权检查）*/
#define CAP_DAC_OVERRIDE  1   /* 忽略文件 DAC（rwx）权限检查 */
#define CAP_DAC_READ_SEARCH 2 /* 忽略目录 r/x 检查，忽略文件 r 检查 */
#define CAP_FOWNER        3   /* 文件所有者特权（忽略 inode 所有权检查）*/
#define CAP_FSETID        4   /* 允许设置 setuid/setgid 位 */
#define CAP_KILL          5   /* 绕过发送信号的权限检查 */
#define CAP_SETGID        6   /* 任意修改进程 gid */
#define CAP_SETUID        7   /* 任意修改进程 uid */
#define CAP_NET_BIND_SERVICE 10 /* 绑定端口 < 1024 */
#define CAP_SYS_ADMIN    21   /* 大量系统管理操作（mount, swapon 等）*/
#define CAP_SYS_RAWIO    17   /* iopl(), /dev/mem, /dev/kmem */
#define CAP_SYS_PTRACE   19   /* ptrace 任意进程 */
#define CAP_MKNOD        27   /* 创建设备节点 */
#define CAP_AUDIT_WRITE  29   /* 写 audit 日志 */
#define CAP_SETFCAP      31   /* 设置文件 capability */
```

```bash
# 查看当前进程的 capability
cat /proc/self/status | grep Cap
# CapInh: 0000000000000000   ← Inheritable: 可被子进程继承的 cap
# CapPrm: 0000000000000000   ← Permitted:   被允许获得的 cap
# CapEff: 0000000000000000   ← Effective:   当前有效的 cap
# CapBnd: 000001ffffffffff   ← Bounding:    硬上限（exec 后不能超过此集合）
# CapAmb: 0000000000000000   ← Ambient:     exec 跨越时继承的 cap

# 解码 capability 位图
python3 - << 'EOF'
cap_names = {
    0: "CHOWN", 1: "DAC_OVERRIDE", 2: "DAC_READ_SEARCH",
    3: "FOWNER", 4: "FSETID", 5: "KILL", 6: "SETGID",
    7: "SETUID", 10: "NET_BIND_SERVICE", 12: "NET_ADMIN",
    17: "SYS_RAWIO", 19: "SYS_PTRACE", 21: "SYS_ADMIN",
    27: "MKNOD", 31: "SETFCAP", 38: "PERFMON", 39: "BPF",
}

with open("/proc/self/status") as f:
    for line in f:
        if line.startswith("Cap"):
            name, val = line.strip().split(":\t")
            caps = int(val, 16)
            active = [cap_names.get(i, f"CAP_{i}")
                      for i in range(64) if caps & (1 << i)]
            print(f"{name}: {val.strip()}")
            if active:
                print(f"  → {', '.join(active)}")
EOF

# 给可执行文件设置文件级 capability（不需要 setuid root）
# 示例：给 ping 设置 NET_RAW capability（现代 Linux 已默认这么做）
getcap /usr/bin/ping
# /usr/bin/ping cap_net_raw=ep   ← e=effective, p=permitted
setcap cap_net_raw+ep /usr/bin/ping  # 需要 root 或 CAP_SETFCAP

# 查看文件的 capability xattr
getfattr -n security.capability /usr/bin/ping
# security.capability: \x01\x00\x00\x02...  ← VFS_CAP_REVISION_2 格式
```

### 3.15 inode 生命周期：从创建到真正回收

inode 的生命周期由两个独立计数器控制：

```
inode 生命周期状态图：

open() 或 mkdir() 或 creat()
         │
         ▼
    i_nlink=1        ← 目录项引用计数（rm 时减 1）
    i_count=1        ← VFS 对象引用计数（close() 时减 1）
         │
    ─────┼──────────────── 硬链接（ln）
    │    │                 i_nlink=2
    │    ▼
    │  写入数据（write / mmap）
    │  更改元数据（chmod, chown）
    │    │
    ├────┼──────────────── unlink（rm）
    │    │                 i_nlink 减 1
    │    │
    │    ▼
    │  i_nlink=0（无目录项引用，文件"不可见"）
    │  i_count ≥ 1（进程仍持有 open fd）
    │    │
    │    │  ← 此状态 = deleted-but-open（磁盘空间未释放）
    │    │
    │    ▼
    │  close(fd) / 进程退出
    │  i_count 减至 0
    │    │
    │    ▼
    │  真正的 inode 回收：
    │    inode_operations.evict_inode()  ← ext4_evict_inode()
    │       → 截断数据块（ext4_truncate）
    │       → 释放 extent 树
    │       → 将 inode 号标记为可重用（bitmap）
    │       → 写 journal 事务
    │    ▼
    └──  inode 编号回到空闲池，磁盘空间释放
```

```bash
# 实际演示 deleted-but-open 状态

# 步骤 1：打开文件并保持 fd
exec 9>/tmp/demo_deleted.txt  # 在当前 shell 开 fd 9
echo "I am still alive" >&9

# 步骤 2：删除目录项（unlink）
rm /tmp/demo_deleted.txt

# 步骤 3：验证文件仍然存在（inode 未回收）
ls -la /proc/$$/fd/9
# lrwx------ 1 user user 64 Apr 12 /proc/1234/fd/9 -> /tmp/demo_deleted.txt (deleted)
# ↑ 注意末尾的 "(deleted)" 标记

# 通过 fd 仍然可以读写
cat /proc/$$/fd/9    # 读出 "I am still alive"
echo "more content" >/proc/$$/fd/9

# 步骤 4：用 lsof 找到所有此类"僵尸文件"
lsof +L1  # +L1 = link count < 1，即 unlinked 但仍 open 的文件
# COMMAND  PID  USER  FD  TYPE DEV  SIZE/OFF  NLINK  NODE NAME
# bash     1234 user   9w REG  8,1    17       0  123456  /tmp/demo_deleted.txt (deleted)
# ↑ NLINK=0 确认目录项已删除，但 inode 还活着

# 步骤 5：不重启情况下回收空间（截断文件内容）
truncate -s 0 /proc/$$/fd/9  # 内容清空，磁盘空间立即释放

# 步骤 6：关闭 fd，inode 最终回收
exec 9>&-

# 验证 inode 中的引用计数（内核 /proc 接口）
python3 - << 'EOF'
import os

# 创建并删除文件，同时保持 fd
import tempfile
f = tempfile.NamedTemporaryFile(delete=False)
fname = f.name
f.write(b"content")
f.flush()

# 删除目录项
os.unlink(fname)

# 通过 /proc/self/fdinfo 看文件状态
fdinfo_path = f"/proc/self/fdinfo/{f.fileno()}"
print(f"文件已删除，但 fd={f.fileno()} 仍有效")
with open(fdinfo_path) as fi:
    print(fi.read())
# pos:  7       ← 当前位置（写了7字节）
# flags: 02     ← O_RDWR
# mnt_id: 23    ← 挂载点 ID

# 通过 stat fd 看 nlink
st = os.fstat(f.fileno())
print(f"st_nlink = {st.st_nlink}")  # = 0（目录项已删除）
print(f"st_size  = {st.st_size}")   # = 7（内容还在）

f.close()  # 这才是真正的 inode 回收时刻
print("fd closed, inode will be evicted now")
EOF
```

### 3.16 时间戳在内核中的更新规则

三种时间戳的更新由 VFS 统一管理：

```c
/* fs/inode.c — 时间戳更新函数 */

/* 更新 atime（内核会做 lazy atime 优化）*/
void touch_atime(const struct path *path) {
    /* 如果挂载了 relatime 或 noatime，可能跳过 */
    /* relatime 规则：只有 atime < mtime 或 atime < ctime 时才更新 */
}

/* 更新 mtime 和 ctime（内容写入时调用）*/
void inode_update_time(struct inode *inode, struct timespec64 *time,
                       int flags) {
    if (flags & S_MTIME) inode->i_mtime = *time;  /* 内容修改时 */
    if (flags & S_CTIME) inode->i_ctime = *time;  /* inode 变化时 */
    if (flags & S_ATIME) inode->i_atime = *time;  /* 访问时 */
    mark_inode_dirty(inode);  /* 标记 inode 需要写回 */
}
```

```bash
# 实验：验证不同操作对三种时间戳的影响
touch /tmp/ts_test.txt
echo "initial" > /tmp/ts_test.txt

show_times() {
    stat /tmp/ts_test.txt | grep -E "Access|Modify|Change"
}

echo "=== 初始状态 ==="
show_times

sleep 1
echo "=== 读取内容（cat）— 只更新 atime ==="
cat /tmp/ts_test.txt > /dev/null
show_times
# Access 变了，Modify 和 Change 不变

sleep 1
echo "=== 修改内容（echo >>）— 更新 mtime + ctime ==="
echo "more" >> /tmp/ts_test.txt
show_times
# Modify 和 Change 都变了，Access 也可能变（取决于 atime 策略）

sleep 1
echo "=== chmod 改权限 — 只更新 ctime ==="
chmod 644 /tmp/ts_test.txt
show_times
# 只有 Change 变了（inode 状态改变，但内容没变）

sleep 1
echo "=== ln 添加硬链接 — 更新 ctime（链接计数变化）==="
ln /tmp/ts_test.txt /tmp/ts_test_link.txt
show_times
# Change 变了（i_nlink 增加了）

# 查看当前挂载的 atime 策略
cat /proc/self/mountinfo | grep -E "relatime|noatime|strictatime"
# 大多数现代系统用 relatime（默认），减少 atime 写操作
```

---

## 常见误区

- 误以为 9 个 `rwx` 位就能解释所有权限问题
- 误以为 `ctime` 是创建时间
- 误以为硬链接是内容复制
- 误以为 deleted-but-open 是异常情况
- 误以为 `chmod` 能解决所有“Permission denied”

---

## 本章小结

| 主题 | 结论 |
|------|------|
| DAC | 是基础权限模型，但不是全部 |
| 目录权限 | 与普通文件权限语义不同 |
| `umask` | 是对象创建协议的一部分 |
| 时间戳 | atime / mtime / ctime 代表不同层次变化 |
| 硬链接 | 是增加名字，不是复制内容 |
| 软链接 | 是路径对象，不是目标对象 |
| 生命周期 | 目录项删除不等于对象立即消失 |

---

## 练习题

### 基础题

**3.1** 为什么目录上的 `x`（执行/遍历）权限在路径语义里比普通文件的 `x` 更容易引发误解？举出一个”目标文件权限没问题但仍然 Permission denied”的具体场景。

**3.2** `umask` 如何影响新建文件和目录的最终权限？如果程序调用 `open(path, O_CREAT, 0666)`，在 `umask=0022` 的环境下最终权限是多少？

### 中级题

**3.3** 为什么 `ctime` 不是”创建时间”？在哪些操作下 `ctime` 会更新而 `mtime` 不会？给出至少三个具体例子。

**3.4** 硬链接与软链接分别工作在哪个层次？为什么硬链接通常不能跨文件系统，而软链接可以？

### 提高题

**3.5** 某个日志进程删除了 `/var/log/app.log`（`unlink`），但磁盘空间没有释放。请设计完整排查方案：①用什么命令找到持有该文件 fd 的进程；②用什么命令确认 inode 引用计数和链接计数；③分析是否有 bind mount 或 hardlink 也持有引用；④在不重启服务的前提下，如何安全地回收空间（提示：考虑 `/proc/<pid>/fd/`）。

---

## 练习答案

**3.1** 目录的 `x` 权限控制”能否穿越该目录继续解析路径”，缺少它路径查找就在该层停止，报 `EACCES`。场景：`/data` 权限为 `755`，`/data/private` 权限为 `700`，其他用户访问 `/data/private/file.txt` 时，`/data` 可列（有 `r`）但 `/data/private` 无法穿越（缺 `x`），报 Permission denied。

**3.2** 最终权限 = 程序指定模式 AND NOT umask = `0666 & ~0022 = 0644`（`rw-r--r--`）。目录创建时同理，`mkdir` 默认模式 `0777 & ~0022 = 0755`。

**3.3** `ctime` 是 inode 状态变化时间，非创建时间（Linux 无专门创建时间字段）。`ctime` 更新而 `mtime` 不更新的场景：①`chmod` 改变权限位；②`chown` 改变所有者；③`ln`（硬链接）增加链接计数；④`rename` 目录项（被 rename 的文件 ctime 变化）。

**3.4** 硬链接工作在目录项层（inode 层）：多个目录项指向同一 inode，inode 号相同。软链接工作在路径层：软链接是独立 inode，内容是目标路径字符串。硬链接不能跨文件系统：因为 inode 号只在同一文件系统内唯一，跨文件系统 inode 号无意义；软链接只存路径字符串，解析时重新走路径查找，可跨文件系统。

**3.5** 排查方案：①`lsof | grep deleted` 或 `lsof /var/log/app.log` 找持有 fd 的进程 PID；②`stat /proc/<pid>/fd/<n>` 查看 inode，`ls -l /proc/<pid>/fd/` 确认 `(deleted)` 标记，`stat` 输出 `Links: 0` 说明目录项已删除但 inode 仍被引用；③`findmnt` 检查是否有 bind mount 指向该 inode；`find / -inum <inode_num> -xdev` 查硬链接；④不重启回收：向进程发 `SIGHUP` 让其重新打开日志文件，或 `truncate -s 0 /proc/<pid>/fd/<n>` 清空文件内容（空间立即释放，进程仍可写入）。

---

## 延伸阅读

1. **Kerrisk**. *The Linux Programming Interface*, Ch.15 — 文件属性、时间戳与权限
2. **man 7 inode** — inode 字段和时间戳完整说明
3. **POSIX.1-2017**, *Base Definitions*, Section 4.4 — 文件访问权限模型
4. **LWN.net**. *Capabilities: why they exist and how they work* — capability 机制详解

---

[← 上一章：路径、文件与目录](./02-paths-files-and-directories.md)

[下一章：inode、dentry 与目录项 →](../part2-core-abstractions/04-inodes-dentries-and-directory-entries.md)

[返回目录](../README.md)
