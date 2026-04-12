# 第15章：bind mount、namespace 与隔离

> 容器里的文件系统隔离并不是“复制一整套根目录”，而是通过 mount namespace、bind mount、propagation、pivot_root 等机制，构造出一个进程可见但并不等于宿主视图的名字空间。

## 学习目标

完成本章后，你将能够：

1. 解释 bind mount、mount namespace、pivot_root、chroot 分别工作在哪一层
2. 理解 propagation（shared/private/slave）为什么决定隔离边界是否漏出
3. 说明容器卷挂载、根文件系统切换、宿主/容器路径错位的根因
4. 认识 idmapped mounts、user namespace 与 rootless 场景的关系
5. 把“同一路径在不同视图下不同对象”当成容器文件系统常态，而不是例外

---

## 正文内容

### 15.1 bind mount 改变的是挂载树，不是创建一个链接文件

bind mount 不是符号链接，也不是复制目录。它做的是：

- 选定一棵已有对象树
- 在挂载树里给它另开一个接入点

这意味着：

- 同一对象可通过多个路径入口被看见
- 新入口仍处在挂载语义层，而不是普通文件层
- 后续挂载遮蔽、只读策略、namespace 视图都可能继续作用在它上面

### 15.2 mount namespace 解决的是“谁看到哪棵挂载树”

mount namespace 不是重新发明文件系统，而是让不同进程组拥有不同的挂载视图。于是：

- 相同路径字符串不再保证指向同一对象
- 容器里的 `/proc`、`/sys`、`/app` 可能和宿主完全不是同一路径树
- 宿主上看到的“原目录内容”，在容器里可能已被 bind mount 或 overlay 覆盖

容器“有自己的根文件系统”这句话，更准确的理解是：“它有自己的挂载视图”。

### 15.3 `chroot` 不是完整隔离，`pivot_root` 更接近真正换根

- `chroot` 改变路径解析的根边界
- `pivot_root` 更接近把当前挂载树根切换到新根，并把旧根移到另一个位置等待处理

只靠 `chroot` 并不能自然提供完整容器隔离语义，因为：

- 挂载视图未必隔离
- 打开的 fd 仍可能指向旧根外部对象
- 许多内核对象和挂载点仍共享宿主视图

所以真正的容器根切换几乎总和 mount namespace 一起讨论。

### 15.4 propagation 决定隔离有没有“漏风”

共享传播语义（shared/private/slave/unbindable）决定挂载事件如何传播：

- shared：彼此传播
- private：完全不传播
- slave：接收上游传播但不向上游传播
- unbindable：不能被 bind mount

这会直接影响：

- 宿主新挂载是否进入容器
- 容器里的挂载是否反向影响宿主
- sidecar / CSI / Kubernetes volume 行为是否符合预期

很多“为什么容器里突然看到宿主新挂载”或“为什么 volume mount 没按预期更新”的问题，本质上是 propagation 没配对。

### 15.5 volume mount、镜像内容和覆盖关系

容器里一个路径常常同时受三层影响：

1. 镜像层提供的默认内容
2. overlayfs merged 视图提供运行时根文件系统
3. bind mount / volume mount 覆盖某个具体子路径

因此一个路径“看不到镜像里的原内容”，未必是文件没了，而是：

- 被 volume mount 遮住了
- 被 bind mount 替换成宿主目录了
- 在另一个 namespace 里看的是完全不同的挂载树

### 15.6 user namespace 和 idmapped mount 会进一步改变身份语义

在 rootless 容器或更现代的挂载设计里，除了“看哪棵树”，还要问“以什么 uid/gid 语义看这棵树”。

idmapped mounts 的直觉是：

- 同一底层对象
- 在某个挂载视图中
- 可以呈现不同的 uid/gid 映射解释

这对无特权容器和共享目录尤其重要，因为它把身份映射从“复制一份文件”变成了“挂载视图层翻译”。

### 15.7 rootless 场景为什么格外复杂

rootless 容器经常把这些问题叠在一起：

- user namespace 身份映射
- mount namespace 视图隔离
- FUSE 或用户态文件系统参与路径
- overlayfs / bind mount 的权限与能力边界

所以 rootless 不只是“没有 root”，而是多层对象与权限语义一起变化。

### 15.8 一个排障框架：容器里路径错了，到底错在哪层

可以先问：

1. 当前进程在哪个 mount namespace？
2. 这个路径是否被 bind mount/volume mount 覆盖？
3. propagation 是否导致宿主/容器视图相互影响？
4. 是对象权限问题，还是名字空间视图问题？
5. 是否存在 idmapped / user namespace 导致的 uid/gid 解释差异？

如果这些问题不先问清楚，容器里的路径问题看起来会特别“玄学”。

---

### 15.9 `struct mnt_namespace` 与挂载树的内核实现

每个 mount namespace 在内核中由 `struct mnt_namespace` 表示：

```c
/* fs/mount.h */
struct mnt_namespace {
    struct ns_common    ns;             /* namespace 通用头（含 inum：/proc/pid/ns/mnt 的 inode 号）*/
    struct mount        *root;          /* 此 namespace 的根挂载 */
    struct list_head    list;           /* 所有 mount 的链表 */
    spinlock_t          ns_lock;        /* 保护 list 的锁 */
    struct user_namespace *user_ns;     /* 所属 user namespace */
    struct ucounts      *ucounts;       /* 用于限制 namespace 数量 */
    u64                 seq;            /* 序列号（mount 事件计数）*/
    wait_queue_head_t   poll;           /* 等待 mount 事件的队列（/proc/self/mountinfo poll 用）*/
    u64                 event;          /* 事件计数（唤醒 poll 等待者）*/
    unsigned int        mounts;         /* 此 namespace 中 mount 的总数 */
    unsigned int        pending_mounts; /* 待处理 mount 数（propagation 中）*/
};

/* struct mount（每个挂载点的完整描述）*/
struct mount {
    struct hlist_node   mnt_hash;       /* 挂载点哈希表节点 */
    struct mount        *mnt_parent;    /* 父挂载 */
    struct dentry       *mnt_mountpoint;/* 挂载点 dentry（在父挂载中的位置）*/
    struct vfsmount     mnt;            /* 公开接口（含 mnt_root 和 mnt_sb）*/

    /* 传播组 */
    struct list_head    mnt_mounts;     /* 子挂载链表 */
    struct list_head    mnt_child;      /* 在父挂载的 mnt_mounts 中的节点 */
    struct list_head    mnt_instance;   /* 同一 superblock 的所有 mount 链表 */

    /* peer/slave/share 传播链表 */
    struct list_head    mnt_share;      /* 共享挂载的 peer 环形链表 */
    struct list_head    mnt_slave_list; /* 此挂载的 slave 列表 */
    struct list_head    mnt_slave;      /* 在 master 的 slave_list 中的节点 */
    struct mount        *mnt_master;    /* slave 的 master 挂载 */

    struct mnt_namespace *mnt_ns;       /* 所属 namespace */
    int                 mnt_id;         /* 唯一挂载 ID（/proc/self/mountinfo 第一列）*/
    int                 mnt_group_id;   /* peer group ID（shared 传播组）*/
    int                 mnt_expiry_mark;/* 是否标记为可过期（autofs 用）*/
    struct hlist_head   mnt_pins;       /* 固定此挂载的 pin 列表（防止被 umount）*/
    struct hlist_node   mnt_mp_list;    /* 挂载点链表节点 */
    struct list_head    mnt_umounting;  /* umount 阶段链表 */
    unsigned int        mnt_flags;      /* MNT_READONLY, MNT_NOSUID 等标志 */
    struct path         mnt_ex_mountpoint; /* 与 idmapped 相关的原始挂载点 */
};
```

**创建和查看 mount namespace**：

```bash
# 查看当前进程的 mount namespace ID
ls -la /proc/$$/ns/mnt
# lrwxrwxrwx 1 user user 0 Apr 12 12:00 /proc/1234/ns/mnt -> mnt:[4026531840]
#                                                              ↑ inode 号 = namespace ID

# 对比两个进程是否在同一 namespace
ls -la /proc/1/ns/mnt /proc/$$/ns/mnt
# → 相同 inode 号 = 同一 namespace

# 使用 unshare 创建新 mount namespace（无需 root，Linux 3.8+）
unshare --mount bash  # 进入新 mount namespace 的 bash
# 此后所有 mount/umount 操作只影响新 namespace，不影响宿主

# 用 nsenter 进入已有 namespace
nsenter --mount=/proc/1234/ns/mnt bash

# 监控 mount namespace 事件（poll /proc/self/mountinfo）
python3 - <<'EOF'
import select, time

with open('/proc/self/mountinfo') as f:
    while True:
        # 等待文件可读（有新的 mount 事件）
        ready = select.select([f], [], [], 5)  # 5 秒超时
        if ready[0]:
            f.seek(0)
            content = f.read()
            print(f"[{time.strftime('%H:%M:%S')}] Mount event! Mounts: {len(content.splitlines())}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] No mount events in 5s")
EOF
```

---

### 15.10 bind mount 内核实现与 propagation 链表

**`do_loopback`：bind mount 的内核函数**：

```
bind mount 执行路径（mount --bind /src /dst）：

sys_mount(source, target, fstype=NULL, flags=MS_BIND, ...)
  └─ do_mount()
        └─ do_loopback(path, source, mnt_flags)
              ├─ lookup_mnt(source)           ← 找到 source 对应的 struct mount
              ├─ copy_tree(old_mnt, source_dentry)  ← 克隆挂载子树
              │       → 创建新的 struct mount，指向同一 superblock
              │       → 新 mount 的 mnt_root = source_dentry
              │       → 不复制文件数据，只复制挂载元数据
              └─ attach_recursive_mnt(new_mnt, target_path)
                    → 将新 mount 附着到 target dentry
                    → 设置 mnt_parent = target 的 mount
                    → 设置 mnt_mountpoint = target dentry
```

**传播机制的链表实现**：

```bash
# 查看挂载传播关系（/proc/self/mountinfo 字段解析）
cat /proc/self/mountinfo
# 36 35 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw,errors=remount-ro
# ↑ID ↑父ID ↑设备 ↑根  ↑挂载点 ↑选项  ↑传播信息     ↑文件系统类型 ↑源设备 ↑挂载选项
# shared:1 → 在 peer group 1 的 shared 挂载
# master:1 → slave 挂载，master 是 peer group 1
# → （无标记）→ private 挂载

# 演示 shared/private/slave 传播
# 场景：宿主新增挂载，观察是否传播到容器 namespace

# 终端1：创建容器 namespace（共享传播）
unshare --mount bash
cat /proc/self/mountinfo | grep "shared:"  # 继承了宿主的 shared 挂载

# 终端2（宿主）：在 /tmp 下创建新挂载
mkdir /tmp/host_mount_test
mount --bind /usr /tmp/host_mount_test
# → 若 /tmp 是 shared，则容器 namespace 也能看到此新挂载（"漏风"）

# 解决方案：先把容器 namespace 中的根挂载改为 private
unshare --mount --propagation private bash
# → 任何宿主新增挂载都不会传播到此容器

# Docker 的实际做法（从 /proc/1/mountinfo 观察）
nsenter --mount=/proc/$(docker inspect --format='{{.State.Pid}}' container_name)/ns/mnt \
    cat /proc/self/mountinfo | head -5
# 容器根通常是 private 或 slave，防止宿主传播
```

**完整的容器文件系统隔离实验**：

```bash
# 用纯 bash 实现最简单的容器文件系统隔离
# 步骤 1: 创建根文件系统（使用 busybox）
mkdir -p /tmp/mycontainer/{bin,proc,sys,dev,tmp,etc}
cp /bin/busybox /tmp/mycontainer/bin/
/tmp/mycontainer/bin/busybox --install /tmp/mycontainer/bin/

# 步骤 2: 用 unshare 创建隔离的 mount namespace + pivot_root
unshare --mount --pid --fork bash << 'CONTAINER_EOF'
# 现在在新 mount namespace 中
# 把 /tmp/mycontainer 变成真正的根

# 先把当前根改成 private（防止传播）
mount --make-rprivate /

# 把 /tmp/mycontainer bind mount 到自身（pivot_root 要求目标是挂载点）
mount --bind /tmp/mycontainer /tmp/mycontainer

# 创建 pivot_root 需要的 old_root 目录
mkdir -p /tmp/mycontainer/old_root

# 执行 pivot_root
cd /tmp/mycontainer
pivot_root . old_root

# 挂载 /proc（新 namespace 中需要自己的 procfs）
mount -t proc proc /proc

# 取消挂载 old_root（清除宿主文件系统的访问）
umount -l /old_root

# 现在真正在隔离的根文件系统中
ls /
exec /bin/sh
CONTAINER_EOF
```

---

### 15.11 idmapped mounts：挂载层的 uid/gid 翻译

Linux 5.12 引入 idmapped mounts，让同一物理存储在不同挂载视图中呈现不同的 uid/gid 映射：

```
idmapped mount 的工作原理：

宿主文件系统：  文件 A 的 uid=1000（宿主 alice）
                文件 B 的 uid=1001（宿主 bob）

普通 bind mount：容器内 uid 仍是 1000/1001 → 容器内用户可能无权访问

idmapped mount：通过 ioctl(MOUNT_ATTR_IDMAP) 设置映射：
  宿主 uid 1000  →  容器内 uid 0（root）
  宿主 uid 1001  →  容器内 uid 1（daemon）

效果：容器内用户以 uid=0 访问文件 A，内核在检查权限时自动翻译回 uid=1000

内核实现路径：
  vfs_getattr() → generic_fillattr()
    → i_uid_into_vfsuid(mnt_userns, inode)
        → map_id_up(uid_map, inode->i_uid)  ← 按挂载的 uid 映射表翻译
```

```bash
# 查看是否支持 idmapped mount
grep "idmapped" /proc/kallsyms 2>/dev/null | head -3  # 有输出说明支持

# 用 mount-idmapped 工具演示（需要 util-linux >= 2.39）
# 场景：让非 root 的 uid=1000 用户可以访问 root 创建的文件
mkdir /tmp/root_files
echo "root content" > /tmp/root_files/secret.txt  # root 拥有
chown root:root /tmp/root_files/secret.txt
chmod 600 /tmp/root_files/secret.txt

# 创建 idmapped mount：把 root(0) → user(1000) 的访问视图
# （需要 root 权限创建映射）
mount-idmapped --map-mount=b:0:1000:1 /tmp/root_files /tmp/user_view

# 现在 uid=1000 的用户以 /tmp/user_view 访问，内核认为他/她就是 root
ls -la /tmp/user_view/secret.txt  # 显示 uid=1000 拥有（实际上是 root）
# 非特权用户 su -s /bin/bash user -c "cat /tmp/user_view/secret.txt"  # 可以读取

# 清理
umount /tmp/user_view

# 在 Podman rootless 中观察 idmapped mount 的使用
podman run --rm alpine cat /proc/self/mountinfo | grep idmap
# → rootless Podman 自动使用 idmapped mounts 实现 uid 映射
```

---

### 15.12 `pivot_root` 与 `chroot` 的内核路径对比

```c
/* fs/namespace.c — pivot_root 系统调用 */
SYSCALL_DEFINE2(pivot_root, const char __user *, new_root,
                             const char __user *, put_old)
{
    /* 核心操作序列：*/
    /* 1. 查找 new_root 和 put_old 的路径 */
    /* 2. 验证 new_root 必须是独立的挂载点 */
    /* 3. 验证 put_old 必须在 new_root 下 */
    /* 4. 验证当前 root 也必须是挂载点 */

    /* 临界区（锁住整个 namespace）：*/
    /* 5. 把 old_root 的挂载移到 put_old 下（旧根变成 new_root 下的一个目录）*/
    /* 6. 把 new_root 的挂载设为当前 namespace 的根 */
    /* 7. 更新所有进程的 root 和 pwd */
}
```

```bash
# 对比 chroot 和 pivot_root 的安全边界

# chroot 的安全问题：
# 1. 仍使用宿主 mount namespace（能看到宿主所有挂载点）
# 2. root 用户可以 break out（通过 open("/../"), chdir("/../")）
# 3. /proc、/sys 未隔离

# pivot_root 配合 mount namespace 的安全优势：
# 1. 完整替换挂载树根
# 2. old_root 可以被 umount，彻底断开宿主访问
# 3. 配合 user namespace 可实现无特权容器

# strace 对比：
strace chroot /tmp/mycontainer /bin/sh 2>&1 | grep -E "chroot|mount|pivot"
# chroot("/tmp/mycontainer") = 0
# → 只有一个 chroot 系统调用

strace unshare --mount bash -c "pivot_root . old_root" 2>&1 | grep -E "unshare|pivot|mount"
# unshare(CLONE_NEWNS)   = 0
# mount("", "/", NULL, MS_SLAVE|MS_REC, NULL) = 0  ← 先改成 slave
# mount("/tmp/mycontainer", "/tmp/mycontainer", NULL, MS_BIND|MS_REC, NULL) = 0
# pivot_root(".", "old_root") = 0
# umount2("old_root", MNT_DETACH) = 0              ← 断开宿主根
```

---

### 15.13 容器挂载问题的完整排查工具链

**工具一：用 `findmnt` 和 `nsenter` 进入容器视图**

```bash
# 找到容器进程 PID（以 Docker 为例）
CONTAINER="my_container"
PID=$(docker inspect --format '{{.State.Pid}}' $CONTAINER)
echo "Container PID: $PID"

# 查看容器的完整挂载树（从宿主角度）
nsenter --mount=/proc/$PID/ns/mnt -- findmnt --tree
# TARGET                          SOURCE                TYPE    OPTIONS
# /                               overlay               overlay rw,lowerdir=...
# ├─/proc                         proc                  proc    rw,nosuid
# ├─/sys                          sysfs                 sysfs   ro,nosuid
# ├─/dev                          tmpfs                 tmpfs   rw
# │ ├─/dev/pts                    devpts                devpts  rw
# │ └─/dev/shm                    shm                   tmpfs   rw,size=65536k
# ├─/app                          /dev/sdb1             ext4    rw   ← volume mount
# └─/etc/hosts                    /dev/sda1[/...]       ext4    rw   ← bind mount

# 查看某个特定挂载点的详细信息
nsenter --mount=/proc/$PID/ns/mnt -- findmnt --output=TARGET,SOURCE,FSTYPE,OPTIONS,PROPAGATION /app
# TARGET  SOURCE    FSTYPE  OPTIONS  PROPAGATION
# /app    /dev/sdb1 ext4    rw,...   shared:15  ← shared 意味着可能向宿主传播

# 比较容器和宿主看到的同一路径
echo "=== 宿主视角 ==="
stat /var/lib/docker/volumes/myapp/_data/config.yaml 2>/dev/null

echo "=== 容器视角 ==="
nsenter --mount=/proc/$PID/ns/mnt -- stat /app/config.yaml 2>/dev/null

# 检查容器 mount namespace ID
ls -la /proc/$PID/ns/mnt
# lrwxrwxrwx 1 root root 0 Apr 12 /proc/1234/ns/mnt -> mnt:[4026532456]
ls -la /proc/self/ns/mnt
# lrwxrwxrwx 1 root root 0 Apr 12 /proc/self/ns/mnt -> mnt:[4026531840]
# 不同 ID = 不同 namespace（已隔离）
```

**工具二：分析 propagation 泄漏**

```bash
# 场景：Kubernetes 节点上，一个 Pod 使用 hostPath volume
# 宿主上的新挂载是否会泄漏到 Pod？

# 查看 Pod 的 mount propagation 配置
kubectl get pod mypod -o jsonpath='{.spec.volumes[*].hostPath}'
kubectl get pod mypod -o jsonpath='{.spec.containers[*].volumeMounts[*].mountPropagation}'
# 可能的值：
#   None（默认）      ← private/slave，宿主变化不传播进容器
#   HostToContainer   ← slave，宿主新挂载会传播进容器
#   Bidirectional     ← shared，双向传播（需要 privileged）

# 验证传播行为
HOST_NEW_MOUNT=/tmp/k8s_test_mount
mkdir -p $HOST_NEW_MOUNT
mount --bind /usr $HOST_NEW_MOUNT

# 检查容器是否看到新挂载
nsenter --mount=/proc/$PID/ns/mnt -- ls $HOST_NEW_MOUNT 2>/dev/null
# → 若有输出：传播生效（HostToContainer 或 Bidirectional）
# → 若失败：传播隔离（None）

umount $HOST_NEW_MOUNT
rmdir $HOST_NEW_MOUNT
```

**工具三：观察和调试 overlayfs + bind mount 的叠加**

```bash
# 容器运行时 rootfs 的真实结构分析
PID=$(docker inspect --format '{{.State.Pid}}' my_container)

# 查看 overlayfs 的实际挂载参数
cat /proc/$PID/mountinfo | grep "^[0-9]* [0-9]* 0:52"
# 或
findmnt --pid $PID --output=SOURCE,TARGET,OPTIONS | grep overlay
# SOURCE   TARGET  OPTIONS
# overlay  /       rw,lowerdir=/var/lib/docker/overlay2/abc.../diff:
#                            /var/lib/docker/overlay2/def.../diff,
#                  upperdir=/var/lib/docker/overlay2/xyz.../diff,
#                  workdir=/var/lib/docker/overlay2/xyz.../work

# 分析 lowerdir 层次（镜像层从上到下）
LOWERDIR=$(cat /proc/$PID/mountinfo | grep "overlay" | \
    grep -o "lowerdir=[^,]*" | cut -d= -f2)
echo "Lower layers (newest first):"
IFS=':' read -ra LAYERS <<< "$LOWERDIR"
for i in "${!LAYERS[@]}"; do
    echo "  [$i] ${LAYERS[$i]}"
    ls "${LAYERS[$i]}" 2>/dev/null | head -3
done

# 检查 upper layer（容器的写层）
UPPERDIR=$(cat /proc/$PID/mountinfo | grep "overlay" | \
    grep -o "upperdir=[^,]*" | cut -d= -f2)
echo "\nUpper layer (container writes): $UPPERDIR"
ls -la "$UPPERDIR" | head -20

# 找出容器写了哪些文件（upper 里的所有文件）
find "$UPPERDIR" -not -name "*.wh.*" 2>/dev/null | head -20
# 带 .wh. 前缀的是 whiteout（已删除的文件）
```

### 15.14 Kubernetes CSI 与 volume 挂载的挂载树分析

```bash
# Kubernetes 使用 CSI 插件时，volume 如何挂载到 Pod

# CSI 挂载流程（以节点为例）：
# 1. kubelet 调用 NodeStageVolume → 把 PV 挂载到全局 staging 目录
#    /var/lib/kubelet/plugins/kubernetes.io/csi/pv/<pv-name>/globalmount/
# 2. kubelet 调用 NodePublishVolume → bind mount 到 Pod 目录
#    /var/lib/kubelet/pods/<pod-uid>/volumes/kubernetes.io~csi/<pv-name>/mount/
# 3. 容器运行时把 Pod volume 目录 bind mount 进容器

# 查看所有 CSI 挂载
mount | grep "kubernetes.io~csi"
# /dev/sdb1 on /var/lib/kubelet/plugins/kubernetes.io/csi/pv/pvc-xxx/globalmount type ext4 (rw)
# /dev/sdb1 on /var/lib/kubelet/pods/yyy/volumes/kubernetes.io~csi/pvc-xxx/mount type ext4 (rw)

# 追踪 Pod volume 到底层设备的完整路径
POD_UID="yyy-zzz-..."
PV_NAME="pvc-xxx-..."

echo "=== Global mount (staging) ==="
ls /var/lib/kubelet/plugins/kubernetes.io/csi/pv/$PV_NAME/globalmount/

echo "=== Pod mount (published) ==="
ls /var/lib/kubelet/pods/$POD_UID/volumes/kubernetes.io~csi/$PV_NAME/mount/

echo "=== Container sees ==="
PID=$(crictl inspect $(crictl pods --name mypod -q) 2>/dev/null | \
    python3 -c "import json,sys; d=json.load(sys.stdin); print(d['info']['pid'])")
nsenter --mount=/proc/$PID/ns/mnt -- ls /data 2>/dev/null

# 检查 mountPropagation 的实际内核状态
cat /proc/$(ps -ef | grep kubelet | grep -v grep | awk '{print $2}')/mountinfo | \
    grep "$PV_NAME" | awk '{print $1, $2, $7, $9, $NF}'
# MountID ParentID PropGroup Mountpoint Options
```

### 15.15 rootless 容器的 mount namespace + user namespace 联动

```bash
# rootless Podman 如何在无 root 权限下实现容器挂载

# 查看 rootless Podman 创建的 namespace 链
podman run -d --name rootless_test alpine sleep 60

# 获取 PID
PID=$(podman inspect rootless_test --format '{{.State.Pid}}')

# 查看 namespace 配置
echo "=== Namespace IDs ==="
for ns in mnt uts ipc pid user net; do
    host=$(ls -la /proc/self/ns/$ns | awk '{print $NF}')
    container=$(ls -la /proc/$PID/ns/$ns | awk '{print $NF}')
    same=$([ "$host" = "$container" ] && echo "SHARED" || echo "ISOLATED")
    echo "  $ns: $same ($container)"
done

# 查看 user namespace uid/gid 映射
echo "=== UID Mapping ==="
cat /proc/$PID/uid_map
# 0   1000   1   ← 容器内 uid 0 映射到宿主 uid 1000
# 1   100000 65536  ← 容器内 uid 1-65536 映射到宿主 100000+

echo "=== GID Mapping ==="
cat /proc/$PID/gid_map

# 在 rootless 环境下验证 idmapped mount
# 容器内的 "root" 创建的文件在宿主上显示为 uid=1000
nsenter --mount=/proc/$PID/ns/mnt -- touch /tmp/from_container
ls -la /proc/$PID/root/tmp/from_container
# -rw-r--r-- 1 user user 0 Apr 12 /tmp/...
# ↑ 宿主上显示为普通用户（uid=1000），容器内显示为 root（uid=0）

podman stop rootless_test && podman rm rootless_test
```

---

## 本章小结

| 主题 | 结论 |
|------|------|
| bind mount | 在挂载树层重新接入已有对象树 |
| mount namespace | 定义不同进程组看到的挂载视图 |
| `chroot` / `pivot_root` | 都影响根边界，但隔离强度和层次不同 |
| propagation | 决定挂载事件是否泄露或同步 |
| idmapped / user namespace | 让身份语义也变成视图的一部分 |
| 容器路径问题 | 往往首先是视图层问题，而不是文件消失 |

---

## 练习题

1. 为什么 bind mount 不应被理解成“目录快捷方式”？
2. mount namespace 和 `pivot_root` 在容器里分别解决什么问题？
3. propagation 为什么会影响 volume mount 的行为？
4. idmapped mount 的直觉是什么？
5. 容器里同一路径和宿主不一致时，第一批应该排查哪几层？
