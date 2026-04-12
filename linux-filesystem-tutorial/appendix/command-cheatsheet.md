# 附录 A1：常用命令速查

## 路径与元数据

```bash
pwd
ls -la
ls -li <path>
stat <path>
namei -om <path>
readlink <path>
realpath <path>
```

## 容量与 inode

```bash
df -h
df -i
du -sh <dir>
find . -xdev -printf '%i %p\n' | head
```

## 挂载与设备

```bash
findmnt
findmnt /
findmnt -T <path>
mount | head
lsblk
blkid
```

## 分区、格式化与持久挂载

```bash
parted /dev/<disk> print
parted /dev/<disk> --script mklabel gpt mkpart primary 1MiB 100%
mkfs.ext4 -L <label> /dev/<partition>
mkfs.xfs -L <label> /dev/<partition>
mount /dev/<partition> <mountpoint>
findmnt -T <mountpoint>
blkid /dev/<partition>
findmnt --verify
```

## LVM 与动态扩容

```bash
pvcreate /dev/<partition>
vgcreate <vg> /dev/<partition>
lvcreate -n <lv> -L 100G <vg>
lvextend -r -l +100%FREE /dev/<vg>/<lv>
pvresize /dev/<partition>
resize2fs /dev/<partition-or-lv>
xfs_growfs <mountpoint>
```

## 打开文件与排障

```bash
lsof | head
lsof +L1
strace -e trace=file <cmd>
strace -yy -e openat,openat2,rename,fsync,fdatasync <cmd>
iostat -xz 1
pidstat -d 1
filefrag -v <path>
```

## 缓存与 writeback

```bash
grep -E 'Dirty|Writeback|Cached|MemAvailable' /proc/meminfo
vmstat 1
cat /proc/vmstat | grep -E 'pgpgin|pgpgout|pswpin|pswpout|nr_dirty|nr_writeback'
slabtop -o
```

## namespace、挂载传播与容器

```bash
findmnt -o TARGET,SOURCE,FSTYPE,OPTIONS,PROPAGATION
cat /proc/self/mountinfo | head -n 20
lsns -t mnt
nsenter --mount=/proc/<pid>/ns/mnt -- findmnt
```

## NFS 与远端文件系统观察

```bash
nfsstat -m
cat /proc/self/mountstats | head -n 40
ss -tanp | head
```

## 高级性能与跟踪

```bash
perf stat <cmd>
perf record -g -- <cmd>
perf report
bpftrace -e 'tracepoint:vfs:vfs_open { @[comm] = count(); }'
```

## 说明

- `df -h` 关注文件系统视角的容量
- `du -sh` 关注目录树累计占用
- `df -i` 用于检查 inode 是否耗尽
- `namei -om` 适合把路径逐段展开，观察软链接、权限和挂载边界
- `findmnt` 用于理解目录树背后的挂载关系
- `findmnt -T <path>` 用于确认一个具体路径到底落在哪个挂载点上
- `parted ... mklabel/mkpart` 会改分区表，操作前必须再次确认目标设备
- `mkfs.ext4` / `mkfs.xfs` 会初始化文件系统，目标分区上的旧数据通常不可恢复
- `blkid` 和 UUID 比 `/dev/sdX` 更适合写入 `/etc/fstab`
- `findmnt --verify` 适合在改完 `/etc/fstab` 后先做结构核对
- `pvresize`、`lvextend`、`resize2fs`、`xfs_growfs` 分别工作在 LVM 物理卷、逻辑卷和文件系统层，不要混层执行
- `lsof +L1` 常用于发现 deleted-but-open 文件
- `strace -yy -e openat,openat2,...` 能帮助你确认程序真实走了哪些路径相关系统调用
- `iostat -xz 1` 适合观察设备层忙碌度、等待和队列
- `pidstat -d 1` 适合把 I/O 压力归因到具体进程
- `filefrag -v` 可帮助观察文件 extent / 碎片化直觉
- `/proc/self/mountinfo` 比 `mount` 更接近内核真实挂载视图
- `lsns -t mnt` 和 `nsenter` 适合排查容器/namespace 看到的路径为什么不同
- `nfsstat -m`、`/proc/self/mountstats` 适合观察 NFS 挂载参数、往返统计和重传情况
- `vmstat`、`/proc/vmstat`、`slabtop` 能帮助区分 page cache、回写和内核缓存压力
- `perf` 适合先做 CPU/调用栈归因，再决定是否需要更细的块层或文件系统层工具
- `bpftrace` 很强，但不一定默认安装，且常需要更高权限
