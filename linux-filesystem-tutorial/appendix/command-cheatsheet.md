# 附录 A1：常用命令速查

## 路径与元数据

```bash
pwd
ls -la
ls -li <path>
stat <path>
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
mount | head
lsblk
blkid
```

## 打开文件与排障

```bash
lsof | head
lsof +L1
strace -e trace=file <cmd>
iostat -xz 1
pidstat -d 1
filefrag -v <path>
```

## 说明

- `df -h` 关注文件系统视角的容量
- `du -sh` 关注目录树累计占用
- `df -i` 用于检查 inode 是否耗尽
- `findmnt` 用于理解目录树背后的挂载关系
- `lsof +L1` 常用于发现 deleted-but-open 文件
- `iostat -xz 1` 适合观察设备层忙碌度、等待和队列
- `pidstat -d 1` 适合把 I/O 压力归因到具体进程
- `filefrag -v` 可帮助观察文件 extent / 碎片化直觉
