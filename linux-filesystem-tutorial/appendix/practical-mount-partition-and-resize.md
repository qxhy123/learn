# 附录 A5：分区、mkfs、挂载与动态扩容实操

这份附录的目标不是把你训练成存储管理员，而是把教程里已经讲过的对象层、挂载层、空间层，串成一条真正能动手的实践路线。

重点不是“背命令”，而是每一步都知道自己正在改哪一层：

- 块设备层
- 分区层
- 文件系统实例层
- 挂载关系层
- 容量可见性层

---

## 一、先记住三个安全原则

### 1. 这些操作很多都是破坏性的

尤其是：

- `parted mklabel`
- `parted mkpart`
- `mkfs.ext4`
- `mkfs.xfs`
- `pvcreate`

它们面对错误设备时，后果通常不是“命令报错”，而是“旧数据没了”。

### 2. 永远先确认目标设备，再执行写操作

最小检查集：

```bash
lsblk -f
blkid
findmnt
```

至少先回答：

- 我操作的是哪块盘？
- 这块盘上现在有没有分区和文件系统？
- 它有没有已经被挂载到某个路径？

### 3. 不要把多层变化混成一步

你在做的可能分别是：

- 改磁盘大小
- 改分区表
- 初始化文件系统
- 建立挂载关系
- 修改开机持久挂载
- 扩大文件系统可用空间

这些动作经常相邻，但不是同一层。

---

## 二、最小路径一：新盘初始化并挂载到 `/data`

下面以“新增一块数据盘，挂载到 `/data`”为例。

### 第 1 步：识别新盘

```bash
lsblk -f
```

你要先确认：

- 哪个设备是新盘，例如 `/dev/vdb`
- 它现在是否没有分区、没有文件系统
- 你没有误选当前系统盘

### 第 2 步：创建分区表和分区

```bash
parted /dev/vdb --script mklabel gpt
parted /dev/vdb --script mkpart primary 1MiB 100%
```

这一步改变的是：

- 这块盘上的分区布局

你还没有：

- 创建文件系统
- 建立挂载关系

### 第 3 步：在分区上创建文件系统

ext4 示例：

```bash
mkfs.ext4 -L data /dev/vdb1
```

xfs 示例：

```bash
mkfs.xfs -L data /dev/vdb1
```

这一步改变的是：

- `/dev/vdb1` 上现在有了一个文件系统实例

### 第 4 步：创建挂载点并临时挂载

```bash
mkdir -p /data
mount /dev/vdb1 /data
findmnt -T /data
```

这一步改变的是：

- 当前运行系统的挂载树

它还不自动意味着：

- 重启后系统还会把这个实例挂回 `/data`

### 第 5 步：使用 UUID 配置持久挂载

先查 UUID：

```bash
blkid /dev/vdb1
```

然后在 `/etc/fstab` 中写类似条目：

```fstab
UUID=<uuid>  /data  ext4  defaults,nofail  0  2
```

如果是 xfs：

```fstab
UUID=<uuid>  /data  xfs  defaults,nofail  0  2
```

为什么优先用 UUID：

- `/dev/sdX` 这类名字在重启、热插拔、云环境里可能漂移
- UUID 更接近“这个文件系统实例是谁”

### 第 6 步：验证持久挂载配置

```bash
findmnt --verify
mount -a
findmnt -T /data
```

这里的目标不是“命令执行过”，而是确认：

- `fstab` 条目能被正确解析
- `/data` 最终真的指向你预期的文件系统实例

---

## 三、最小路径二：LVM 版初始化

如果你不想把文件系统直接建在分区上，而想保留后续在线扩容的灵活性，常见链路是：

```text
磁盘 -> 分区 -> PV -> VG -> LV -> 文件系统 -> mount
```

示例流程：

```bash
parted /dev/vdb --script mklabel gpt
parted /dev/vdb --script mkpart primary 1MiB 100%
pvcreate /dev/vdb1
vgcreate vgdata /dev/vdb1
lvcreate -n lvdata -L 100G vgdata
mkfs.ext4 /dev/vgdata/lvdata
mount /dev/vgdata/lvdata /data
```

这条路径比“直接在分区上 mkfs”多了两层：

- PV/VG/LV

它的价值在于：

- 后续扩容更灵活
- 多块盘整合更自然

代价在于：

- 排障时你必须分清自己卡在哪一层

---

## 四、动态扩容：先问自己扩的是哪一层

动态扩容最容易误判的地方是：你以为系统只剩“一步没做”，实际上可能有三步都还没做。

典型链条：

```text
云盘 / SAN LUN 扩容
-> 内核看到更大的设备
-> 分区或 PV 变大
-> LV 变大
-> 文件系统 grow
-> `df -h` 才看到更大容量
```

---

## 五、常见扩容路径 A：分区直接承载文件系统

假设 `/dev/vdb1` 直接承载 ext4 或 xfs。

### 第 1 步：先确认底层设备真的变大了

```bash
lsblk
```

如果磁盘本身还是旧大小，后面的 grow 都无从谈起。

### 第 2 步：让分区占到新的设备空间

常见做法包括：

```bash
parted /dev/vdb print
parted /dev/vdb --script resizepart 1 100%
```

有些环境也会用 `growpart`，但本质上做的是同一层工作：扩大分区边界。

### 第 3 步：扩大文件系统

ext4：

```bash
resize2fs /dev/vdb1
```

xfs：

```bash
xfs_growfs /data
```

这里最容易搞错的是：

- `resize2fs` 目标通常是块设备或 LV
- `xfs_growfs` 常直接针对挂载点

### 第 4 步：验证

```bash
findmnt -T /data
df -h /data
```

你要确认的不只是“容量变大了”，还要确认：

- 你看的确实是正确挂载点
- 没有在错误路径或错误实例上做 grow

---

## 六、常见扩容路径 B：LVM 承载文件系统

如果文件系统建在逻辑卷上，扩容链路通常是：

### 场景 1：底层磁盘变大，原 PV 跟着变大

```bash
pvresize /dev/vdb1
lvextend -L +100G /dev/vgdata/lvdata
resize2fs /dev/vgdata/lvdata
```

如果是 xfs：

```bash
pvresize /dev/vdb1
lvextend -L +100G /dev/vgdata/lvdata
xfs_growfs /data
```

### 场景 2：新增一块磁盘，加入现有 VG

```bash
pvcreate /dev/vdc1
vgextend vgdata /dev/vdc1
lvextend -l +100%FREE /dev/vgdata/lvdata
resize2fs /dev/vgdata/lvdata
```

如果你想一步把 LV 和文件系统一起长大，可以用：

```bash
lvextend -r -l +100%FREE /dev/vgdata/lvdata
```

但要清楚：

- `-r` 不是“LVM magically 变大”
- 它只是帮你串起了 LV 层和文件系统层的增长动作

---

## 七、为什么 xfs、ext4 的增长手法看起来不同

最常见的操作差异是：

- ext4：常用 `resize2fs`
- xfs：常用 `xfs_growfs`

这背后不是“命令名字不同”这么简单，而是两类文件系统对 grow/shrink 的管理接口不同。

工程上你真正要记住的是：

- 先确认文件系统类型
- 再用匹配它的增长工具
- 不要把“底层设备变大了”误当成“文件系统已经用上了”

---

## 八、缩容不是扩容的镜像操作

扩容常常能做成标准化流程；缩容通常更危险。

原因包括：

- 文件系统要先确认数据能安全挪到更小空间
- 某些文件系统只支持增长，不支持在线缩小
- 一旦块层先缩了、文件系统还没缩，损坏风险会非常高

实践上更稳妥的态度是：

- 默认把缩容视为高风险变更
- 先备份、演练恢复
- 必要时使用离线窗口，而不是在线硬做

---

## 九、最常见的五类误判

1. **把 `/dev/sdX` 当成稳定标识**
   结果是重启后挂错盘或挂不上。

2. **把 `mkfs` 当成“启用一下文件系统”**
   实际上它通常是初始化并覆盖原有内容。

3. **底层磁盘变大后就以为 `df -h` 会自动变大**
   中间还可能隔着分区、LVM、文件系统三层。

4. **在错误对象上执行 grow**
   例如把 xfs 当 ext4 用，或把应该对挂载点执行的命令拿去对原始设备执行。

5. **只验证命令成功，不验证最终视图**
   正确做法是最后总要回到 `findmnt -T <path>` 和 `df -h <path>`。

---

## 十、一条最小复盘清单

每做完一次初始化或扩容，至少记下：

- 目标设备是谁
- 文件系统类型是什么
- 挂载点是什么
- UUID/LABEL 是什么
- `fstab` 条目是什么
- 扩容链路做到哪一层了
- 最终 `findmnt` 和 `df -h` 看到了什么

这样下次出问题时，你看到的就不是一团“好像扩过了”，而是一条可追溯的对象与挂载关系链。
