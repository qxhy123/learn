# 第12章：mmap、direct I/O 与 fsync

> 真正高阶的 I/O 语义不在“读写接口有几种”，而在于这些接口分别把数据放到哪一层、何时可见、何时持久、与谁共享状态，以及崩溃后谁对结果负责。

## 学习目标

完成本章后，你将能够：

1. 区分 buffered I/O、`mmap`、O_DIRECT、`msync`、`fsync`、`fdatasync` 的职责边界
2. 理解 page fault、脏页传播、目录 `fsync`、rename 协议之间的关系
3. 认识 `mmap` 与 direct I/O 混用时的一致性风险
4. 说明为什么 `rename + fsync(file) + fsync(dir)` 是协议，而不是口诀
5. 把 I/O 接口选择与恢复语义、缓存污染、延迟和吞吐联系起来
6. 理解 writeback error、`close()`、DAX、`syncfs()` 这些常被忽视的边界

---

## 正文内容

### 12.1 buffered I/O 是默认路径，不是“低配版”

很多程序默认通过 `read` / `write` 走 buffered I/O。它的特点是：

- 借助 page cache
- 读可以命中缓存和 readahead
- 写通常先形成脏页，再走 writeback
- 接口简单，适配通用文件工作负载

它的问题不在于“慢”，而在于语义常被误解：很多应用以为 `write()` 成功等于持久化完成，实际上 buffered I/O 只是把数据交给内核缓存路径。

### 12.2 `mmap` 改变的是访问方式，不是魔法捷径

`mmap` 把文件页映射到进程地址空间，让访问看起来像普通内存读写。它带来的好处包括：

- 某些场景下减少显式拷贝
- 让随机访问模型更自然
- 让文件页与虚拟内存系统深度融合

但代价也明显：

- 脏页何时真正写回变得更不直观
- page fault 成为数据进入执行路径的重要时机
- 与 `truncate`、并发修改、文件洞、信号异常的交互更复杂

### 12.3 page fault 在 `mmap` 里为什么重要

访问一个映射区域时，真正读取数据常常发生在 page fault：

1. CPU 访问尚未建立的页映射
2. 内核处理 fault
3. 找到或装入相应文件页
4. 建立页表映射

这说明 `mmap` 不是“提前把整个文件装进内存”，而是把“访问文件内容”变成“按页缺失驱动的虚拟内存事件”。

### 12.4 `msync`、`fsync`、`fdatasync` 各补哪一层缺口

这些接口容易被混为一谈，但层次不同：

- `msync`：更偏向映射页与底层同步请求
- `fsync(fd)`：推动该文件数据和必要元数据持久化
- `fdatasync(fd)`：更偏数据相关元数据，减少非必要属性同步

但它们都不自动等于：

- 父目录项已持久化
- 多文件事务已成立
- 业务语义已提交

### 12.5 O_DIRECT 试图绕过什么

O_DIRECT 的核心不是“更高端”，而是试图绕开 page cache，常见动机包括：

- 应用自己就有 buffer pool
- 避免 page cache 被一次性扫描污染
- 追求更可控的 I/O 路径与延迟特征

但它的代价也很真实：

- 对齐要求严格
- 小随机 I/O 不一定更快
- 失去 page cache 带来的重用收益
- 与普通 buffered I/O 混用时一致性更复杂

### 12.6 `mmap` 与 O_DIRECT 混用为什么危险

如果同一文件既通过 `mmap` / buffered I/O 走 page cache，又通过 O_DIRECT 绕开 cache，那么你必须非常清楚：

- 哪条路径先看到修改
- cache 中是否还有旧页
- direct I/O 是否绕开了你以为存在的缓存一致性保证
- 应用是否自己承担了刷写、屏障和失效协议

很多数据库和存储引擎因此会明确约束：某类文件只走一种路径。

### 12.7 `rename + fsync(file) + fsync(dir)` 为什么是协议

安全写入一个新版本文件时，常见协议是：

1. 写临时文件
2. `fsync` 临时文件
3. `rename` 为目标名字
4. `fsync` 父目录

这不是迷信配方，而是分别补这些缺口：

- 临时文件 `fsync`：内容和必要元数据可靠化
- `rename`：目录项切换原子可见
- 目录 `fsync`：新名字关系持久化

少掉其中一步，不一定平时出错，但在崩溃测试中就可能暴露漏洞。

### 12.8 `sync_file_range`、`io_uring` 这类接口为什么属于更高阶话题

进一步的 I/O 接口如 `sync_file_range`、`io_uring` 不只是“更快 API”，而是在让你更细粒度控制：

- 提交时机
- 等待时机
- 批量化
- 异步完成
- 与 page cache / direct I/O 的交互

但控制力越强，程序越需要自己理解语义边界。否则“性能优化”很容易变成“把恢复保证悄悄删掉”。

### 12.9 目录 `fsync` 常被忽略，是因为大家盯错了对象层

很多工程师愿意 `fsync(file)`，却忘了目录对象本身也发生了变化。创建、删除、rename，改变的不是文件内容，而是父目录的名字映射。

这再次说明：

- 文件内容层和目录项层不是同一层
- 更新协议如果只覆盖文件，不覆盖目录，仍可能在崩溃后丢名字

### 12.10 一个选择框架：我该用哪条 I/O 路径

可以先问：

1. 我的负载是顺序还是随机？
2. 我是否已有自己的缓存层？
3. 我更怕 cache 污染、尾延迟、吞吐不稳，还是恢复语义缺口？
4. 我是否能正确处理 `fsync`、`msync`、目录持久化和错误传播？
5. 我是否真的需要 O_DIRECT / `mmap`，还是只是以为“更底层就更快”？

高阶接口不是奖励，而是责任更重。

### 12.11 writeback error 为什么经常“很晚才出现”

很多程序做了大量 `write()`，过程中一切顺利，于是就默认“写入成功”。真正危险的地方在于：

- 某些底层错误可能发生在更晚的 writeback 阶段
- 应用第一次感知错误，可能不是在对应那次 `write()` 上
- 如果程序从不认真检查 `fsync()`、`close()`、日志或错误码，就可能把失败误当成成功

这类问题尤其容易出现在：

- 设备空间/介质错误
- 远端文件系统的异步回写失败
- 应用自己把“写进 page cache”误当成“稳定提交”

所以从协议角度看：

- `write()` 更像“提交给内核缓存/路径”
- `fsync()` / `fdatasync()` 更接近“逼迫错误浮出水面”
- `close()` 在很多场景里仍然不应该被当成可靠持久化确认点

### 12.12 `syncfs()`、批量持久化与“单文件视角”之外的边界

`fsync(fd)` 是单文件视角，而 `syncfs()` 更接近“把某个挂载上的脏数据整体向稳定边界推进”。

这提醒我们两件事：

- 文件系统语义不只属于单个文件，还属于挂载实例和回写域
- 当你做 checkpoint、快照前刷写、批量提交或故障注入测试时，单纯盯一个 fd 往往不够

但 `syncfs()` 也不是万能提交协议。它更适合：

- 缩小脏数据窗口
- 在系统级测试里把异步回写显式推进
- 辅助观察“当前挂载点上究竟还有多少未完成状态”

它不自动替代：

- 应用对版本号、manifest、目录项切换的控制
- 多文件事务语义

### 12.13 DAX 为什么不是“更猛的 O_DIRECT”

DAX（Direct Access）常被误解成“既然都绕过 page cache，那它和 O_DIRECT 差不多”。实际上它关注的是另一类路径：

- 针对支持持久内存/pmem 这类介质的直接映射访问
- 尽量绕过 page cache，并减少传统块 I/O 路径中的拷贝和页缓存管理

它和 O_DIRECT 的差异在于：

- O_DIRECT 主要是在传统块设备语义下绕开 page cache
- DAX 则进一步改变了文件页、页缓存和地址空间交互方式
- DAX 下你更要明确 cache line flush、持久化屏障和失败恢复边界

对大多数应用来说，理解 DAX 的价值不在于马上使用它，而在于知道：

- “绕过 page cache”并不是单一技术
- 越接近介质，应用越要亲自承担语义责任

### 12.14 常见负载的 I/O 语义选择表

| 场景 | 常见选择 | 主要收益 | 最容易漏掉的风险 |
|------|----------|----------|------------------|
| 普通应用配置/小文件更新 | buffered I/O + `fsync(temp)` + `rename` + `fsync(dir)` | 简单、兼容、语义可控 | 忘记目录 `fsync`，把 `close()` 当提交 |
| 数据库/自带 buffer pool | O_DIRECT 或受控 buffered I/O | 减少 page cache 污染，更好控制回写 | 混用路径导致一致性复杂，错误传播处理不足 |
| 大文件随机访问 | `mmap` 或显式 I/O | 编程模型自然，局部访问高效 | `truncate`、fault、并发修改导致边界复杂 |
| pmem / 持久内存场景 | DAX | 更低层访问介质 | 持久化屏障和恢复协议更难，调试成本高 |

真正的高阶选择不是“用最底层的”，而是：

**让接口形态和你的恢复协议、观测能力、团队心智成本匹配。**

---

### 12.15 `struct vm_area_struct`：mmap 的内核数据结构

每次 `mmap()` 调用在内核中创建一个 `struct vm_area_struct`（VMA），描述一段虚拟地址区间。

```c
/* include/linux/mm_types.h */
struct vm_area_struct {
    /* 虚拟地址区间边界 */
    unsigned long   vm_start;       /* 区间起始地址（包含）*/
    unsigned long   vm_end;         /* 区间结束地址（不包含）*/

    /* 链表与树结构（进程地址空间组织）*/
    struct vm_area_struct *vm_next, *vm_prev;  /* 双向链表 */
    struct rb_node  vm_rb;          /* 红黑树节点（快速按地址查找）*/

    /* 与文件的关联 */
    struct file     *vm_file;       /* 映射的文件（匿名映射为 NULL）*/
    unsigned long   vm_pgoff;       /* 映射在文件中的起始偏移（页为单位）*/
    struct address_space *vm_private_data; /* 文件系统私有数据 */

    /* 权限与标志 */
    pgprot_t        vm_page_prot;   /* 页表项保护位（读/写/执行）*/
    unsigned long   vm_flags;       /* VMA 标志（见下）*/

    /* 操作函数表 */
    const struct vm_operations_struct *vm_ops;

    /* 匿名映射链表（共享匿名页追踪）*/
    struct list_head anon_vma_chain;
    struct anon_vma  *anon_vma;     /* 匿名页的反向映射根 */

    /* 与进程 mm 的关联 */
    struct mm_struct *vm_mm;        /* 所属进程的内存描述符 */
    struct mempolicy *vm_policy;    /* NUMA 内存策略 */
};

/* vm_flags 常用标志 */
#define VM_READ     0x00000001      /* 可读 */
#define VM_WRITE    0x00000002      /* 可写 */
#define VM_EXEC     0x00000004      /* 可执行 */
#define VM_SHARED   0x00000008      /* MAP_SHARED（修改影响文件）*/
#define VM_MAYWRITE 0x00000020      /* mprotect 可以升级为可写 */
#define VM_GROWSDOWN 0x00000100     /* 可向下增长（栈 VMA）*/
#define VM_PFNMAP   0x00000400      /* 物理页帧映射（无 struct page）*/
#define VM_DONTEXPAND 0x00040000    /* 不允许通过 mremap 扩展 */
#define VM_LOCKED   0x00002000      /* mlock：页必须常驻内存 */
#define VM_HUGETLB  0x00400000      /* huge TLB 映射 */
#define VM_MIXEDMAP 0x10000000      /* 混合普通页和 pfnmap */
#define VM_DONTDUMP 0x04000000      /* core dump 时跳过 */
```

**`struct vm_operations_struct`：VMA 的操作表**：

```c
struct vm_operations_struct {
    void   (*open)(struct vm_area_struct *);
        /* VMA 被 fork/mremap 复制时调用（如增加引用计数）*/

    void   (*close)(struct vm_area_struct *);
        /* VMA 被 munmap/exit 销毁时调用 */

    vm_fault_t (*fault)(struct vm_fault *vmf);
        /* 最核心：缺页时调用，负责找到/装入物理页 */
        /* ext4/xfs 实现通常是 filemap_fault（返回 page cache 中的页）*/
        /* 设备驱动可实现自己的 fault（如 GPU 内存）*/

    vm_fault_t (*huge_fault)(struct vm_fault *vmf, enum page_entry_size pe_size);
        /* Huge page fault 处理 */

    int    (*mmap)(struct file *, struct vm_area_struct *);
        /* 文件系统在 mmap 建立 VMA 时的初始化钩子 */

    void   (*page_mkwrite)(struct vm_fault *vmf);
        /* COW 页即将被写入时调用（ext4 journal 在此处理 ordered mode 语义）*/
        /* ext4_page_mkwrite → jbd2_journal_start → mark_inode_dirty */

    void   (*pfn_mkwrite)(struct vm_fault *vmf);
        /* 物理页帧映射的写权限升级 */

    int    (*access)(struct vm_area_struct *, unsigned long, void *, int, int);
        /* ptrace/proc 访问此 VMA 的内容（/proc/pid/mem 读写）*/

    const char *(*name)(struct vm_area_struct *);
        /* /proc/pid/maps 中显示的区间名字 */
};
```

**观察进程 VMA 布局**：

```bash
# /proc/pid/maps：查看进程 mmap 布局
cat /proc/$$/maps
# 7f1234560000-7f1234561000 r--p 00000000 08:01 1234567  /lib/x86_64-linux-gnu/libm.so.2
# ↑ vm_start         ↑ vm_end
#                     ↑权限: r=读, w=写, x=执行, s=共享/p=私有
#                           ↑ 文件偏移（页为单位），0 = 文件开头
#                                 ↑ 设备 major:minor    ↑ inode号  ↑ 文件路径

# /proc/pid/smaps：更详细，包含每个 VMA 的内存统计
cat /proc/$$/smaps | head -40
# 7f1234560000-7f1234561000 r--p 00000000 08:01 1234567 /lib/x86_64-linux-gnu/libm.so.2
# Size:                  4 kB    ← VMA 总大小（含未分配）
# KernelPageSize:        4 kB    ← 内核页大小
# MMUPageSize:           4 kB    ← MMU 实际使用页大小
# Rss:                   4 kB    ← 常驻内存（已建立页表且在物理内存中）
# Pss:                   2 kB    ← 按比例分摊的常驻（共享库按共享进程数分摊）
# Shared_Clean:          4 kB    ← 共享干净页（未修改，多进程共享物理页）
# Shared_Dirty:          0 kB    ← 共享脏页（修改过）
# Private_Clean:         0 kB    ← 私有干净页（COW 后未修改）
# Private_Dirty:         0 kB    ← 私有脏页（COW 后已修改）
# Referenced:            4 kB    ← 近期被访问过（用于内存回收判断）
# Anonymous:             0 kB    ← 匿名页（无文件后备）
# LazyFree:              0 kB    ← madvise(MADV_FREE) 标记的待回收页
# AnonHugePages:         0 kB    ← 透明大页（THP）
# ShmemPmdMapped:        0 kB    ← 共享内存大页
# FilePmdMapped:         0 kB    ← 文件页大页
# Shared_Hugetlb:        0 kB
# Private_Hugetlb:       0 kB
# Swap:                  0 kB    ← 已换出到 swap 的匿名页
# SwapPss:               0 kB
# Locked:                0 kB    ← mlock 锁定的页（不可换出）
# THPeligible:           1       ← 是否满足 THP 合并条件

# smaps_rollup：所有 VMA 的聚合统计（更快）
cat /proc/$$/smaps_rollup
# Rss:               45678 kB   ← 进程总常驻内存
# Pss:               23456 kB   ← 按比例分摊后的真实内存占用

# /proc/pid/pagemap：每个虚拟页的物理帧映射（需要 root）
python3 -c "
import struct, os

pid = os.getpid()
# 读取 /proc/pid/maps 找到一个 VMA
with open(f'/proc/{pid}/maps') as f:
    for line in f:
        if '[heap]' in line:
            parts = line.split()
            start, end = [int(x, 16) for x in parts[0].split('-')]
            break

# pagemap 中每个虚拟页对应 8 字节
PAGE_SIZE = 4096
pagemap_offset = (start // PAGE_SIZE) * 8

with open(f'/proc/{pid}/pagemap', 'rb') as pm:
    pm.seek(pagemap_offset)
    entry = struct.unpack('Q', pm.read(8))[0]  # 8字节 uint64

# 解析 pagemap 条目
pfn = entry & 0x7fffffffffffff        # bits 0-54: 物理页帧号
soft_dirty = (entry >> 55) & 1        # bit 55: 软脏位
exclusive = (entry >> 56) & 1         # bit 56: 页是否唯一映射
file_or_shared = (entry >> 61) & 1    # bit 61: file-backed 或 shared-anon
swapped = (entry >> 62) & 1           # bit 62: 是否在 swap 中
present = (entry >> 63) & 1           # bit 63: 物理页是否存在于内存

print(f'heap start: 0x{start:x}')
print(f'pfn: {pfn}, present: {present}, file_backed: {file_or_shared}')
print(f'physical addr: 0x{pfn * PAGE_SIZE:x}')
"
```

---

### 12.16 page fault 处理路径：从硬件异常到文件页装入

当 CPU 访问 `mmap` 区域中未建立页表的地址时，触发缺页异常（`#PF`），陷入内核：

```
page fault 处理栈（x86_64，Linux 6.x）：

CPU #PF 异常
  └─ do_page_fault()                    [arch/x86/mm/fault.c]
        └─ handle_page_fault()
              └─ __do_page_fault()
                    ├─ find_vma(mm, address)      ← 在红黑树中找到地址对应的 VMA
                    │   如果没找到 → SIGSEGV
                    │
                    └─ handle_mm_fault(vma, address, flags, regs)
                                                   [mm/memory.c]
                          ├─ 检查 vm_flags 与 fault 类型是否匹配
                          │   （写一个只读 VMA → SIGSEGV）
                          │
                          ├─ pgd → p4d → pud → pmd → pte 逐级页表遍历
                          │
                          └─ handle_pte_fault(vmf)
                                ├─ 情况 1：pte 为空（从未访问）
                                │   → do_fault(vmf)
                                │       ├─ 只读：do_read_fault()
                                │       │       → vma->vm_ops->fault(vmf)
                                │       │           ← ext4: filemap_fault()
                                │       │               → find_get_page() 查 page cache
                                │       │               → 若命中：直接建页表（minor fault）
                                │       │               → 若未命中：
                                │       │                   → filemap_read_page()
                                │       │                       → a_ops->read_folio()
                                │       │                           → ext4_read_folio()
                                │       │                               → 提交 bio，等待磁盘 I/O
                                │       │                   → 页装入后：建页表（major fault）
                                │       │
                                │       └─ 写时复制：do_cow_fault()
                                │               → 分配新匿名页
                                │               → 拷贝内容（copy_user_highpage）
                                │               → 调用 vma->vm_ops->page_mkwrite() ← ext4 日志
                                │               → 建写权限页表
                                │
                                ├─ 情况 2：pte 指向 swap entry
                                │   → do_swap_page()（从 swap 设备读回）
                                │
                                └─ 情况 3：pte 有效但无写权限（COW）
                                    → wp_page_copy()
                                        → 分配新页，拷贝，更新 pte
```

**ext4 的 `filemap_fault` 实现细节**：

```c
/* mm/filemap.c — filemap_fault 简化版逻辑 */
vm_fault_t filemap_fault(struct vm_fault *vmf)
{
    struct file *file = vmf->vma->vm_file;
    struct address_space *mapping = file->f_mapping;
    struct inode *inode = mapping->host;
    pgoff_t offset = vmf->pgoff;    /* 文件内页偏移 */
    struct page *page;
    vm_fault_t ret;

    /* 步骤 1：查 page cache */
    page = find_get_page(mapping, offset);

    if (!page) {
        /* page cache 未命中：触发磁盘 I/O（major fault）*/
        count_vm_event(PGMAJFAULT);
        ret = VM_FAULT_MAJOR;

        /* 触发 readahead：不只读一页，预读后续页 */
        page = do_async_mmap_readahead(vmf, page);
        if (!page) {
            /* readahead 也没有：同步读取 */
            page = filemap_alloc_folio(GFP_KERNEL, 0);
            /* 提交 bio，等待 I/O 完成 */
            error = mapping->a_ops->read_folio(file, page_folio(page));
        }
    } else {
        /* page cache 命中：minor fault，只需建页表 */
        count_vm_event(PGMINORSFAULT);
        ret = 0;
    }

    /* 步骤 2：把物理页映射到进程虚拟地址 */
    vmf->page = page;
    return ret | VM_FAULT_LOCKED;
}
```

**观察 page fault 类型**：

```bash
# perf stat 统计 minor/major fault 数量
perf stat -e minor-faults,major-faults ./your_program
# 或
perf stat -e page-faults,major-faults dd if=/dev/zero of=/tmp/test bs=4K count=10000

# 通过 /proc/pid/stat 累积统计
awk '{print "minor_faults:", $10, "major_faults:", $12}' /proc/$$/stat

# 实时监控 page fault 率（vmstat）
vmstat 1
# r  b   swpd   free   buff  cache   si  so    bi    bo   in   cs  us  sy  id  wa
# 1  0      0  12345   1234  56789    0   0   256     0  1024 2048  5   3  92   0
# ↑                                           bi = blocks read in（major fault 引发 I/O）

# 用 bpftrace 追踪 page fault 事件
bpftrace -e '
software:page-faults:1 {
    @[comm] = count();
}
interval:s:5 {
    print(@); clear(@);
}'

# 追踪 major fault（磁盘 I/O 的 page fault）
bpftrace -e '
kprobe:do_swap_page,
kprobe:do_read_fault {
    printf("major fault: %s pid=%d addr=0x%lx\n",
           func, pid, ((struct vm_fault*)arg0)->address);
}'
```

---

### 12.17 O_DIRECT 的对齐要求与 DMA 路径

O_DIRECT 绕过 page cache，数据从用户缓冲区直接通过 DMA 传输到设备。这要求：

```
O_DIRECT 三个对齐约束（内核在 generic_file_direct_write 中检查）：
  1. 文件偏移（offset）必须是 logical_block_size 的整数倍
  2. 用户缓冲区地址必须是 logical_block_size 的整数倍
  3. 传输长度（len）必须是 logical_block_size 的整数倍

其中 logical_block_size 通常是底层块设备的扇区大小：
  - 传统 HDD/SSD：512 字节（但现代设备大多是 4096 字节 Physical Block Size）
  - NVMe：通常 512 字节逻辑扇区，但实际对齐建议 4096 字节（内存分配对齐更好）
```

**查询设备对齐要求**：

```bash
# 查看设备的逻辑/物理扇区大小
cat /sys/block/sda/queue/logical_block_size    # 通常 512
cat /sys/block/sda/queue/physical_block_size   # 通常 512 或 4096
cat /sys/block/nvme0n1/queue/logical_block_size
cat /sys/block/nvme0n1/queue/physical_block_size

# 用 blockdev 查询
blockdev --getss /dev/sda   # 逻辑扇区大小
blockdev --getpbsz /dev/sda # 物理扇区大小
blockdev --getbsz /dev/sda  # 文件系统块大小（ext4 通常 4096）

# O_DIRECT 的对齐违规会返回 EINVAL
python3 -c "
import os, mmap, ctypes, errno

path = '/tmp/direct_test'

# 创建测试文件
with open(path, 'wb') as f:
    f.write(b'x' * 8192)

fd = os.open(path, os.O_RDONLY | os.O_DIRECT)

# 方式 1：posix_memalign 分配对齐缓冲区（正确方式）
libc = ctypes.CDLL('libc.so.6', use_errno=True)
buf = ctypes.create_string_buffer(4096)
# 确保地址 4096 字节对齐
aligned_buf = ctypes.create_string_buffer(8192)
aligned_ptr = (ctypes.addressof(aligned_buf) + 4095) & ~4095

try:
    n = os.read(fd, 4096)   # 读到非对齐缓冲区 → EINVAL
    print(f'read ok: {n} bytes')
except OSError as e:
    print(f'EINVAL: {e}')    # errno 22

os.close(fd)

# 方式 2：使用 mmap 分配保证对齐的缓冲区（常见做法）
fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
buf = mmap.mmap(-1, 4096)   # 匿名 mmap，page 对齐
ptr = ctypes.c_char_p(ctypes.addressof(ctypes.c_char.from_buffer(buf)))
# pread 直接读到这个对齐缓冲区
n = os.pread(fd, 4096, 0)
print(f'O_DIRECT aligned read: {n} bytes')
os.close(fd)
os.unlink(path)
"
```

**O_DIRECT 的内核执行路径**：

```
write(fd_direct, buf, 4096) with O_DIRECT:

  vfs_write()
    └─ generic_file_write_iter()
          └─ 检测到 O_DIRECT → direct_write = true
          └─ generic_file_direct_write()
                ├─ iov_iter_alignment() 检查对齐（违规 → EINVAL）
                ├─ mapping->a_ops->direct_IO()
                │       ↓ ext4 实现
                │   ext4_direct_IO()
                │     └─ __blockdev_direct_IO()
                │           ├─ 把用户 iov 转换为 bio（无需 page cache 页）
                │           │   bio->bi_io_vec 直接指向用户内存
                │           │   （需要 get_user_pages pin 住内存不被换出）
                │           ├─ submit_bio(bio)  → 提交到块设备层
                │           └─ 等待 bio 完成（同步 O_DIRECT），或异步（io_uring）
                │
                └─ 若文件 page cache 中有重叠的脏页：
                    invalidate_inode_pages2_range() ← 使脏页失效，保证一致性
                    （这是 O_DIRECT 和 buffered I/O 混用危险的根源）

性能对比（4KB 随机读，NVMe SSD）：
  buffered I/O（第一次，冷 cache）：~150μs  ← 含 page cache 分配开销
  buffered I/O（第二次，热 cache）：~1μs   ← 完全命中 page cache
  O_DIRECT（每次）：               ~80μs   ← 无 cache，但也无 cache 分配开销
  → O_DIRECT 只在特定场景（如数据库自管 buffer pool）才有净收益
```

---

### 12.18 `io_uring`：异步 I/O 的现代内核接口

`io_uring`（Linux 5.1+）是对 `aio`（POSIX AIO）的彻底重写，通过共享内存环形队列实现零拷贝的批量异步 I/O 提交。

**核心数据结构**：

```c
/* include/uapi/linux/io_uring.h */

/* 提交队列项（SQE）：用户填写，内核消费 */
struct io_uring_sqe {
    __u8    opcode;         /* IORING_OP_READ, IORING_OP_WRITE, IORING_OP_FSYNC 等 */
    __u8    flags;          /* IOSQE_FIXED_FILE, IOSQE_IO_DRAIN 等 */
    __u16   ioprio;         /* I/O 优先级 */
    __s32   fd;             /* 文件描述符（或 fixed file table index）*/
    union {
        __u64   off;        /* 文件偏移（-1 = 当前位置）*/
        __u64   addr2;      /* IORING_OP_PROVIDE_BUFFERS 用 */
    };
    union {
        __u64   addr;       /* 数据缓冲区用户地址 */
        __u64   splice_off_in;
    };
    __u32   len;            /* 字节数 */
    union {
        __kernel_rwf_t  rw_flags;    /* pread/pwrite flags（O_DSYNC 等）*/
        __u32   fsync_flags;         /* IORING_FSYNC_DATASYNC */
        __u16   poll_events;         /* poll 事件掩码 */
        __u32   sync_range_flags;    /* sync_file_range flags */
        __u32   msg_flags;           /* sendmsg/recvmsg flags */
        __u32   timeout_flags;       /* IORING_TIMEOUT_ABS 等 */
        __u32   accept_flags;
        __u32   cancel_flags;
        __u32   open_flags;
        __u32   statx_flags;
        __u32   fadvise_advice;
        __u32   splice_flags;
    };
    __u64   user_data;      /* 用户透传 cookie（在 CQE 中返回）*/
    union {
        __u16   buf_index;  /* 预注册 buffer 的 index */
        __u16   buf_group;
    };
    __u16   personality;    /* 执行操作时使用的 credentials id */
    union {
        __s32   splice_fd_in;
        __u32   file_index;  /* IORING_OP_OPENAT 注册的 fixed file index */
    };
    __u64   addr3;
    __u64   __pad2[1];
};

/* 完成队列项（CQE）：内核填写，用户消费 */
struct io_uring_cqe {
    __u64   user_data;  /* 与 SQE 的 user_data 对应（用于匹配请求）*/
    __s32   res;        /* 操作结果（成功 = 读写字节数；失败 = -errno）*/
    __u32   flags;      /* IORING_CQE_F_BUFFER（buffer 选择）等 */
};

/* 主要操作码（opcode）*/
/* IORING_OP_NOP          (0)  无操作（测试延迟用）*/
/* IORING_OP_READV        (1)  readv（scatter-gather 读）*/
/* IORING_OP_WRITEV       (2)  writev（scatter-gather 写）*/
/* IORING_OP_FSYNC        (4)  fsync */
/* IORING_OP_READ_FIXED   (5)  读到预注册缓冲区（零拷贝）*/
/* IORING_OP_WRITE_FIXED  (6)  从预注册缓冲区写（零拷贝）*/
/* IORING_OP_POLL_ADD     (7)  poll fd */
/* IORING_OP_SENDMSG     (10)  sendmsg */
/* IORING_OP_RECVMSG     (11)  recvmsg */
/* IORING_OP_TIMEOUT     (12)  超时 */
/* IORING_OP_ACCEPT      (14)  accept */
/* IORING_OP_CONNECT     (16)  connect */
/* IORING_OP_OPENAT      (18)  openat */
/* IORING_OP_CLOSE       (19)  close */
/* IORING_OP_STATX       (21)  statx */
/* IORING_OP_READ        (22)  read（简化版 readv，单缓冲区）*/
/* IORING_OP_WRITE       (23)  write（简化版 writev）*/
/* IORING_OP_SEND        (26)  send */
/* IORING_OP_RECV        (27)  recv */
/* IORING_OP_SPLICE      (29)  splice（zero-copy 管道传输）*/
/* IORING_OP_PROVIDE_BUFFERS (31)  向内核注册 buffer 池 */
/* IORING_OP_CANCEL      (36)  取消已提交的操作 */
/* IORING_OP_LINK_TIMEOUT(37)  有限时间的链式操作 */
/* IORING_OP_SOCKET      (53)  socket */
/* IORING_OP_URING_CMD   (54)  NVMe passthrough 等设备命令 */
```

**io_uring 的工作原理**：

```
io_uring 架构（io_uring_setup 返回两个 mmap 区域）：

用户态                           内核态
┌──────────────────────────┐    ┌──────────────────────────┐
│  SQ Ring（提交环）        │    │  内核 io_ring_ctx         │
│  ┌────┬────┬────┬────┐   │    │  ┌──────────────────┐   │
│  │SQE │SQE │SQE │SQE │   │◄──►│  │  io_wq 工作队列  │   │
│  └────┴────┴────┴────┘   │    │  │  （异步线程池）   │   │
│  sq_head    sq_tail      │    │  └──────────────────┘   │
│                          │    │                          │
│  CQ Ring（完成环）        │    │  ┌──────────────────┐   │
│  ┌────┬────┬────┬────┐   │◄──►│  │  固定文件表      │   │
│  │CQE │CQE │CQE │CQE │   │    │  │（避免 fd 查找）  │   │
│  └────┴────┴────┴────┘   │    │  └──────────────────┘   │
│  cq_head    cq_tail      │    │                          │
└──────────────────────────┘    └──────────────────────────┘
           ↑ mmap 共享内存（用户和内核都能直接读写，无系统调用）

提交流程：
  1. 用户在 SQ Ring 中填写 SQE（无系统调用）
  2. 更新 sq_tail（原子操作，仍无系统调用）
  3. io_uring_enter(ring_fd, to_submit, min_complete, flags)
        → 若用 SQPOLL 模式，内核线程轮询 SQ，连 3 都不需要
  4. 内核处理 SQE（可能立即完成，或由 io_wq 异步处理）
  5. 内核在 CQ Ring 中写入 CQE，更新 cq_tail
  6. 用户轮询 cq_head != cq_tail，收割 CQE（无系统调用）
```

**使用 liburing 的示例**：

```c
/* 需要安装 liburing: apt install liburing-dev */
#include <liburing.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define QUEUE_DEPTH 64
#define BLOCK_SIZE  4096

int main(void)
{
    struct io_uring ring;
    /* 初始化：创建 SQ/CQ 环形队列，各 QUEUE_DEPTH 项 */
    io_uring_queue_init(QUEUE_DEPTH, &ring, 0);

    int fd = open("/tmp/uring_test", O_RDWR | O_CREAT | O_TRUNC, 0644);
    char *buf = aligned_alloc(4096, BLOCK_SIZE);  /* 对齐分配（O_DIRECT 要求）*/

    /* === 批量提交 8 个写操作 === */
    for (int i = 0; i < 8; i++) {
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        memset(buf, 'A' + i, BLOCK_SIZE);
        io_uring_prep_write(sqe, fd, buf, BLOCK_SIZE, (off_t)i * BLOCK_SIZE);
        sqe->user_data = i;    /* 标记请求 ID */
    }
    io_uring_submit(&ring);    /* 一次系统调用提交所有 8 个 */

    /* === 收割完成事件 === */
    for (int i = 0; i < 8; i++) {
        struct io_uring_cqe *cqe;
        io_uring_wait_cqe(&ring, &cqe);  /* 等待一个完成 */

        if (cqe->res < 0) {
            fprintf(stderr, "写入失败 id=%llu: %s\n",
                    cqe->user_data, strerror(-cqe->res));
        } else {
            printf("写入完成 id=%llu: %d bytes\n", cqe->user_data, cqe->res);
        }
        io_uring_cqe_seen(&ring, cqe);  /* 消费这个 CQE */
    }

    /* === 链式操作：write → fsync（保证顺序）*/
    struct io_uring_sqe *write_sqe = io_uring_get_sqe(&ring);
    io_uring_prep_write(write_sqe, fd, buf, BLOCK_SIZE, 0);
    write_sqe->flags |= IOSQE_IO_LINK;  /* 链到下一个操作 */

    struct io_uring_sqe *fsync_sqe = io_uring_get_sqe(&ring);
    io_uring_prep_fsync(fsync_sqe, fd, 0);  /* 在 write 完成后执行 */
    io_uring_submit(&ring);

    /* 等待两个完成 */
    struct io_uring_cqe *cqes[2];
    io_uring_wait_cqe_nr(&ring, cqes, 2, 0);

    io_uring_queue_exit(&ring);
    close(fd);
    free(buf);
    return 0;
}
/* 编译：gcc -o uring_test uring_test.c -luring */
```

**观察 io_uring 性能**：

```bash
# 用 fio 测试 io_uring vs libaio vs sync 的性能差异
fio --name=uring_test --ioengine=io_uring --iodepth=64 \
    --rw=randread --bs=4k --size=1G --filename=/tmp/fio_test \
    --runtime=30 --time_based --group_reporting

fio --name=sync_test --ioengine=sync --iodepth=1 \
    --rw=randread --bs=4k --size=1G --filename=/tmp/fio_test \
    --runtime=30 --time_based --group_reporting

# 典型结果（NVMe SSD）：
# io_uring iodepth=64：IOPS=450K, lat avg=142μs
# libaio  iodepth=64：IOPS=420K, lat avg=152μs
# sync    iodepth=1 ：IOPS=120K, lat avg=8μs（单线程，无并发深度）

# 观察 io_uring 内核线程（SQPOLL 模式）
ps aux | grep "iou-"
# root   1234  0.5  0.0  0  0 ?  S  00:00  0:30 [iou-sqp-1234]
# ↑ SQPOLL 内核轮询线程（以 fd 持有者的 uid 运行）

# 追踪 io_uring 操作（bpftrace）
bpftrace -e '
tracepoint:io_uring:io_uring_submit_sqe {
    printf("submit: op=%d fd=%d user_data=%lu\n", args->op, args->fd, args->user_data);
}
tracepoint:io_uring:io_uring_complete {
    printf("complete: user_data=%lu res=%d\n", args->user_data, args->res);
}'
```

---

### 12.19 `msync` 与 `fsync` 的内核实现差异

`msync` 和 `fsync` 最终都依赖 page cache 的回写机制，但切入点不同：

```
msync(addr, len, MS_SYNC) 的内核路径：

  msync(addr, len, flags)
    └─ do_msync()                       [mm/msync.c]
          ├─ find_vma(mm, addr)         ← 找到对应 VMA
          ├─ vma_interval_tree_foreach(vma, ...)  ← 遍历重叠的 VMA
          └─ vfs_fsync_range(file, start, end, 0) [fs/sync.c]
                └─ file->f_op->fsync(file, start, end, datasync)
                      ↓ ext4 实现
                   ext4_sync_file()
                     ├─ 若 data=journal：jbd2_log_wait_commit()
                     ├─ filemap_write_and_wait_range(mapping, start, end)
                     │       → 把 [start, end] 范围内的脏页提交回写
                     │       → 等待这些页的 I/O 完成
                     └─ 发送 flush/FUA（确保到达稳定存储）

MS_ASYNC vs MS_SYNC vs MS_INVALIDATE：
  MS_ASYNC：只调度回写，不等待（返回前可能页还是脏的）
  MS_SYNC ：调度回写并等待完成（返回时数据已落盘）
  MS_INVALIDATE：使 mmap 区域无效，强制下次访问重新从文件读取
               （用于文件被其他进程修改时的视图刷新）
```

```
fsync(fd) 的内核路径：

  fsync(fd)
    └─ do_fsync(fd, 0)                  [fs/sync.c]
          └─ vfs_fsync(file, 0)
                └─ file->f_op->fsync(file, 0, LLONG_MAX, 0)
                      ↓ ext4
                   ext4_sync_file()
                     ├─ writeback_inodes_sb_if_idle(sb)（有脏 inode 时）
                     ├─ filemap_write_and_wait(mapping)   ← 全文件回写
                     ├─ ext4_flush_completed_IO(inode)    ← dio 完成确认
                     ├─ jbd2_complete_transaction(journal, commit_tid)
                     │       ← 等待包含此 inode 修改的事务完成 commit
                     └─ blkdev_issue_flush(bdev)          ← 设备 cache 刷新

fdatasync(fd) 的内核路径：
  do_fsync(fd, 1)   ← datasync=1
    └─ vfs_fsync(file, 1)
          └─ ext4_sync_file(file, 0, LLONG_MAX, 1)
                ↑ datasync=1: 跳过 inode 的 atime/mtime 等非关键元数据
                ← 只刷文件大小变化（如 append 后的 i_size）才是必须的
                ← 对 no-op write（覆盖现有内容，不改变大小）可以跳过 inode 回写
```

**实际测量 fsync 和 fdatasync 延迟差异**：

```bash
python3 - <<'EOF'
import os, time, statistics, random

path = '/tmp/sync_bench'
SIZE = 4096

# 创建测试文件
with open(path, 'wb') as f:
    f.write(b'\x00' * SIZE * 1000)

# 测试 1: fsync（含完整元数据更新）
fd = os.open(path, os.O_WRONLY)
delays_fsync = []
for i in range(100):
    os.lseek(fd, i * SIZE, os.SEEK_SET)
    os.write(fd, b'x' * SIZE)
    t0 = time.monotonic()
    os.fsync(fd)
    delays_fsync.append((time.monotonic() - t0) * 1000)
os.close(fd)

# 测试 2: fdatasync（仅数据相关元数据）
fd = os.open(path, os.O_WRONLY)
delays_fds = []
for i in range(100):
    os.lseek(fd, i * SIZE, os.SEEK_SET)
    os.write(fd, b'y' * SIZE)
    t0 = time.monotonic()
    os.fdatasync(fd)
    delays_fds.append((time.monotonic() - t0) * 1000)
os.close(fd)

# 测试 3: msync（mmap 后修改）
import mmap
fd = os.open(path, os.O_RDWR)
m = mmap.mmap(fd, SIZE * 100)
delays_msync = []
for i in range(100):
    m[i * SIZE:(i+1) * SIZE] = b'z' * SIZE
    t0 = time.monotonic()
    m.flush(i * SIZE, SIZE)  # msync 对应的 Python 接口
    delays_msync.append((time.monotonic() - t0) * 1000)
m.close()
os.close(fd)

print(f"fsync:     avg={statistics.mean(delays_fsync):.2f}ms  "
      f"p99={sorted(delays_fsync)[98]:.2f}ms  "
      f"max={max(delays_fsync):.2f}ms")
print(f"fdatasync: avg={statistics.mean(delays_fds):.2f}ms  "
      f"p99={sorted(delays_fds)[98]:.2f}ms  "
      f"max={max(delays_fds):.2f}ms")
print(f"msync:     avg={statistics.mean(delays_msync):.2f}ms  "
      f"p99={sorted(delays_msync)[98]:.2f}ms  "
      f"max={max(delays_msync):.2f}ms")

os.unlink(path)
EOF
# 典型 ext4/NVMe 结果：
# fsync:     avg=0.82ms  p99=1.23ms  max=4.56ms
# fdatasync: avg=0.71ms  p99=1.10ms  max=4.12ms   ← 略快于 fsync（跳过部分元数据）
# msync:     avg=0.79ms  p99=1.18ms  max=4.34ms   ← 接近 fsync（最终走同一路径）

# strace 验证各接口的系统调用
strace -e trace=fsync,fdatasync,msync python3 -c "
import os, mmap
fd = os.open('/tmp/t', os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
os.ftruncate(fd, 4096)
os.write(fd, b'x' * 4096)
os.fsync(fd)      # → strace: fsync(3) = 0
os.fdatasync(fd)  # → strace: fdatasync(3) = 0
m = mmap.mmap(fd, 4096)
m[0:10] = b'hello.....'
m.flush()         # → strace: msync(0x7f..., 4096, MS_SYNC) = 0
m.close(); os.close(fd); os.unlink('/tmp/t')
"
```

---

### 12.20 mmap 的一致性边界：truncate、sigbus 与并发

mmap 区域有几个容易忽视的一致性陷阱：

```
陷阱 1：truncate 到比 mmap 区域更小的大小

  文件大小 = 4096 bytes
  mmap(fd, length=8192, ...)  ← 映射了文件后 4096 字节的"洞"

  ftruncate(fd, 2048)         ← 文件变小

  此时访问 [2048, 8192) 的映射区域 → SIGBUS（总线错误）！

  原因：内核无法为文件洞之外的地址提供物理页
  防御：mmap 前检查文件大小；或处理 SIGBUS 信号

陷阱 2：MAP_SHARED 下的并发修改竞争

  进程 A: ptr[100] = 1;          ← 修改 page cache 中的页
  进程 B: int x = ptr[100];      ← 可能读到旧值（没有 mfence 保证）
  ← mmap shared 提供文件层的最终可见性，不提供多核 CPU cache 一致性

  正确用法：使用 atomic 操作或 futex 协调并发写入

陷阱 3：mmap + direct I/O 混用导致数据撕裂

  进程 A 通过 mmap 读取地址 0x1000：读到 page cache 中的旧版本
  进程 B 通过 O_DIRECT 写入同一文件偏移：绕过 page cache 写入磁盘

  结果：A 读到 page cache 的旧数据，B 写到磁盘，两者不一致
  内核的处理：O_DIRECT 写入会调用 invalidate_inode_pages2_range()
              使 page cache 中的脏页失效，但存在 TOCTOU 窗口
```

**检测 SIGBUS 和内存访问异常**：

```bash
# 演示 truncate 导致 SIGBUS
python3 -c "
import mmap, os, signal, ctypes

# 创建测试文件
fd = os.open('/tmp/sigbus_test', os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
os.write(fd, b'x' * 4096)  # 文件大小 4096

# 映射 8192 字节（超出文件大小的部分是洞）
m = mmap.mmap(fd, 8192)

# 安装 SIGBUS 处理器
def sigbus_handler(sig, frame):
    print('SIGBUS! 访问了文件洞之外的地址')
    # 真实程序应该在这里 longjmp 或退出
    exit(1)
signal.signal(signal.SIGBUS, sigbus_handler)

# 访问文件内的区域 → OK
print(f'文件内: {m[0]}')  # OK

# 截断到 2048
os.ftruncate(fd, 2048)

# 访问文件外的区域 → SIGBUS
try:
    print(f'文件外: {m[4000]}')  # → SIGBUS
except:
    pass

m.close()
os.close(fd)
os.unlink('/tmp/sigbus_test')
"

# 观察 mmap + COW 的内存行为
python3 -c "
import mmap, os

# 创建测试文件
with open('/tmp/cow_test', 'wb') as f:
    f.write(b'A' * 4096)

# MAP_PRIVATE：COW 语义（写时复制，不影响文件）
fd = os.open('/tmp/cow_test', os.O_RDWR)
m_private = mmap.mmap(fd, 4096, mmap.MAP_PRIVATE)

# MAP_SHARED：写入会影响文件
m_shared = mmap.mmap(fd, 4096, mmap.MAP_SHARED)

m_private[0] = ord('B')  # 写 MAP_PRIVATE → 触发 COW，文件内容不变
m_shared[0] = ord('C')   # 写 MAP_SHARED → 写入 page cache，file 内容变化

m_private.close()
m_shared.close()
os.close(fd)

with open('/tmp/cow_test', 'rb') as f:
    print(f'文件内容: {chr(f.read(1)[0])}')  # 应该是 C（shared 修改的）

os.unlink('/tmp/cow_test')
"
```

---

## 本章小结

| 主题 | 结论 |
|------|------|
| buffered I/O | 是通用默认路径，不是低配版 |
| `mmap` | 把文件页访问变成虚拟内存事件 |
| O_DIRECT | 绕过 page cache，但代价和约束明显 |
| `msync` / `fsync` / `fdatasync` | 工作层次不同，不能混用概念 |
| rename 协议 | 是内容、可见性、目录持久化三层协作 |
| 高阶接口 | 控制力越强，语义责任越重 |
| writeback error | 往往晚于 `write()` 暴露，必须靠同步点和错误检查兜底 |
| DAX / `syncfs()` | 把视角从“单次写调用”扩展到介质路径和挂载级提交边界 |

---

## 练习题

1. `mmap` 为什么不是“把文件整体装入内存”的简单别名？
2. O_DIRECT 适合哪些场景，又为什么常常不适合作为默认路径？
3. 为什么 `fsync(file)` 不能替代目录 `fsync`？
4. `mmap` 与 direct I/O 混用时最危险的边界是什么？
5. 如何判断某次 I/O 优化是不是偷偷牺牲了恢复语义？
6. 为什么有些底层写入失败并不会在对应那次 `write()` 立即暴露？
7. DAX 与 O_DIRECT 都会绕过部分 page cache 路径，但为什么不能把它们当成同一件事？
