# 第13章：链表双指针高级操作

## 13.1 链表双指针的挑战

数组双指针的核心优势——随机访问——在链表中不复存在。链表只能顺序访问，这要求我们在一次遍历中同时收集多个位置的信息。

链表双指针的本质：**用多个指针记住关键节点，避免反复遍历**。

本章系统梳理链表中所有经典双指针技巧，并深入分析边界条件和实现细节。

---

## 13.2 找链表中点（深度分析）

```python
def find_middle(head):
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow
```

**不同终止条件的效果**：

| 终止条件 | 奇数长度（5节点） | 偶数长度（6节点） |
|----------|------------------|------------------|
| `while fast and fast.next` | 返回第3个（正中） | 返回第4个（后中点） |
| `while fast.next and fast.next.next` | 返回第3个（正中） | 返回第3个（前中点） |

```python
# 偶数链表：1→2→3→4→5→6
# 后中点版（常用于链表归并排序的下半部分）
# slow 停在 4

# 前中点版（用于需要保留前半段完整性的场景）
def find_middle_first(head):
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    return slow  # 偶数时返回 3（前中点）
```

**用于链表归并排序**：

```python
def sort_list(head):
    """链表归并排序：O(n log n) 时间，O(log n) 空间（递归栈）"""
    if not head or not head.next:
        return head

    # 找前中点，断开链表
    mid = find_middle_first(head)
    right_head = mid.next
    mid.next = None   # 断开！

    # 递归排序
    left = sort_list(head)
    right = sort_list(right_head)

    # 合并
    return merge(left, right)

def merge(l1, l2):
    dummy = ListNode(0)
    curr = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    curr.next = l1 or l2
    return dummy.next
```

---

## 13.3 Floyd 判圈：环长度和入口（完整推导）

**判断有环**（第4章已介绍）：

```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False
```

**求环的长度**：

```python
def cycle_length(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            # 相遇后继续走，再次相遇时计算步数
            count = 1
            fast = fast.next
            while slow is not fast:
                fast = fast.next
                count += 1
            return count
    return 0
```

**找环入口**（完整数学推导）：

```
设：
  a = 链表头到环入口的距离
  b = 环的长度

第一次相遇时：
  slow 走了 S 步：S = a + x（x 是在环内走的步数）
  fast 走了 F 步：F = a + x + n*b（n 是 fast 多走的圈数）
  由于 F = 2S：
    2(a + x) = a + x + n*b
    a + x = n*b
    a = n*b - x

"从相遇点走 a 步" = "在环内走 n*b - x 步"
                  = "从相遇点沿环走 n*b - x 步"
                  = "回到环入口"（因为环长为 b，走 n*b 正好整圈，再减 x 回到入口）

结论：从 head 和相遇点同时出发，每次各走 1 步，再次相遇处即为环入口。
```

```python
def detect_cycle(head):
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            # 找环入口
            ptr = head
            while ptr is not slow:
                ptr = ptr.next
                slow = slow.next
            return ptr

    return None
```

---

## 13.4 相交链表

**问题**：给定两个链表，找它们的交叉节点。

**朴素思路**：用哈希表记录第一个链表所有节点，遍历第二个链表查找。O(n) 空间。

**双指针优化**（O(1) 空间）：

```python
def get_intersection_node(headA, headB):
    """
    关键洞察：
    设链表 A 长 a+c，链表 B 长 b+c，c 是公共部分长度。
    
    指针 pA 遍历 A 后接着遍历 B：走 a + c + b 步到交点
    指针 pB 遍历 B 后接着遍历 A：走 b + c + a 步到交点
    
    两者步数相同，同时到达交点！
    （若无交点，两者同时到达 None）
    """
    pA, pB = headA, headB

    while pA is not pB:
        pA = pA.next if pA else headB
        pB = pB.next if pB else headA

    return pA  # 交点或 None

# 测试（构建交叉链表）
# A: 4→1→\
#          8→4→5
# B: 5→6→1→/
```

**为什么有效**：
- 若有交点：pA 走 `a + c + b` 步，pB 走 `b + c + a` 步，两者同时到达交点
- 若无交点：pA 走 `a + b` 步，pB 走 `b + a` 步，同时到达 None（`pA is pB is None`）

---

## 13.5 链表反转（多指针经典）

**迭代反转**：

```python
def reverse_list(head):
    prev = None
    curr = head

    while curr:
        next_node = curr.next   # 保存下一个
        curr.next = prev        # 反转指针
        prev = curr             # prev 前进
        curr = next_node        # curr 前进

    return prev  # 新的头节点

# 测试
# 1→2→3→4→5 → 5→4→3→2→1
```

**反转链表的一部分（LeetCode 92）**：

```python
def reverse_between(head, left, right):
    """反转 [left, right] 范围内的节点（1-indexed）"""
    dummy = ListNode(0)
    dummy.next = head
    pre = dummy

    # 走到 left 的前驱
    for _ in range(left - 1):
        pre = pre.next

    # 反转 [left, right] 段
    curr = pre.next
    for _ in range(right - left):
        next_node = curr.next
        curr.next = next_node.next
        next_node.next = pre.next
        pre.next = next_node

    return dummy.next
```

**K 个一组翻转链表（LeetCode 25）**：

```python
def reverse_k_group(head, k):
    """每 k 个节点为一组，翻转"""
    # 检查是否有 k 个节点
    curr = head
    for _ in range(k):
        if not curr:
            return head   # 不足 k 个，不翻转
        curr = curr.next

    # 翻转前 k 个
    prev = None
    curr = head
    for _ in range(k):
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node

    # 递归处理剩余部分，连接到当前组的尾节点（原来的 head）
    head.next = reverse_k_group(curr, k)
    return prev  # 新的头节点（原来的第 k 个节点）
```

---

## 13.6 复制带随机指针的链表

每个节点有 `next` 和 `random` 指针，深拷贝链表。

```python
def copy_random_list(head):
    """三步法：O(n) 时间，O(1) 额外空间（不算输出）"""
    if not head:
        return None

    # 步骤1：在每个节点后插入其拷贝
    curr = head
    while curr:
        copy = Node(curr.val, curr.next, None)
        curr.next = copy
        curr = copy.next

    # 步骤2：设置拷贝节点的 random 指针
    curr = head
    while curr:
        if curr.random:
            curr.next.random = curr.random.next
        curr = curr.next.next

    # 步骤3：分离原链表和拷贝链表
    curr = head
    copy_head = head.next
    while curr:
        copy = curr.next
        curr.next = copy.next
        copy.next = copy.next.next if copy.next else None
        curr = curr.next

    return copy_head
```

---

## 13.7 链表双指针操作总结

| 技巧 | 指针配置 | 时间 | 空间 |
|------|----------|------|------|
| 找中点 | 1:2 快慢 | O(n) | O(1) |
| 判断环 | Floyd | O(n) | O(1) |
| 找环入口 | Floyd + 重置 | O(n) | O(1) |
| 相交节点 | 互换遍历 | O(n) | O(1) |
| 倒数第K个 | 间距K | O(n) | O(1) |
| 反转 | prev/curr/next | O(n) | O(1) |

所有操作均为 O(n) 时间、O(1) 空间——这是链表双指针的最大价值。

---

## 13.8 本章小结

链表双指针的核心：

1. **快慢指针**：步差产生位置信息（中点、环检测）
2. **间距维护**：保持两指针固定间距（倒数第K个）
3. **互换遍历**：两个等长路径同时结束（相交链表）
4. **三指针反转**：prev/curr/next 协同前进

**下一章：接雨水与单调性分析——对撞指针的极致应用**

---

## LeetCode 推荐题目

- [21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/) ⭐
- [148. 排序链表](https://leetcode.cn/problems/sort-list/) ⭐⭐⭐
- [160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/) ⭐⭐
- [92. 反转链表 II](https://leetcode.cn/problems/reverse-linked-list-ii/) ⭐⭐
- [25. K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/) ⭐⭐⭐
- [138. 随机链表的复制](https://leetcode.cn/problems/copy-list-with-random-pointer/) ⭐⭐
