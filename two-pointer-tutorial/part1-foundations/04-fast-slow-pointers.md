# 第4章：快慢指针

## 4.1 问题引入：链表是否有环？

给定链表头节点 `head`，判断链表中是否有环。

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

**暴力想法**：用哈希表记录所有访问过的节点，若再次访问则有环。O(n) 时间，O(n) 空间。

**挑战**：能否 O(1) 空间？

这就引出了**Floyd 判圈算法**（龟兔赛跑算法）。

---

## 4.2 Floyd 判圈算法：直觉推导

想象在一个操场上跑步：
- **龟**（slow）：每步走 1 格
- **兔**（fast）：每步走 2 格

**若无环**：兔子先到达终点（null），永远不会再遇到龟。

**若有环**：兔子在环里不停跑，龟也进入环。此时兔每步比龟多走 1 格，**相对速度为 1**，兔一定会追上龟（从后方超越，等价于在环内相遇）。

```
链表：1 → 2 → 3 → 4 → 5 → 3（回到节点3形成环）

步骤  slow    fast
  0     1       1
  1     2       3
  2     3       5
  3     4       4  ← 相遇！（4→5→3→4, fast走两步）
```

---

## 4.3 代码实现

```python
def has_cycle(head):
    slow = fast = head

    while fast and fast.next:  # fast 为 None 或 fast.next 为 None 表示无环
        slow = slow.next
        fast = fast.next.next

        if slow is fast:  # 使用 is 比较节点身份，不是值
            return True

    return False
```

**终止条件分析**：
- `fast is None`：链表长度为偶数，fast 到达末尾
- `fast.next is None`：链表长度为奇数，fast 在最后一个节点

**为何用 `is` 而非 `==`**：比较的是节点**对象身份**（内存地址），不是节点值。两个值相同的不同节点不算相遇。

---

## 4.4 进阶：找环的入口节点

不仅判断有无环，还要找到环的起始节点。

**数学推导**：

设链表头到环入口的距离为 `a`，环的长度为 `b`。

当 slow 和 fast 第一次相遇时：
- slow 走了 `a + m`（m 是环内走的距离）
- fast 走了 `a + m + k*b`（k 圈多走的路）
- 因为 fast 速度是 slow 的 2 倍：`2(a + m) = a + m + k*b`
- 化简：`a + m = k*b`，即 `a = k*b - m`

这意味着：**从相遇点再走 `a` 步，恰好到达环入口**（因为 `k*b - m` 步相当于在环内绕了几圈后回到入口）。

**算法**：相遇后，将一个指针重置到 head，两个指针同速前进，再次相遇处即为环入口。

```python
def detect_cycle(head):
    slow = fast = head

    # 第一步：找相遇点
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            break
    else:
        return None  # 无环

    # 第二步：找环入口
    slow = head  # 重置 slow 到头部
    while slow is not fast:
        slow = slow.next
        fast = fast.next  # fast 现在也是每次走 1 步

    return slow  # 相遇处即为环入口
```

**验证**：
```
链表：0 → 1 → 2 → 3 → 4 → 2（环入口为节点2）
a = 2（0到2的距离），b = 3（2→3→4→2）

第一次相遇（设在节点4，即 m=2）：
  a = k*b - m = 1*3 - 2 = 1？不对...

让我们实际模拟：
步骤  slow  fast
  0     0     0
  1     1     2
  2     2     4
  3     3     3  ← 相遇（3→4→2→3，fast走两步）

相遇在节点3，m=1（从入口2走了1步到3）
a = k*b - m = 1*3 - 1 = 2 ✓（head到入口恰好2步）
```

---

## 4.5 经典应用一：找链表中点

```python
def find_middle(head):
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow  # 奇数长度返回正中间，偶数长度返回后半段的第一个节点
```

**数学直觉**：fast 速度是 slow 的 2 倍，当 fast 到达终点时，slow 恰好走了一半。

```
奇数：1 → 2 → 3 → 4 → 5
步骤  slow  fast
  0     1     1
  1     2     3
  2     3     5（fast.next=None，停止）
中点：3 ✓

偶数：1 → 2 → 3 → 4
步骤  slow  fast
  0     1     1
  1     2     3
  2     3     None（fast.next=4, fast.next.next=None，停止）
中点：3（后半段第一个）✓
```

**应用**：链表排序（归并排序需要找中点分割）。

---

## 4.6 经典应用二：倒数第 K 个节点

不遍历两遍链表，一遍找到倒数第 k 个节点。

```python
def find_kth_from_end(head, k):
    fast = slow = head

    # 第一步：fast 先走 k 步
    for _ in range(k):
        if fast is None:
            return None  # 链表长度不足 k
        fast = fast.next

    # 第二步：fast 和 slow 同速前进，fast 到终点时 slow 在倒数第 k 个
    while fast:
        fast = fast.next
        slow = slow.next

    return slow
```

**思维模型**：fast 和 slow 保持 k 的间距，当 fast 到达 None 时，slow 距链表末尾恰好 k 步。

```
链表：1 → 2 → 3 → 4 → 5，k=2

初始 fast 走2步：fast=3
      slow=1, fast=3（间距2）

同步推进：
  slow=2, fast=4
  slow=3, fast=5
  slow=4, fast=None（停止）

返回 slow=4，即倒数第2个 ✓
```

**实际应用**：LeetCode 19题——删除链表倒数第 N 个节点时，需要找到倒数第 N+1 个节点（待删节点的前驱）。

```python
def remove_nth_from_end(head, n):
    dummy = ListNode(0)
    dummy.next = head
    fast = slow = dummy

    # fast 先走 n+1 步（多走一步是为了让 slow 停在目标节点的前驱）
    for _ in range(n + 1):
        fast = fast.next

    while fast:
        fast = fast.next
        slow = slow.next

    slow.next = slow.next.next  # 删除节点
    return dummy.head
```

---

## 4.7 经典应用三：判断链表是否为回文

要求 O(1) 空间。

```python
def is_palindrome(head):
    # 1. 找中点
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # 2. 反转后半部分
    prev = None
    curr = slow
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    right_half = prev

    # 3. 对比前后两半
    left, right = head, right_half
    result = True
    while right:  # 右半可能比左半短一个节点（奇数长度时）
        if left.val != right.val:
            result = False
            break
        left = left.next
        right = right.next

    # 4. 还原链表（可选，好的习惯）
    # ... (反转回来)

    return result

# 测试
# 1→2→2→1：中点后为 2→1，反转为 1→2，与前半 1→2 比较 ✓
# 1→2→3→2→1：中点后为 3→2→1，...
```

---

## 4.8 快慢指针速率变体

不一定是 1:2，速率可以按需调整：

```python
# 速率 1:3：找链表 1/3 处
slow = fast = head
while fast and fast.next and fast.next.next:
    slow = slow.next
    fast = fast.next.next.next

# 速率可变：根据条件动态调整
# 用于"跳跃游戏"等问题
```

**"幸运数"问题**（数字替换游戏中检测环）：

```python
def is_happy(n):
    def digit_square_sum(x):
        total = 0
        while x:
            total += (x % 10) ** 2
            x //= 10
        return total

    slow = n
    fast = digit_square_sum(n)

    while fast != 1 and slow != fast:
        slow = digit_square_sum(slow)
        fast = digit_square_sum(digit_square_sum(fast))

    return fast == 1  # fast==1 表示找到幸福数，slow==fast 表示有环（非幸福数）

# 测试
print(is_happy(19))  # True（1²+9²=82→8²+2²=68→...→1）
print(is_happy(2))   # False
```

---

## 4.9 复杂度分析

| 操作 | 时间 | 空间 |
|------|------|------|
| 判断有环 | O(n) | O(1) |
| 找环入口 | O(n) | O(1) |
| 找中点 | O(n) | O(1) |
| 倒数第 K 个 | O(n) | O(1) |
| 回文链表 | O(n) | O(1) |

所有操作都是 O(n) 时间、O(1) 空间，是链表类问题的最优解。

---

## 4.10 本章小结

快慢指针的核心：

| 技巧 | 原理 | 应用 |
|------|------|------|
| 判断有环 | 速度差使得相遇不可避免 | 环形链表 |
| 找环入口 | 相遇后等距推进 | Floyd 算法 |
| 找中点 | 速比 1:2，fast 到终点时 slow 在中间 | 链表归并排序 |
| 倒数第 K 个 | 维护固定间距 | 删除倒数第 N 节点 |

**下一章：有序数组与两数之和——对撞指针的第一个经典战场**

---

## LeetCode 推荐题目

- [141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/) ⭐
- [142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/) ⭐⭐
- [876. 链表的中间结点](https://leetcode.cn/problems/middle-of-the-linked-list/) ⭐
- [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/) ⭐⭐
- [234. 回文链表](https://leetcode.cn/problems/palindrome-linked-list/) ⭐⭐
- [202. 快乐数](https://leetcode.cn/problems/happy-number/) ⭐⭐
