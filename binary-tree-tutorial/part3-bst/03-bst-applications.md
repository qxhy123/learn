# 第11章：BST 应用与变体

## 11.1 有序集合/映射（std::map / TreeMap）

BST 是实现有序字典的基础，Python 的 `sortedcontainers.SortedList` 和 C++ 的 `std::map` 均基于平衡 BST：

```python
from sortedcontainers import SortedList

# 等价于 C++ 的 std::multiset
sl = SortedList([5, 3, 7, 1, 4])
sl.add(6)
print(sl)           # SortedList([1, 3, 4, 5, 6, 7])
print(sl[0])        # 最小值: 1
print(sl[-1])       # 最大值: 7
sl.discard(3)       # 删除
idx = sl.bisect_left(5)  # 类似二分查找
```

## 11.2 用数组+BST实现排行榜

```python
class Leaderboard:
    """
    支持以下操作的排行榜：
    - 添加玩家分数
    - 查询前 K 名分数之和
    - 重置玩家分数
    """
    def __init__(self):
        self.scores = {}  # player_id -> score
        from sortedcontainers import SortedList
        self.sorted_scores = SortedList()
    
    def add_score(self, player_id, score):
        if player_id in self.scores:
            self.sorted_scores.remove(self.scores[player_id])
        self.scores[player_id] = self.scores.get(player_id, 0) + score
        self.sorted_scores.add(self.scores[player_id])
    
    def top(self, k):
        return sum(self.sorted_scores[-k:])  # 最后 k 个（最大）
    
    def reset(self, player_id):
        self.sorted_scores.remove(self.scores[player_id])
        del self.scores[player_id]
```

## 11.3 BST 转链表

**中序遍历 + 指针操作**，将 BST 原地转换为有序双向链表：

```python
def bst_to_doubly_linked_list(root):
    """
    将 BST 原地转为有序双向链表
    left 指针作为 prev，right 指针作为 next
    """
    if root is None:
        return None
    
    first = [None]   # 链表头
    last = [None]    # 链表尾（当前处理的节点）
    
    def inorder(node):
        if node is None:
            return
        
        inorder(node.left)
        
        if last[0]:
            # 连接前一节点和当前节点
            last[0].right = node
            node.left = last[0]
        else:
            first[0] = node  # 第一个节点
        
        last[0] = node
        inorder(node.right)
    
    inorder(root)
    
    # 使链表成环（可选）
    # first[0].left = last[0]
    # last[0].right = first[0]
    
    return first[0]
```

## 11.4 两棵 BST 的合并与比较

### 合并两棵 BST

```python
def merge_bsts(root1, root2):
    """
    合并两棵 BST 为一棵有序 BST
    
    思路：
    1. 分别中序遍历得到两个有序数组
    2. 合并两个有序数组
    3. 从有序数组构建平衡 BST
    """
    def inorder(root):
        result = []
        def helper(node):
            if node:
                helper(node.left)
                result.append(node.val)
                helper(node.right)
        helper(root)
        return result
    
    def merge_sorted(a, b):
        result = []
        i = j = 0
        while i < len(a) and j < len(b):
            if a[i] <= b[j]:
                result.append(a[i]); i += 1
            else:
                result.append(b[j]); j += 1
        result.extend(a[i:])
        result.extend(b[j:])
        return result
    
    def sorted_to_bst(nums, lo, hi):
        if lo > hi:
            return None
        mid = (lo + hi) // 2
        node = TreeNode(nums[mid])
        node.left = sorted_to_bst(nums, lo, mid - 1)
        node.right = sorted_to_bst(nums, mid + 1, hi)
        return node
    
    merged = merge_sorted(inorder(root1), inorder(root2))
    return sorted_to_bst(merged, 0, len(merged) - 1)
```

## 11.5 平衡 BST 的构建

将有序数组构建为高度最小的 BST（二分取中点）：

```python
def sorted_array_to_bst(nums):
    """
    有序数组 → 平衡 BST（LeetCode 108）
    每次取中点作为根，递归构建
    """
    def build(lo, hi):
        if lo > hi:
            return None
        mid = (lo + hi) // 2
        node = TreeNode(nums[mid])
        node.left = build(lo, mid - 1)
        node.right = build(mid + 1, hi)
        return node
    
    return build(0, len(nums) - 1)

def sorted_list_to_bst(head):
    """
    有序链表 → 平衡 BST（LeetCode 109）
    快慢指针找中点
    """
    def find_mid(start, end):
        slow = fast = start
        while fast != end and fast.next != end:
            slow = slow.next
            fast = fast.next.next
        return slow
    
    def build(start, end):
        if start == end:
            return None
        mid = find_mid(start, end)
        node = TreeNode(mid.val)
        node.left = build(start, mid)
        node.right = build(mid.next, end)
        return node
    
    return build(head, None)
```

## 11.6 BST 恢复（两个节点被错误交换）

```python
def recover_bst(root):
    """
    恢复被交换的两个 BST 节点（LeetCode 99）
    
    中序遍历找到两处"逆序对"：
    - 第一处：前者是第一个错误节点
    - 第二处：后者是第二个错误节点
    """
    first = second = prev = None
    
    def inorder(node):
        nonlocal first, second, prev
        if node is None:
            return
        
        inorder(node.left)
        
        if prev and prev.val > node.val:
            if first is None:
                first = prev  # 第一个逆序对的前者
            second = node     # 最新逆序对的后者
        
        prev = node
        inorder(node.right)
    
    inorder(root)
    
    # 交换两个错误节点的值
    first.val, second.val = second.val, first.val
```

## 11.7 Treap（随机化 BST）

Treap = BST + Heap。每个节点有两个键：
- `key`：满足 BST 性质
- `priority`：随机数，满足堆性质

通过随机优先级，以高概率保证树的平衡性：

```python
import random

class TreapNode:
    def __init__(self, key):
        self.key = key
        self.priority = random.random()  # 随机优先级
        self.left = None
        self.right = None
        self.size = 1

class Treap:
    def __init__(self):
        self.root = None
    
    def _size(self, node):
        return node.size if node else 0
    
    def _update(self, node):
        if node:
            node.size = 1 + self._size(node.left) + self._size(node.right)
    
    def _split(self, node, key):
        """按 key 分裂：左树所有键 ≤ key，右树所有键 > key"""
        if node is None:
            return None, None
        
        if node.key <= key:
            left, right = self._split(node.right, key)
            node.right = left
            self._update(node)
            return node, right
        else:
            left, right = self._split(node.left, key)
            node.left = right
            self._update(node)
            return left, node
    
    def _merge(self, left, right):
        """合并两棵树（左树所有键 ≤ 右树所有键）"""
        if left is None:
            return right
        if right is None:
            return left
        
        if left.priority > right.priority:
            left.right = self._merge(left.right, right)
            self._update(left)
            return left
        else:
            right.left = self._merge(left, right.left)
            self._update(right)
            return right
    
    def insert(self, key):
        left, right = self._split(self.root, key)
        new_node = TreapNode(key)
        self.root = self._merge(self._merge(left, new_node), right)
    
    def delete(self, key):
        left, right = self._split(self.root, key - 1)
        _, right = self._split(right, key)
        self.root = self._merge(left, right)
    
    def kth(self, k):
        """第 k 小元素（1-indexed）"""
        node = self.root
        while node:
            left_size = self._size(node.left)
            if k == left_size + 1:
                return node.key
            elif k <= left_size:
                node = node.left
            else:
                k -= left_size + 1
                node = node.right
        return -1
```

## 11.8 BST 变体对比

| 变体 | 平衡方式 | 特点 |
|------|---------|------|
| AVL 树 | 严格高度平衡 | 查找最快，旋转多 |
| 红黑树 | 近似平衡 | 插删少旋转，实用 |
| Treap | 随机优先级 | 实现简单，概率平衡 |
| 跳表 | 概率层级 | 非树结构，等价效果 |
| Splay | 访问节点移根 | 缓存友好 |

## 小结

- BST 是有序字典、排行榜等数据结构的基础
- BST 转有序链表利用中序遍历
- 有序数组转平衡 BST 用二分取中点
- Treap 用随机优先级以高概率保持平衡

## 练习

1. 用 BST 实现一个支持 `insert`, `delete`, `rank(val)`（查询排名）的数据结构
2. 实现 Treap 的 `merge` 操作（合并两棵 Treap，假设左树所有键 < 右树所有键）
3. 从 BST 中找出所有路径和等于某值的路径

---

**上一章：** [BST 增删查改](02-bst-operations.md) | **下一章（Part 4）：** [AVL 树](../part4-balanced-trees/01-avl.md)
