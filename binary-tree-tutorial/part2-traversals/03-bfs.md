# 第7章：广度优先遍历（BFS / 层序遍历）

## 7.1 BFS 的核心思想

BFS 用**队列**（FIFO）实现，确保按层从左到右访问节点。

```
        1          ← Level 0
       / \
      2   3        ← Level 1
     / \   \
    4   5   6      ← Level 2

BFS 顺序：1, 2, 3, 4, 5, 6
```

与 DFS 的区别：

| | DFS | BFS |
|-|-----|-----|
| 数据结构 | 栈（显式/隐式） | 队列 |
| 遍历方向 | 深度优先 | 宽度优先 |
| 空间复杂度 | O(h)（树高） | O(w)（最宽层的节点数） |
| 适合场景 | 路径问题、全局性质 | 最短路径、层级处理 |

## 7.2 基本 BFS 实现

```python
from collections import deque

def level_order(root):
    """层序遍历，返回所有节点值（一维列表）"""
    if root is None:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        result.append(node.val)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return result
```

## 7.3 逐层处理（分层 BFS）

更常用的是"逐层处理"，需要知道每层的边界：

```python
def level_order_by_level(root):
    """
    层序遍历，按层返回（二维列表）
    LeetCode 102
    """
    if root is None:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)  # 当前层节点数
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result

# 输出：[[1], [2, 3], [4, 5, 6]]
```

## 7.4 BFS 的经典应用

### 树的最小深度

```python
def min_depth(root):
    """
    最小深度：根节点到最近叶节点的路径长度
    注意：不是简单的 min(left_depth, right_depth)
    （若一侧为空，不算叶节点）
    BFS 找到第一个叶节点时即为最小深度
    """
    if root is None:
        return 0
    
    queue = deque([(root, 1)])
    
    while queue:
        node, depth = queue.popleft()
        
        # 找到叶节点
        if node.left is None and node.right is None:
            return depth
        
        if node.left:
            queue.append((node.left, depth + 1))
        if node.right:
            queue.append((node.right, depth + 1))
    
    return 0
```

### 二叉树右视图

```python
def right_side_view(root):
    """
    右视图：每层最右边的节点
    LeetCode 199
    """
    if root is None:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            if i == level_size - 1:  # 每层最后一个节点
                result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return result
```

### 锯齿形层序遍历

```python
def zigzag_level_order(root):
    """
    奇数层从左到右，偶数层从右到左
    LeetCode 103
    """
    if root is None:
        return []
    
    result = []
    queue = deque([root])
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        current_level = deque()
        
        for _ in range(level_size):
            node = queue.popleft()
            
            if left_to_right:
                current_level.append(node.val)
            else:
                current_level.appendleft(node.val)  # 头部插入
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(list(current_level))
        left_to_right = not left_to_right
    
    return result
```

### 填充每个节点的下一个右侧节点指针

```python
class Node:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next  # 指向同层右侧节点

def connect(root):
    """
    填充 next 指针（LeetCode 117，任意二叉树）
    
    O(n) 时间，O(w) 空间（BFS 方案）
    """
    if root is None:
        return root
    
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        prev = None
        
        for _ in range(level_size):
            node = queue.popleft()
            
            if prev:
                prev.next = node
            prev = node
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return root
```

### 每层的平均值

```python
def average_of_levels(root):
    """每层节点值的平均值（LeetCode 637）"""
    if root is None:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level_sum = 0
        
        for _ in range(level_size):
            node = queue.popleft()
            level_sum += node.val
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level_sum / level_size)
    
    return result
```

## 7.5 BFS 双向遍历（进阶）

有时需要从**多个源节点**同时 BFS：

```python
def find_leaves_distance_k(root, target_val, k):
    """
    找到距目标节点距离为 k 的所有节点
    LeetCode 863
    
    思路：先建父节点映射，再从目标节点 BFS
    """
    parent = {}
    
    def build_parent(node, par):
        if node is None:
            return
        parent[node] = par
        build_parent(node.left, node)
        build_parent(node.right, node)
    
    build_parent(root, None)
    
    # 找目标节点
    def find_target(node, val):
        if node is None:
            return None
        if node.val == val:
            return node
        return find_target(node.left, val) or find_target(node.right, val)
    
    target = find_target(root, target_val)
    
    # 从目标节点 BFS（可以向父节点走）
    visited = {target}
    queue = deque([target])
    dist = 0
    
    while queue and dist < k:
        for _ in range(len(queue)):
            node = queue.popleft()
            
            for neighbor in [node.left, node.right, parent[node]]:
                if neighbor and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        dist += 1
    
    return [node.val for node in queue]
```

## 7.6 空间复杂度分析

BFS 的空间取决于**最宽层的节点数**：

- **完美二叉树**：最宽层（最后层）有 n/2 个节点，空间 O(n)
- **链状树**：每层只有一个节点，空间 O(1)
- **平均情况**：O(w)，w 是树的最大宽度

> 结论：对于宽树，BFS 的空间消耗远大于 DFS。

## 小结

| 功能 | BFS 实现要点 |
|------|------------|
| 基本层序 | 队列 + popleft |
| 分层处理 | 记录 level_size |
| 最短路径 | 第一次到达即最短 |
| 右视图 | 每层最后一个节点 |
| 锯齿形 | deque 头尾交替插入 |

## 练习

1. 实现"层序遍历的逆序"（从最后一层到第一层）
2. 找出二叉树每层的最大值
3. 判断一棵二叉树是否为完全二叉树（用 BFS）

---

**上一章：** [DFS 迭代](02-dfs-iterative.md) | **下一章：** [Morris 遍历](04-morris.md)
