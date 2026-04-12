# 第10章：BST 增删查改

## 10.1 查找（Search）

已在第9章介绍，这里给出完整迭代版本：

```python
def search(root, val):
    """O(h) 查找，推荐迭代写法"""
    while root:
        if root.val == val:
            return root
        root = root.left if val < root.val else root.right
    return None
```

## 10.2 插入（Insert）

**规则**：找到合适的空位置插入叶节点（不改变现有结构）。

```python
def insert(root, val):
    """
    BST 插入（递归）
    时间 O(h)，空间 O(h)
    """
    if root is None:
        return TreeNode(val)  # 找到插入位置
    
    if val < root.val:
        root.left = insert(root.left, val)
    elif val > root.val:
        root.right = insert(root.right, val)
    # val == root.val：BST 通常不插入重复值
    
    return root

def insert_iterative(root, val):
    """BST 插入（迭代，O(1) 额外空间）"""
    new_node = TreeNode(val)
    
    if root is None:
        return new_node
    
    curr = root
    while True:
        if val < curr.val:
            if curr.left is None:
                curr.left = new_node
                return root
            curr = curr.left
        elif val > curr.val:
            if curr.right is None:
                curr.right = new_node
                return root
            curr = curr.right
        else:
            return root  # 重复值，不插入
```

**插入顺序对树高的影响**：

```python
# 按有序序列插入 → 退化树
root = None
for x in [1, 2, 3, 4, 5]:
    root = insert(root, x)
# 结果：1→2→3→4→5（链表，高度=4）

# 按随机顺序插入 → 相对平衡
root = None
for x in [3, 1, 5, 2, 4]:
    root = insert(root, x)
# 结果：平衡树，高度=2
```

## 10.3 删除（Delete）— 最复杂的操作

BST 删除分三种情况：

### 情况1：删除叶节点（无子节点）

直接删除：

```
删除 4:
      5              5
     / \    →       / \
    3   7           3   7
   / \             /
  2   4           2
```

### 情况2：删除有一个子节点的节点

用子节点替换：

```
删除 3（只有左子节点2）:
      5              5
     / \    →       / \
    3   7           2   7
   /
  2
```

### 情况3：删除有两个子节点的节点（关键！）

用**中序后继**（右子树最小值）替换，再删除后继：

```
删除 5（有两个子节点）:
      5              6
     / \    →       / \
    3   7           3   7
       /               /
      6               ← 6 被移走
```

**统一实现**：

```python
def delete(root, val):
    """
    BST 删除（递归）
    时间 O(h)，空间 O(h)
    """
    if root is None:
        return None
    
    if val < root.val:
        root.left = delete(root.left, val)
    elif val > root.val:
        root.right = delete(root.right, val)
    else:
        # 找到要删除的节点
        
        # 情况1 & 2：至多一个子节点
        if root.left is None:
            return root.right
        if root.right is None:
            return root.left
        
        # 情况3：两个子节点
        # 找中序后继（右子树最小值）
        successor = root.right
        while successor.left:
            successor = successor.left
        
        # 用后继值替换当前节点值
        root.val = successor.val
        
        # 删除后继节点（后继最多只有右子节点）
        root.right = delete(root.right, successor.val)
    
    return root
```

### 另一种策略：用前驱替换

```python
def delete_with_predecessor(root, val):
    """用中序前驱（左子树最大值）替换"""
    if root is None:
        return None
    
    if val < root.val:
        root.left = delete_with_predecessor(root.left, val)
    elif val > root.val:
        root.right = delete_with_predecessor(root.right, val)
    else:
        if root.left is None:
            return root.right
        if root.right is None:
            return root.left
        
        # 找中序前驱（左子树最大值）
        predecessor = root.left
        while predecessor.right:
            predecessor = predecessor.right
        
        root.val = predecessor.val
        root.left = delete_with_predecessor(root.left, predecessor.val)
    
    return root
```

## 10.4 操作的完整测试

```python
class BST:
    """BST 的完整封装"""
    
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        self.root = self._insert(self.root, val)
    
    def _insert(self, node, val):
        if node is None:
            return TreeNode(val)
        if val < node.val:
            node.left = self._insert(node.left, val)
        elif val > node.val:
            node.right = self._insert(node.right, val)
        return node
    
    def search(self, val):
        return self._search(self.root, val)
    
    def _search(self, node, val):
        if node is None or node.val == val:
            return node
        return self._search(node.left if val < node.val else node.right, val)
    
    def delete(self, val):
        self.root = self._delete(self.root, val)
    
    def _delete(self, node, val):
        if node is None:
            return None
        if val < node.val:
            node.left = self._delete(node.left, val)
        elif val > node.val:
            node.right = self._delete(node.right, val)
        else:
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            
            # 找后继
            succ = node.right
            while succ.left:
                succ = succ.left
            node.val = succ.val
            node.right = self._delete(node.right, succ.val)
        return node
    
    def inorder(self):
        result = []
        def _inorder(node):
            if node:
                _inorder(node.left)
                result.append(node.val)
                _inorder(node.right)
        _inorder(self.root)
        return result

# 测试
bst = BST()
for x in [5, 3, 7, 2, 4, 6, 8]:
    bst.insert(x)

print(bst.inorder())        # [2, 3, 4, 5, 6, 7, 8]

bst.delete(3)  # 删除有两个子节点的节点
print(bst.inorder())        # [2, 4, 5, 6, 7, 8]

bst.delete(8)  # 删除叶节点
print(bst.inorder())        # [2, 4, 5, 6, 7]

print(bst.search(5))        # <TreeNode val=5>
print(bst.search(9))        # None
```

## 10.5 修改（Update）

BST 中没有直接的"修改"操作（改变值可能破坏 BST 性质），正确做法是**先删除再插入**：

```python
def update(root, old_val, new_val):
    """BST 修改 = 删除旧值 + 插入新值"""
    root = delete(root, old_val)
    root = insert(root, new_val)
    return root
```

## 10.6 范围查询

```python
def range_query(root, lo, hi):
    """
    返回 BST 中 [lo, hi] 范围内的所有值
    利用 BST 性质剪枝，效率高于全遍历
    """
    result = []
    
    def dfs(node):
        if node is None:
            return
        
        # 剪枝：当前值大于 lo，左子树可能有满足条件的
        if node.val > lo:
            dfs(node.left)
        
        # 当前值在范围内
        if lo <= node.val <= hi:
            result.append(node.val)
        
        # 剪枝：当前值小于 hi，右子树可能有满足条件的
        if node.val < hi:
            dfs(node.right)
    
    dfs(root)
    return result

def range_sum(root, lo, hi):
    """BST 范围求和（LeetCode 938）"""
    if root is None:
        return 0
    
    if root.val < lo:
        return range_sum(root.right, lo, hi)
    if root.val > hi:
        return range_sum(root.left, lo, hi)
    
    return root.val + range_sum(root.left, lo, hi) + range_sum(root.right, lo, hi)
```

## 10.7 各操作复杂度总结

| 操作 | 平均（平衡） | 最坏（退化） |
|------|-----------|-----------|
| 查找 | O(log n) | O(n) |
| 插入 | O(log n) | O(n) |
| 删除 | O(log n) | O(n) |
| 最小/大值 | O(log n) | O(n) |
| 前驱/后继 | O(log n) | O(n) |
| 范围查询(k个结果) | O(log n + k) | O(n + k) |

## 小结

- 插入：找到空位置插入叶节点
- 删除分三种情况：无子、单子、双子（用后继替换）
- 修改 = 删除 + 插入
- BST 操作效率依赖树的平衡性

## 练习

1. 实现 BST 的 `floor(val)` 和 `ceil(val)` 操作
2. 将两棵 BST 合并为一棵新 BST
3. 从 BST 中删除一个范围 `[lo, hi]` 内的所有节点

---

**上一章：** [BST 概念](01-bst-concepts.md) | **下一章：** [BST 应用与变体](03-bst-applications.md)
