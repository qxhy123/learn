# 第12章：AVL 树

## 12.1 动机：BST 的退化问题

普通 BST 在最坏情况下退化为链表（O(n) 操作）。AVL 树（1962年，Adelson-Velsky & Landis）是第一种**自平衡 BST**，通过**旋转**操作维持高度平衡。

**平衡因子（Balance Factor）**：

```
BF(v) = height(v.right) - height(v.left)
```

AVL 树保证：每个节点的 **|BF| ≤ 1**

## 12.2 四种旋转

旋转是 AVL 树维持平衡的核心操作，分为四种情况：

### 右旋（LL 失衡）

```
失衡（BF(z) = -2, BF(y) = -1 或 0）:
        z(-2)                 y(0)
       /                    /     \
      y(-1)      →         x(0)   z(0)
     /
    x

绕 z 右旋
```

```python
def rotate_right(z):
    """右旋（LL 旋转）"""
    y = z.left
    T3 = y.right  # y 的右子树
    
    # 旋转
    y.right = z
    z.left = T3
    
    # 更新高度（先更新 z，再更新 y）
    z.height = 1 + max(height(z.left), height(z.right))
    y.height = 1 + max(height(y.left), height(y.right))
    
    return y  # y 成为新的根
```

### 左旋（RR 失衡）

```
失衡（BF(z) = 2, BF(y) = 1 或 0）:
    z(2)                     y(0)
      \                    /     \
      y(1)      →         z(0)   x(0)
        \
        x

绕 z 左旋
```

```python
def rotate_left(z):
    """左旋（RR 旋转）"""
    y = z.right
    T2 = y.left
    
    y.left = z
    z.right = T2
    
    z.height = 1 + max(height(z.left), height(z.right))
    y.height = 1 + max(height(y.left), height(y.right))
    
    return y
```

### 左右旋（LR 失衡）

```
失衡（BF(z) = -2, BF(y) = 1）:
    z(-2)         z(-2)           x(0)
   /             /               /    \
  y(1)   →     x(0)    →       y(0)  z(0)
    \           /
    x          y

先对 y 左旋，再对 z 右旋
```

### 右左旋（RL 失衡）

```
失衡（BF(z) = 2, BF(y) = -1）:
先对 y 右旋，再对 z 左旋
```

## 12.3 完整 AVL 树实现

```python
class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1  # 新节点高度为 1

class AVLTree:
    
    def _height(self, node):
        return node.height if node else 0
    
    def _bf(self, node):
        """平衡因子"""
        return self._height(node.right) - self._height(node.left)
    
    def _update_height(self, node):
        node.height = 1 + max(self._height(node.left), 
                               self._height(node.right))
    
    def _rotate_right(self, z):
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        self._update_height(z)
        self._update_height(y)
        return y
    
    def _rotate_left(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        self._update_height(z)
        self._update_height(y)
        return y
    
    def _rebalance(self, node):
        """检查并修复失衡，返回新根"""
        self._update_height(node)
        bf = self._bf(node)
        
        # LL 失衡：右旋
        if bf < -1 and self._bf(node.left) <= 0:
            return self._rotate_right(node)
        
        # LR 失衡：先左旋再右旋
        if bf < -1 and self._bf(node.left) > 0:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        
        # RR 失衡：左旋
        if bf > 1 and self._bf(node.right) >= 0:
            return self._rotate_left(node)
        
        # RL 失衡：先右旋再左旋
        if bf > 1 and self._bf(node.right) < 0:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        
        return node  # 已平衡
    
    def insert(self, root, key):
        """插入后自底向上重平衡"""
        if root is None:
            return AVLNode(key)
        
        if key < root.key:
            root.left = self.insert(root.left, key)
        elif key > root.key:
            root.right = self.insert(root.right, key)
        else:
            return root  # 重复键不插入
        
        return self._rebalance(root)
    
    def _min_node(self, node):
        while node.left:
            node = node.left
        return node
    
    def delete(self, root, key):
        """删除后自底向上重平衡"""
        if root is None:
            return None
        
        if key < root.key:
            root.left = self.delete(root.left, key)
        elif key > root.key:
            root.right = self.delete(root.right, key)
        else:
            if root.left is None:
                return root.right
            if root.right is None:
                return root.left
            
            # 两个子节点：用后继替换
            succ = self._min_node(root.right)
            root.key = succ.key
            root.right = self.delete(root.right, succ.key)
        
        return self._rebalance(root)
    
    def search(self, root, key):
        """同 BST 查找"""
        if root is None or root.key == key:
            return root
        if key < root.key:
            return self.search(root.left, key)
        return self.search(root.right, key)

# 测试
avl = AVLTree()
root = None
for x in [1, 2, 3, 4, 5, 6, 7]:  # 有序插入，BST 会退化
    root = avl.insert(root, x)

# AVL 自动保持平衡，树高 ≈ log(7) ≈ 3
def print_height(avl_tree, node):
    return avl_tree._height(node)

print(f"树高: {print_height(avl, root)}")  # 应该是 3，而非 6（退化链）
```

## 12.4 旋转次数分析

| 操作 | 最多旋转次数 |
|------|------------|
| 插入 | 1 次（单旋或双旋） |
| 删除 | O(log n) 次 |

> 这是 AVL 树的一个缺点：删除最多需要 O(log n) 次旋转。红黑树改进了这一点。

## 12.5 AVL 树高度上界

高度为 h 的 AVL 树，最少节点数 $N(h)$：

$$N(0) = 1, \quad N(1) = 2, \quad N(h) = N(h-1) + N(h-2) + 1$$

这与斐波那契数列相关：$N(h) \approx \phi^h / \sqrt{5}$

因此 $h \leq 1.44 \log_2(n+2)$，保证了 **O(log n)** 的树高。

## 12.6 与红黑树对比

| | AVL 树 | 红黑树 |
|-|--------|--------|
| 平衡严格度 | 更严格（高度差≤1） | 较宽松（高度差≤2倍） |
| 查找 | 稍快 | 稍慢 |
| 插入旋转 | ≤ 2 | ≤ 2 |
| 删除旋转 | O(log n) | ≤ 3 |
| 实现复杂度 | 中等 | 复杂 |
| 应用场景 | 读多写少 | 读写均衡（如 std::map） |

## 小结

- AVL 树通过四种旋转保持 |BF| ≤ 1
- 插入/删除/查找均为 O(log n)
- 插入最多 2 次旋转，删除最多 O(log n) 次
- 树高上界 ≈ 1.44 log₂(n)

## 练习

1. 手动模拟插入序列 `[3, 2, 1, 4, 5, 6, 7]` 时 AVL 树的所有旋转过程
2. 在 AVL 节点中增加 `size` 字段，实现 O(log n) 的 `kth_smallest`
3. 实现 AVL 树的范围删除操作

---

**上一章（Part 3）：** [BST 应用](../part3-bst/03-bst-applications.md) | **下一章：** [红黑树](02-red-black.md)
