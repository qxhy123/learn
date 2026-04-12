# 第6章：深度优先遍历（迭代实现）

## 6.1 为什么需要迭代实现

递归的本质是利用**函数调用栈**。迭代实现用**显式栈**模拟这一过程。

使用迭代的理由：
1. **防止栈溢出**：Python 默认递归深度限制为 1000
2. **可控制暂停/恢复**：用于生成器、惰性求值
3. **面试常考**：考察对栈的理解

## 6.2 前序遍历（迭代）

思路：用栈模拟递归，由于栈是 LIFO，**先压右子树，再压左子树**。

```python
def preorder_iterative(root):
    """
    前序遍历迭代实现
    时间：O(n)，空间：O(h)
    """
    if root is None:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)      # 访问根
        
        # 先压右（后访问），再压左（先访问）
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result

# 验证：
#       1
#      / \
#     2   3
#    / \
#   4   5
# 输出：[1, 2, 4, 5, 3]
```

## 6.3 中序遍历（迭代）

中序遍历的迭代稍复杂，需要"深入左链"再回溯：

```python
def inorder_iterative(root):
    """
    中序遍历迭代实现
    
    思路：
    1. 沿左链一直压栈
    2. 弹出节点，访问
    3. 转向右子树，重复
    """
    result = []
    stack = []
    curr = root
    
    while curr or stack:
        # 沿左链压栈
        while curr:
            stack.append(curr)
            curr = curr.left
        
        # 弹出并访问
        curr = stack.pop()
        result.append(curr.val)
        
        # 转向右子树
        curr = curr.right
    
    return result
```

**过程演示**（树：1→左2→左4）：

```
初始: curr=1, stack=[]

循环1: 压入 1,2,4; stack=[1,2,4], curr=None
  弹出 4, 访问 4, curr=4.right=None
循环2: 跳过内层while; 弹出 2, 访问 2, curr=2.right=5
循环3: 压入 5; stack=[1,5], curr=None
  弹出 5, 访问 5, curr=None
循环4: 弹出 1, 访问 1, curr=1.right=3
循环5: 压入 3; 弹出 3, 访问 3
结果: [4, 2, 5, 1, 3] ✓
```

## 6.4 后序遍历（迭代）

### 方法一：反转前序（技巧）

后序 = 左→右→根，前序 = 根→左→右
→ "根→右→左" 再反转 = 后序

```python
def postorder_iterative_v1(root):
    """方法一：修改前序（根右左），结果反转"""
    if root is None:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        # 注意：先压左（后访问），再压右（先访问）
        # 与前序相反：前序先压右
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    
    return result[::-1]  # 反转得到后序
```

### 方法二：显式标记（标准做法）

```python
def postorder_iterative_v2(root):
    """
    方法二：使用 prev 指针记录上次访问的节点
    用于判断右子树是否已被访问
    """
    result = []
    stack = []
    curr = root
    prev = None
    
    while curr or stack:
        # 沿左链压栈
        while curr:
            stack.append(curr)
            curr = curr.left
        
        curr = stack[-1]  # 窥视栈顶
        
        # 如果右子树存在且未被访问
        if curr.right and curr.right != prev:
            curr = curr.right  # 转向右子树
        else:
            stack.pop()
            result.append(curr.val)  # 访问
            prev = curr
            curr = None
    
    return result
```

## 6.5 统一模板（颜色标记法）

一种优雅的统一方案：用元组 `(node, visited)` 标记节点是否已被"处理过"。

```python
def traverse_unified(root, order='inorder'):
    """
    统一三种遍历的迭代实现
    order: 'preorder', 'inorder', 'postorder'
    """
    if root is None:
        return []
    
    result = []
    # 元组：(节点, 是否为"访问"操作)
    stack = [(root, False)]
    
    while stack:
        node, visited = stack.pop()
        
        if node is None:
            continue
        
        if visited:
            # 真正访问节点
            result.append(node.val)
        else:
            # 按遍历顺序，逆序压入（因为栈先进后出）
            if order == 'inorder':
                # 中序：左→根→右，逆序压：右→根→左
                stack.append((node.right, False))
                stack.append((node, True))      # 标记为"待访问"
                stack.append((node.left, False))
            
            elif order == 'preorder':
                # 前序：根→左→右，逆序压：右→左→根
                stack.append((node.right, False))
                stack.append((node.left, False))
                stack.append((node, True))
            
            elif order == 'postorder':
                # 后序：左→右→根，逆序压：根→右→左
                stack.append((node, True))
                stack.append((node.right, False))
                stack.append((node.left, False))
    
    return result

# 测试
root = build_tree([1, 2, 3, 4, 5])
print(traverse_unified(root, 'inorder'))    # [4, 2, 5, 1, 3]
print(traverse_unified(root, 'preorder'))   # [1, 2, 4, 5, 3]
print(traverse_unified(root, 'postorder'))  # [4, 5, 2, 3, 1]
```

## 6.6 迭代实现的应用：BST 迭代器

实际场景中，迭代实现可以做成**惰性求值的迭代器**，非常有用：

```python
class BSTIterator:
    """
    BST 中序遍历迭代器
    next() 和 hasNext() 均为 O(1) 均摊时间，O(h) 空间
    LeetCode 173
    """
    def __init__(self, root):
        self.stack = []
        self._push_left(root)
    
    def _push_left(self, node):
        """将左链压栈"""
        while node:
            self.stack.append(node)
            node = node.left
    
    def next(self):
        """返回下一个最小值"""
        node = self.stack.pop()
        if node.right:
            self._push_left(node.right)
        return node.val
    
    def hasNext(self):
        return bool(self.stack)

# 使用
it = BSTIterator(root)
while it.hasNext():
    print(it.next(), end=' ')
```

## 6.7 复杂度对比

| 实现方式 | 时间 | 空间 | 栈溢出风险 |
|---------|------|------|-----------|
| 递归 | O(n) | O(h) | 是（h > 1000） |
| 迭代 | O(n) | O(h) | 否 |
| Morris | O(n) | O(1) | 否 |

## 小结

- 迭代用显式栈代替递归调用栈
- 前序迭代最简单；后序最复杂
- 统一模板（颜色标记法）可以处理所有三种遍历
- BST 迭代器是迭代中序的实用变体

## 练习

1. 用迭代实现中序遍历，并应用于"验证 BST"问题
2. 实现一个双向迭代器（支持 prev() 操作）
3. 用 Python 生成器（yield）改写迭代中序遍历

---

**上一章：** [DFS 递归](01-dfs-recursive.md) | **下一章：** [广度优先遍历 BFS](03-bfs.md)
