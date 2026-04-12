# 第4章：二叉树的存储与表示方法

## 4.1 两种主流存储方式

二叉树有两种主流存储方式，各有优劣：

| 方式 | 适用场景 | 空间效率 | 访问方式 |
|------|---------|---------|---------|
| 链式存储 | 一般二叉树、BST | 灵活，但有指针开销 | 指针跳转 |
| 数组存储 | 完全二叉树、堆 | 无指针开销 | 下标计算 |

## 4.2 链式存储（指针表示）

最常见的方式，每个节点包含值和左右指针。

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 构建：
#       1
#      / \
#     2   3
root = TreeNode(1, TreeNode(2), TreeNode(3))
```

**优点**：
- 插入/删除节点 O(1)（已知位置时）
- 天然支持任意形状的树

**缺点**：
- 每个节点额外存储两个指针（内存开销）
- 缓存局部性差（节点分散在堆内存中）

## 4.3 数组存储（完全二叉树）

对于完全二叉树，可以用数组按**层序**存储，通过下标公式直接定位父子节点。

### 下标规律（1-indexed）

```
数组:  [_, 1, 2, 3, 4, 5, 6, 7]  （下标0处留空）
          根  第二层   第三层

树结构:
          1 (idx=1)
         / \
    (2) 2   3 (idx=3)
       / \ / \
   (4)4 5 6  7(idx=7)
```

```python
# 对于节点在数组中的下标 i（从1开始）：
parent(i)  = i // 2
left(i)    = 2 * i
right(i)   = 2 * i + 1

# 对于 0-indexed（更常用）：
parent(i)  = (i - 1) // 2
left(i)    = 2 * i + 1
right(i)   = 2 * i + 2
```

### 实现数组二叉树

```python
class ArrayBinaryTree:
    """基于数组的完全二叉树（0-indexed）"""
    
    def __init__(self, arr):
        self.data = arr[:]  # 复制数组
    
    def size(self):
        return len(self.data)
    
    def val(self, i):
        """获取下标 i 的节点值"""
        if i < 0 or i >= self.size():
            return None
        return self.data[i]
    
    def left(self, i):
        """左子节点下标"""
        return 2 * i + 1
    
    def right(self, i):
        """右子节点下标"""
        return 2 * i + 2
    
    def parent(self, i):
        """父节点下标"""
        return (i - 1) // 2
    
    def is_leaf(self, i):
        return self.left(i) >= self.size()
    
    def level_order(self):
        """层序遍历"""
        return self.data[:]
    
    def inorder(self, i=0):
        """中序遍历（递归）"""
        if i >= self.size() or self.data[i] is None:
            return []
        result = []
        result.extend(self.inorder(self.left(i)))
        result.append(self.data[i])
        result.extend(self.inorder(self.right(i)))
        return result

# 使用示例
tree = ArrayBinaryTree([1, 2, 3, 4, 5, 6, 7])
print(tree.val(0))      # 1（根）
print(tree.left(0))     # 1（左子节点下标）
print(tree.val(tree.left(0)))   # 2
print(tree.val(tree.right(0)))  # 3
print(tree.inorder())   # [4, 2, 5, 1, 6, 3, 7]
```

### 数组存储的局限性

对于**非完全二叉树**，数组存储会浪费空间（用 None 填充不存在的节点）：

```
稀疏树（只有右链）:
  1
   \
    2
     \
      3

数组存储（1-indexed）: [_, 1, None, 2, None, None, None, 3]
                            根  空    2 的父节点3 需要下标7

节点数: 3，但数组需要长度 8 = O(2^n)，极度浪费！
```

## 4.4 序列化与反序列化

在实际应用中（如 LeetCode 题目输入、数据库存储），需要将二叉树转换为字符串形式。

### 层序序列化（LeetCode 格式）

```python
from collections import deque

def serialize(root):
    """
    将二叉树序列化为字符串
    格式：层序，None 用 'null' 表示
    示例：[1,2,3,null,null,4,5] → "1,2,3,null,null,4,5"
    """
    if root is None:
        return "null"
    
    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        if node is None:
            result.append("null")
        else:
            result.append(str(node.val))
            queue.append(node.left)
            queue.append(node.right)
    
    # 去掉尾部的 null
    while result and result[-1] == "null":
        result.pop()
    
    return ",".join(result)

def deserialize(data):
    """将字符串反序列化为二叉树"""
    if data == "null" or not data:
        return None
    
    values = data.split(",")
    root = TreeNode(int(values[0]))
    queue = deque([root])
    i = 1
    
    while queue and i < len(values):
        node = queue.popleft()
        
        if i < len(values) and values[i] != "null":
            node.left = TreeNode(int(values[i]))
            queue.append(node.left)
        i += 1
        
        if i < len(values) and values[i] != "null":
            node.right = TreeNode(int(values[i]))
            queue.append(node.right)
        i += 1
    
    return root

# 测试
root = deserialize("1,2,3,null,null,4,5")
print(serialize(root))  # "1,2,3,null,null,4,5"
```

### 前序序列化（带结构信息）

```python
def serialize_preorder(root):
    """前序序列化，# 代表空节点"""
    if root is None:
        return "#"
    left = serialize_preorder(root.left)
    right = serialize_preorder(root.right)
    return f"{root.val},{left},{right}"

def deserialize_preorder(data):
    """前序反序列化"""
    values = iter(data.split(","))
    
    def build():
        val = next(values)
        if val == "#":
            return None
        node = TreeNode(int(val))
        node.left = build()
        node.right = build()
        return node
    
    return build()

# 测试
root = TreeNode(1, TreeNode(2), TreeNode(3))
s = serialize_preorder(root)
print(s)  # "1,2,#,#,3,#,#"
r = deserialize_preorder(s)
print(serialize_preorder(r))  # "1,2,#,#,3,#,#"
```

## 4.5 线索二叉树（进阶）

在二叉树的链式存储中，叶节点的左右指针指向 None，造成浪费。**线索二叉树**利用这些空指针存储遍历的前驱/后继信息。

```python
class ThreadedNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.left_thread = False   # True: left 指向中序前驱
        self.right_thread = False  # True: right 指向中序后继

def build_inorder_threaded(root):
    """
    构建中序线索二叉树
    使用线索后，中序遍历无需递归或栈，O(1) 空间
    """
    prev = [None]  # 记录前一个访问的节点
    
    def thread(node):
        if node is None:
            return
        
        thread(node.left)
        
        # 处理左线索
        if node.left is None:
            node.left = prev[0]
            node.left_thread = True
        
        # 处理右线索（前一节点的右指针）
        if prev[0] and prev[0].right is None:
            prev[0].right = node
            prev[0].right_thread = True
        
        prev[0] = node
        thread(node.right)
    
    thread(root)
    return root
```

## 小结

| 存储方式 | 适用场景 | 父子访问 | 空间复杂度 |
|---------|---------|---------|----------|
| 链式（指针） | 任意二叉树 | 指针跳转 | O(n) |
| 数组 | 完全二叉树 | 下标公式 | O(n)（完全树）|
| 线索 | 频繁遍历 | 线索指针 | O(n) |

## 练习

1. 用数组实现一个最小堆（见 Part4 预习）
2. 实现一个函数，将任意二叉树序列化后重建，验证两棵树相同
3. 思考：为什么数组存储更适合堆？

---

**上一章：** [二叉树的分类](03-types.md) | **下一章（Part 2）：** [深度优先遍历（递归）](../part2-traversals/01-dfs-recursive.md)
