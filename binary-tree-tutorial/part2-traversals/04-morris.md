# 第8章：Morris 遍历（O(1) 空间）

## 8.1 动机：突破空间限制

递归和迭代遍历都需要 O(h) 的额外空间（栈）。**Morris 遍历**通过临时修改树的指针，实现了 **O(1) 空间**的遍历。

| 方法 | 时间 | 空间 | 是否修改树 |
|------|------|------|----------|
| 递归 | O(n) | O(h) | 否 |
| 迭代 | O(n) | O(h) | 否 |
| Morris | O(n) | O(1) | 临时修改，遍历后复原 |

**适用场景**：内存极度受限的嵌入式系统；超大型数据集。

## 8.2 核心思想：线索化

Morris 遍历的关键洞察：

> **当前节点左子树的最右节点（中序前驱）的 right 指针是空的，可以临时指向当前节点，作为"返回路径"。**

```
原树:          临时线索化后:
    4               4
   / \             / \
  2   5           2   5
 / \             / \
1   3           1   3
                     \
                      4  ← 3.right 临时指向 4（中序后继）
```

## 8.3 中序 Morris 遍历

算法步骤：

1. 若 `curr.left` 为空：访问 curr，移向右子节点
2. 若 `curr.left` 非空：
   a. 找到 curr 左子树的**最右节点**（中序前驱）
   b. 若前驱的 right 为空：建立线索（前驱.right = curr），curr 移向左子节点
   c. 若前驱的 right = curr：拆除线索，**访问 curr**，curr 移向右子节点

```python
def morris_inorder(root):
    """
    Morris 中序遍历
    时间 O(n)（每个节点访问不超过 2 次），空间 O(1)
    """
    result = []
    curr = root
    
    while curr:
        if curr.left is None:
            # 左子树为空，直接访问，右移
            result.append(curr.val)
            curr = curr.right
        else:
            # 找中序前驱（左子树最右节点）
            predecessor = curr.left
            while predecessor.right and predecessor.right != curr:
                predecessor = predecessor.right
            
            if predecessor.right is None:
                # 第一次到达：建立线索
                predecessor.right = curr
                curr = curr.left
            else:
                # 第二次到达：拆除线索，访问当前节点
                predecessor.right = None  # 恢复原树
                result.append(curr.val)
                curr = curr.right
    
    return result

# 验证
#       4
#      / \
#     2   5
#    / \
#   1   3
# 中序：[1, 2, 3, 4, 5]
```

## 8.4 前序 Morris 遍历

前序与中序的区别仅在访问时机：**第一次到达节点时访问**（中序是第二次）。

```python
def morris_preorder(root):
    """
    Morris 前序遍历
    在"建立线索"时（第一次到达）访问节点
    """
    result = []
    curr = root
    
    while curr:
        if curr.left is None:
            result.append(curr.val)  # 访问
            curr = curr.right
        else:
            predecessor = curr.left
            while predecessor.right and predecessor.right != curr:
                predecessor = predecessor.right
            
            if predecessor.right is None:
                result.append(curr.val)  # 第一次到达时访问（与中序不同）
                predecessor.right = curr
                curr = curr.left
            else:
                predecessor.right = None
                curr = curr.right  # 不访问（已在第一次访问过）
    
    return result
```

## 8.5 后序 Morris 遍历

后序最复杂，需要"逆序打印右链"的技巧：

```python
def morris_postorder(root):
    """
    Morris 后序遍历
    
    思路：
    - 加一个虚拟节点 dump，dump.left = root
    - 每次拆线索时，逆序打印前驱到 curr.left 的右链
    """
    def reverse_print(from_node, to_node):
        """逆序打印从 from_node 到 to_node 的右链"""
        # 先反转链
        if from_node == to_node:
            return [from_node.val]
        
        nodes = []
        node = from_node
        while node != to_node:
            nodes.append(node.val)
            node = node.right
        nodes.append(to_node.val)
        return nodes[::-1]
    
    result = []
    dump = TreeNode(0)
    dump.left = root
    curr = dump
    
    while curr:
        if curr.left is None:
            curr = curr.right
        else:
            predecessor = curr.left
            while predecessor.right and predecessor.right != curr:
                predecessor = predecessor.right
            
            if predecessor.right is None:
                predecessor.right = curr
                curr = curr.left
            else:
                # 逆序打印右链
                result.extend(reverse_print(curr.left, predecessor))
                predecessor.right = None
                curr = curr.right
    
    return result
```

## 8.6 Morris 遍历的时间复杂度证明

虽然有内层 while 循环，但总时间仍是 O(n)：

- 每条"右链"（含线索）被遍历至多 **2 次**（建立线索 1 次 + 拆除线索 1 次）
- 树中所有边的数量为 n-1
- 因此总操作次数 ≤ 2(n-1) + n = O(n)

## 8.7 实际应用：验证 BST（O(1) 空间）

```python
def is_valid_bst_morris(root):
    """
    验证 BST（Morris 中序遍历，O(1) 空间）
    中序遍历 BST 应为严格递增序列
    """
    prev_val = float('-inf')
    curr = root
    
    while curr:
        if curr.left is None:
            # 访问当前节点
            if curr.val <= prev_val:
                return False
            prev_val = curr.val
            curr = curr.right
        else:
            predecessor = curr.left
            while predecessor.right and predecessor.right != curr:
                predecessor = predecessor.right
            
            if predecessor.right is None:
                predecessor.right = curr
                curr = curr.left
            else:
                predecessor.right = None
                # 访问当前节点
                if curr.val <= prev_val:
                    return False
                prev_val = curr.val
                curr = curr.right
    
    return True
```

## 8.8 与递归/迭代的对比总结

```
内存使用（n=10^6 节点，每节点 28 bytes）：

递归/迭代（平衡树 h=20）: ~20 帧 * 几百字节 ≈ 可忽略
递归/迭代（退化树 h=n）:  ~10^6 帧 * 几百字节 ≈ 数百 MB → 栈溢出！
Morris:                    仅 2 个指针变量 ≈ 16 bytes
```

## 小结

- Morris 遍历通过临时线索化实现 O(1) 空间
- 每个节点被访问至多 2 次，总时间 O(n)
- 遍历结束后树结构完全恢复
- 实现较复杂，一般只在空间极度受限时使用

## 练习

1. 用 Morris 遍历找 BST 第 k 小的元素
2. 在 Morris 中序遍历过程中，找到树的中位数（不存储全部元素）
3. 比较 Morris 与迭代方法在退化树（10^4 节点链）上的实际内存使用

---

**上一章：** [BFS](03-bfs.md) | **下一章（Part 3）：** [BST 概念与性质](../part3-bst/01-bst-concepts.md)
