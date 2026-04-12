# 第18章：树形动态规划（Tree DP）

## 18.1 什么是树形 DP

树形 DP 是在树结构上进行的动态规划，利用树的**递归子结构**：

> 每个节点的最优解 = 利用其所有子树的最优解推导

核心范式：**后序遍历（自底向上）**，子节点先计算，父节点后计算。

```python
def tree_dp(root):
    """树形 DP 通用框架"""
    if root is None:
        return base_case
    
    # 递归获取所有子树的结果
    left_result = tree_dp(root.left)
    right_result = tree_dp(root.right)
    
    # 用子树结果计算当前节点的最优解
    current_result = transition(root.val, left_result, right_result)
    
    return current_result
```

## 18.2 经典问题一：打家劫舍 III

```
房屋排列成二叉树，不能同时抢劫相邻（父子）节点，求最大金额。
（LeetCode 337）
```

**状态定义**：对每个节点 v，定义：
- `dp[v][0]` = 不抢 v 时，以 v 为根的子树最大金额
- `dp[v][1]` = 抢 v 时，以 v 为根的子树最大金额

```python
def rob(root):
    def dp(node):
        if node is None:
            return (0, 0)  # (不选, 选)
        
        left = dp(node.left)
        right = dp(node.right)
        
        # 不选当前节点：子节点可选可不选，取最大
        not_rob = max(left) + max(right)
        
        # 选当前节点：子节点不能选
        rob_curr = node.val + left[0] + right[0]
        
        return (not_rob, rob_curr)
    
    return max(dp(root))

# 测试：
#     3
#    / \
#   2   3
#    \    \
#     3    1
# 答案：3 + 3 + 1 = 7（抢根的右子链）
```

## 18.3 经典问题二：最大路径和

```python
def max_path_sum(root):
    """
    任意路径（不必过根）的最大节点值之和（LeetCode 124）
    
    状态：dp(v) = 以 v 为端点向下延伸的最大路径和
    转移：当前节点可以作为"拐点"连接左右子树
    """
    result = [float('-inf')]
    
    def dp(node):
        if node is None:
            return 0
        
        left = max(0, dp(node.left))    # 负收益舍弃
        right = max(0, dp(node.right))
        
        # 以 node 为拐点的路径
        result[0] = max(result[0], node.val + left + right)
        
        # 向父节点贡献（只能选一侧）
        return node.val + max(left, right)
    
    dp(root)
    return result[0]
```

## 18.4 经典问题三：二叉树染色

```
将树的节点涂色（黑/白），同色节点不能相邻，求最大涂色节点数。
等价于：最大独立集问题
```

```python
def max_independent_set(root):
    """
    最大独立集：选取尽量多的节点，使得没有两个相邻节点都被选中
    
    dp(v) = (不选v的最优值, 选v的最优值)
    """
    def dp(node):
        if node is None:
            return (0, 0)
        
        l_no, l_yes = dp(node.left)
        r_no, r_yes = dp(node.right)
        
        # 不选v：子节点可选可不选
        not_pick = max(l_no, l_yes) + max(r_no, r_yes)
        
        # 选v：子节点不能选
        pick = 1 + l_no + r_no
        
        return (not_pick, pick)
    
    return max(dp(root))
```

## 18.5 经典问题四：树的直径

```python
def diameter_of_binary_tree(root):
    """
    树的直径：最长路径（边数）
    
    dp(v) = 以 v 为根的子树中，从 v 出发的最长路径（节点数-1）
    """
    max_diam = [0]
    
    def dp(node):
        if node is None:
            return 0
        
        left_depth = dp(node.left)
        right_depth = dp(node.right)
        
        # 经过 node 的路径长度
        max_diam[0] = max(max_diam[0], left_depth + right_depth)
        
        return 1 + max(left_depth, right_depth)
    
    dp(root)
    return max_diam[0]
```

## 18.6 树形 DP 在一般树（多叉树）上的应用

### 换根 DP（Re-rooting Technique）

当问题需要**以每个节点为根**计算答案时，"换根 DP" 避免了对每个根重复计算。

**问题**：对树中每个节点，计算以它为根时所有节点到根的距离之和。

```python
def sum_of_distances(n, edges):
    """
    每个节点到所有其他节点的距离之和（LeetCode 834）
    
    两遍 DFS：
    1. 从根 0 出发，计算 ans[0] 和每个子树大小 count[v]
    2. 换根：从父节点推导子节点的答案
    """
    from collections import defaultdict
    
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    count = [1] * n    # 子树大小
    ans = [0] * n      # 答案
    
    # 第一遍：从根0计算 ans[0] 和 count
    def dfs1(node, parent):
        for child in graph[node]:
            if child != parent:
                dfs1(child, node)
                count[node] += count[child]
                ans[node] += ans[child] + count[child]
    
    # 第二遍：换根推导
    def dfs2(node, parent):
        for child in graph[node]:
            if child != parent:
                # ans[child] = ans[node] - count[child] + (n - count[child])
                ans[child] = ans[node] - count[child] + (n - count[child])
                dfs2(child, node)
    
    dfs1(0, -1)
    dfs2(0, -1)
    return ans

# 示例：n=6, edges=[[0,1],[0,2],[2,3],[2,4],[2,5]]
print(sum_of_distances(6, [[0,1],[0,2],[2,3],[2,4],[2,5]]))
# [8, 12, 6, 10, 10, 10]
```

## 18.7 树形背包（树形 DP 进阶）

**问题**：选择树中至多 k 个节点，每个节点有价值，要求选中的节点必须包含根节点（连通），求最大价值。

```python
def tree_knapsack(root, k, values):
    """
    树形背包：选恰好 k 个连通节点（含根）的最大价值
    
    dp[v][j] = 以 v 为根的子树，选 j 个节点（含 v）的最大价值
    """
    from functools import lru_cache
    
    # 构建子节点列表（假设节点有 children 属性）
    def dp(node, budget):
        """从 node 子树中选 budget 个节点（含 node）的最大价值"""
        if budget <= 0:
            return 0
        
        children = node.children if hasattr(node, 'children') else []
        if not children:
            return values[node.val] if budget >= 1 else 0
        
        # 背包：在子树中分配名额
        # dp_child[i][j] = 前 i 棵子树，分配 j 个名额的最大价值
        m = len(children)
        prev = [0] * (budget + 1)
        
        for child in children:
            curr = [0] * (budget + 1)
            for total in range(budget + 1):
                for alloc in range(total):  # 分配给当前子树的名额
                    child_val = dp(child, alloc)
                    if prev[total - alloc] + child_val > curr[total]:
                        curr[total] = prev[total - alloc] + child_val
            prev = curr
        
        return values[node.val] + prev[budget - 1]  # -1 因为根节点已占一个名额
    
    return dp(root, k)
```

## 18.8 常见树形 DP 问题分类

| 问题类型 | 关键状态 | 典型题 |
|---------|---------|--------|
| 最大/最小路径 | 每节点的向上贡献 | 最大路径和、直径 |
| 独立集 | (选/不选当前节点) | 打家劫舍III |
| 换根 DP | 两遍 DFS | 所有节点距离和 |
| 树形背包 | dp[v][j] 选 j 个节点 | 选课问题 |
| 树着色 | 颜色状态 | 最小点着色 |

## 小结

- 树形 DP = 后序遍历 + 从子树向上传递状态
- 关键：定义清晰的子问题（以 v 为根的子树的最优解）
- 换根 DP：两遍 DFS，避免重复计算
- 树形背包：O(n²) 或更优，注意名额分配

## 练习

1. 求树中距离不超过 k 的节点对数
2. 实现"监控二叉树"（LeetCode 968）：用最少摄像头覆盖所有节点
3. 用换根 DP 求每个节点到最远叶节点的距离

---

**上一章：** [Trie](03-trie.md) | **下一章：** [最近公共祖先（LCA）](05-lca.md)
