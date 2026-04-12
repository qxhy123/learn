# 第11章：树形DP

## 11.1 什么是树形DP

**树形DP** 是在树结构上进行的动态规划。由于树的天然递归结构，它特别适合 DFS + 记忆化的方式实现。

**核心特征**：
- 状态定义在每个节点上：`dp[v]` 表示以节点 v 为根的子树的最优解
- 转移：父节点的状态由子节点状态合并而来
- 遍历顺序：后序遍历（先处理子节点，再处理父节点）

---

## 11.2 基础：二叉树打家劫舍（LeetCode 337）

**已在第5章介绍**，作为回顾：

```python
def rob_tree(root):
    def dp(node):
        if not node:
            return 0, 0  # (不偷, 偷)
        
        left_no,  left_yes  = dp(node.left)
        right_no, right_yes = dp(node.right)
        
        not_rob = max(left_no, left_yes) + max(right_no, right_yes)
        rob     = node.val + left_no + right_no
        
        return not_rob, rob
    
    return max(dp(root))
```

---

## 11.3 树上最大独立集

**题目**：给定一棵树，选择一个节点的子集，使得没有两个相邻节点（直接父子关系）都被选中，且选中节点的权重之和最大。

这就是树上的打家劫舍问题的一般化。

```python
def max_independent_set(n, weights, edges):
    # 建邻接表
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    dp = [[0, 0] for _ in range(n)]  # [不选, 选]
    
    def dfs(node, parent):
        dp[node][1] = weights[node]  # 选择当前节点
        
        for child in adj[node]:
            if child == parent:
                continue
            dfs(child, node)
            
            # 不选当前节点：子节点可选可不选
            dp[node][0] += max(dp[child][0], dp[child][1])
            # 选当前节点：子节点不能选
            dp[node][1] += dp[child][0]
    
    dfs(0, -1)
    return max(dp[0][0], dp[0][1])
```

---

## 11.4 树上背包

**题目**：公司员工构成一棵以 CEO 为根的树，每位员工有"快乐值"。
若直接上司参加聚会，该员工不参加；否则可以选择参加或不参加。
选择至多 k 名员工参加，使快乐值最大。

这是**树上背包**的经典形式：

```python
def tree_knapsack(n, happy, edges, k):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    
    # dp[v][j] = 以v为根的子树中选j人时的最大快乐值
    dp = [[0] * (k + 1) for _ in range(n)]
    
    def dfs(v):
        # 初始：只选v自己
        dp[v][1] = happy[v]
        subtree_size = 1
        
        for child in adj[v]:
            dfs(child)
            child_size = sum(1 for x in dp[child] if x > 0)
            
            # 合并子树背包（从大到小，防止重复）
            for j in range(min(subtree_size + child_size, k), -1, -1):
                for c in range(min(j, k + 1)):
                    if j - c <= subtree_size and c <= child_size:
                        dp[v][j] = max(dp[v][j], 
                                       dp[v][j-c] + dp[child][c])
            subtree_size += child_size
    
    dfs(0)
    return max(dp[0])
```

---

## 11.5 换根DP（Re-rooting Technique）

**问题**：当题目要求"以每个节点为根时的某个值"，且每次重新 DFS 太慢时，使用换根DP。

**思路**：两次DFS
1. **第一次DFS**（任意选根，如节点0）：计算每个节点的"向下"信息
2. **第二次DFS**：利用父节点的完整信息，推导子节点"向上"信息，合并得到以该节点为根的答案

### 经典题：所有节点到根的距离之和（LeetCode 834）

**题目**：给定无权树，求每个节点到所有其他节点的距离之和。

**朴素解法**：对每个节点做一次BFS，O(n²)，n=3×10⁴ 时超时。

**换根DP**：O(n)

```python
def sum_of_distances_in_tree(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    count = [1] * n  # count[v] = 以v为根的子树节点数
    ans   = [0] * n  # ans[v]   = v到其子树所有节点的距离和
    
    # 第一次DFS：计算 count 和 ans（以0为根）
    def dfs1(v, parent):
        for child in adj[v]:
            if child == parent:
                continue
            dfs1(child, v)
            count[v] += count[child]
            ans[v] += ans[child] + count[child]
    
    # 第二次DFS：换根，利用父节点的答案推导子节点答案
    def dfs2(v, parent):
        for child in adj[v]:
            if child == parent:
                continue
            # 从 v 换根到 child：
            # - child 子树中的节点距离都减1（有 count[child] 个）
            # - 其余节点（n - count[child] 个）距离都加1
            ans[child] = ans[v] - count[child] + (n - count[child])
            dfs2(child, v)
    
    dfs1(0, -1)
    dfs2(0, -1)
    return ans

# 测试
print(sum_of_distances_in_tree(6, [[0,1],[0,2],[2,3],[2,4],[2,5]]))
# [8, 12, 6, 10, 10, 10]
```

**换根公式推导**：
```
ans[child] = ans[v] + (n - count[child]) - count[child]
           = ans[v] + n - 2*count[child]
```

含义：
- 原来从 v 出发，到 child 子树中的节点距离减少了 count[child]（它们都近了1步）
- 到其余 n-count[child] 个节点距离增加了1（它们都远了1步）

---

## 11.6 树的直径（两次DFS或换根DP）

**题目**：找树中最长的路径（直径）。

**方法1：两次DFS**

```python
def tree_diameter(n, edges):
    adj = [[] for _ in range(n)]
    for u, v, w in edges:  # 带权图
        adj[u].append((v, w))
        adj[v].append((u, w))
    
    def farthest(start):
        dist = [-1] * n
        dist[start] = 0
        stack = [start]
        farthest_node = start
        while stack:
            v = stack.pop()
            for u, w in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + w
                    stack.append(u)
                    if dist[u] > dist[farthest_node]:
                        farthest_node = u
        return farthest_node, dist[farthest_node]
    
    # 任意节点出发找最远点，再从最远点出发找直径
    far1, _ = farthest(0)
    far2, diameter = farthest(far1)
    return diameter
```

**方法2：DFS过程中维护直径**

```python
def tree_diameter_dfs(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    diameter = [0]
    
    def dfs(v, parent):
        # 返回从 v 出发向下的最长路径
        max_depth = 0
        for child in adj[v]:
            if child == parent:
                continue
            child_depth = dfs(child, v) + 1
            # 通过 v 的路径长度 = 当前最长 + child_depth
            diameter[0] = max(diameter[0], max_depth + child_depth)
            max_depth = max(max_depth, child_depth)
        return max_depth
    
    dfs(0, -1)
    return diameter[0]
```

---

## 11.7 树形DP综合：监控摄像头（LeetCode 968）

**题目**：给二叉树每个节点装监控或不装，每个摄像头可覆盖父/子节点。求覆盖所有节点的最少摄像头数量。

**状态**：
- `dp[v][0]`：v 未被覆盖（子树都覆盖，但 v 自己没有）
- `dp[v][1]`：v 被覆盖但没有摄像头（由子节点或父节点覆盖）
- `dp[v][2]`：v 安装了摄像头

```python
def min_camera_cover(root):
    NOT_COVERED = 0
    COVERED = 1
    HAS_CAMERA = 2
    
    def dp(node):
        if not node:
            return float('inf'), 0, float('inf')
        # (not_covered, covered_no_cam, has_camera)
        
        l = dp(node.left)
        r = dp(node.right)
        
        # 当前节点安装摄像头：子节点无论什么状态都可以
        has_camera = min(l) + min(r) + 1
        
        # 当前节点被覆盖（无摄像头）：至少一个子节点有摄像头
        covered = min(
            l[HAS_CAMERA] + min(r[COVERED], r[HAS_CAMERA]),
            r[HAS_CAMERA] + min(l[COVERED], l[HAS_CAMERA])
        )
        
        # 当前节点未被覆盖：子节点都被覆盖但无摄像头
        not_covered = min(l[COVERED], l[HAS_CAMERA]) + min(r[COVERED], r[HAS_CAMERA])
        
        return not_covered, covered, has_camera
    
    result = dp(root)
    return min(result[COVERED], result[HAS_CAMERA])
```

---

## 11.8 本章小结

| 技巧 | 适用场景 |
|------|---------|
| 后序DFS | 大多数树形DP，子节点信息汇总到父节点 |
| 换根DP（两次DFS） | 需要以每个节点为根的答案 |
| 树上背包 | 树上有容量约束的选择问题 |
| 多状态定义 | 节点有多种状态（选/不选/覆盖/...） |

**下一章：状压DP——用二进制表示集合状态**

---

## LeetCode 推荐题目

- [337. 打家劫舍 III](https://leetcode.cn/problems/house-robber-iii/) ⭐⭐
- [834. 树中距离之和](https://leetcode.cn/problems/sum-of-distances-in-tree/) ⭐⭐⭐
- [968. 监控二叉树](https://leetcode.cn/problems/binary-tree-cameras/) ⭐⭐⭐
- [124. 二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/) ⭐⭐⭐
- [1245. 树的直径](https://leetcode.cn/problems/tree-diameter/) ⭐⭐
