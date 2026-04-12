# 第19章：最近公共祖先（LCA）

## 19.1 问题定义

**最近公共祖先（Lowest Common Ancestor，LCA）**：

给定一棵树和两个节点 u、v，找到它们的**最近公共祖先**——即既是 u 的祖先也是 v 的祖先的节点中，**深度最大**的那个。

```
         1
        / \
       2   3
      / \
     4   5
        / \
       6   7

LCA(4, 5) = 2
LCA(4, 6) = 2
LCA(6, 3) = 1
LCA(4, 4) = 4（节点是自身的祖先）
```

**应用**：
- 路径查询：u 到 v 的路径 = u→LCA(u,v)→v
- 路径长度 = depth(u) + depth(v) - 2 × depth(LCA)
- 树上差分

## 19.2 朴素算法 O(n)

```python
def lca_naive(root, p, q):
    """
    朴素 LCA：后序遍历
    
    如果当前节点是 p 或 q，直接返回
    左右子树各自搜索：
    - 两边各找到一个 → 当前节点是 LCA
    - 只有一边找到 → 返回那边
    - 两边都没找到 → 返回 None
    
    时间 O(n)，空间 O(h)
    """
    if root is None:
        return None
    if root == p or root == q:
        return root  # 找到其中一个，直接返回（包含祖先关系）
    
    left = lca_naive(root.left, p, q)
    right = lca_naive(root.right, p, q)
    
    if left and right:
        return root  # 两边各找到一个，当前节点是 LCA
    return left or right  # 只有一边找到

# 优点：简单直接
# 缺点：每次查询 O(n)，多次查询效率低
```

## 19.3 倍增 LCA（Binary Lifting）O(n log n) 预处理 + O(log n) 查询

适合**多次 LCA 查询**的场景：

```python
import math

class LCA_BinaryLifting:
    """
    倍增 LCA 算法
    预处理：O(n log n)
    每次查询：O(log n)
    
    核心思想：
    ancestor[v][k] = v 的 2^k 级祖先
    通过二进制跳跃快速找到 LCA
    """
    
    def __init__(self, n, edges, root=0):
        self.n = n
        self.LOG = max(1, math.ceil(math.log2(n + 1)))
        self.depth = [0] * n
        self.ancestor = [[-1] * (self.LOG + 1) for _ in range(n)]
        
        # 建邻接表
        from collections import defaultdict, deque
        self.graph = defaultdict(list)
        for u, v in edges:
            self.graph[u].append(v)
            self.graph[v].append(u)
        
        # BFS 预处理深度和直接父节点
        visited = [False] * n
        queue = deque([root])
        visited[root] = True
        self.ancestor[root][0] = root  # 根的父节点设为自身
        
        while queue:
            u = queue.popleft()
            for v in self.graph[u]:
                if not visited[v]:
                    visited[v] = True
                    self.depth[v] = self.depth[u] + 1
                    self.ancestor[v][0] = u  # 直接父节点
                    queue.append(v)
        
        # DP 预处理 2^k 级祖先
        for k in range(1, self.LOG + 1):
            for v in range(n):
                if self.ancestor[v][k-1] != -1:
                    self.ancestor[v][k] = self.ancestor[self.ancestor[v][k-1]][k-1]
    
    def query(self, u, v):
        """O(log n) 查询 LCA(u, v)"""
        # 确保 u 比 v 深
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        
        # 步骤1：将 u 提升到与 v 同层
        diff = self.depth[u] - self.depth[v]
        for k in range(self.LOG + 1):
            if (diff >> k) & 1:
                u = self.ancestor[u][k]
        
        if u == v:
            return u  # v 是 u 的祖先
        
        # 步骤2：同时向上跳跃，找到 LCA 的子节点
        for k in range(self.LOG, -1, -1):
            if (self.ancestor[u][k] != self.ancestor[v][k]):
                u = self.ancestor[u][k]
                v = self.ancestor[v][k]
        
        return self.ancestor[u][0]  # 父节点即为 LCA
    
    def distance(self, u, v):
        """u 到 v 的路径长度"""
        lca = self.query(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[lca]

# 测试
n = 7
edges = [(0,1),(0,2),(1,3),(1,4),(4,5),(4,6)]
lca = LCA_BinaryLifting(n, edges, root=0)
print(lca.query(3, 5))   # 1
print(lca.query(5, 6))   # 4
print(lca.query(3, 6))   # 1
print(lca.distance(3, 6)) # depth(3)+depth(6)-2*depth(1) = 2+3-2*1 = 3
```

## 19.4 Tarjan 离线 LCA O(n + q)

当所有查询**提前已知**时，可以使用 Tarjan 算法（并查集 + DFS）一次处理所有查询：

```python
class LCA_Tarjan:
    """
    Tarjan 离线 LCA
    时间 O(n + q * α(n)) ≈ O(n + q)
    
    思路：DFS 过程中，当从子树返回时，
    将子树节点并入父节点的集合
    若此时某查询的另一端已被访问，则 LCA = 另一端所在集合的根
    """
    
    def __init__(self, n, edges):
        from collections import defaultdict
        self.graph = defaultdict(list)
        for u, v in edges:
            self.graph[u].append(v)
            self.graph[v].append(u)
        
        self.parent = list(range(n))
        self.visited = [False] * n
        self.answers = {}
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)
    
    def solve(self, root, queries):
        """
        queries: [(u, v)] 列表
        返回每个查询的 LCA
        """
        from collections import defaultdict
        query_map = defaultdict(list)
        for i, (u, v) in enumerate(queries):
            query_map[u].append((v, i))
            query_map[v].append((u, i))
        
        results = [None] * len(queries)
        
        def dfs(u, parent):
            self.visited[u] = True
            
            for v in self.graph[u]:
                if v != parent:
                    dfs(v, u)
                    self.union(v, u)  # 合并子树到父节点
            
            # 处理以 u 为端点的查询
            for (v, idx) in query_map[u]:
                if self.visited[v]:
                    results[idx] = self.find(v)
        
        dfs(root, -1)
        return results

# 测试
n = 7
edges = [(0,1),(0,2),(1,3),(1,4),(4,5),(4,6)]
queries = [(3,5), (5,6), (3,6), (0,6)]

tarjan = LCA_Tarjan(n, edges)
print(tarjan.solve(0, queries))  # [1, 4, 1, 0]
```

## 19.5 树上路径问题（LCA 应用）

### 路径上的节点值之和

```python
def path_sum(node_vals, depth, ancestor, lca_solver, u, v):
    """
    u 到 v 路径上的节点值之和
    
    利用前缀和：从根到每个节点的前缀和
    path_sum(u, v) = prefix[u] + prefix[v] - prefix[lca] - prefix[parent(lca)]
    """
    lca = lca_solver.query(u, v)
    lca_parent = ancestor[lca][0]  # lca 的父节点
    
    return (node_vals[u] + node_vals[v] 
            - node_vals[lca] - (node_vals[lca_parent] if lca_parent != lca else 0))
```

### 树上差分（批量路径修改）

```python
def path_update_all(n, edges, paths, delta):
    """
    对多条路径上的所有节点加 delta
    用树上差分 + LCA：
    diff[u] += delta
    diff[v] += delta
    diff[lca] -= delta
    diff[parent(lca)] -= delta
    
    最后 DFS 求差分前缀和，即为每个节点的实际增量
    """
    diff = [0] * n
    
    for (u, v) in paths:
        lca = query_lca(u, v)  # 假设已实现
        par_lca = parent[lca]
        diff[u] += delta
        diff[v] += delta
        diff[lca] -= delta
        if par_lca != lca:
            diff[par_lca] -= delta
    
    # DFS 求子树差分和
    result = diff[:]
    def dfs(u, par):
        for v in adj[u]:
            if v != par:
                dfs(v, u)
                result[u] += result[v]
    
    dfs(0, -1)
    return result
```

## 19.6 三种算法对比

| 算法 | 预处理 | 单次查询 | 适用场景 |
|------|--------|---------|---------|
| 朴素递归 | O(n) | O(n) | 查询少，代码简单 |
| 倍增 | O(n log n) | O(log n) | 在线查询，最常用 |
| Tarjan 离线 | O(n + q) | - | 所有查询已知，极高效 |
| Euler Tour + RMQ | O(n log n) | O(1) | 查询极多（竞赛） |

## 19.7 LCA 与 RMQ 的等价性

LCA 问题可以转化为 **区间最小值查询（RMQ）** 问题：

1. 对树进行欧拉游（Euler Tour），得到 DFS 访问序列
2. 两节点 u、v 的 LCA = 欧拉序中 u 到 v 之间**深度最小**的节点

```python
def lca_via_euler_tour(root, n):
    """
    用欧拉游 + 稀疏表实现 O(1) LCA 查询
    预处理 O(n log n)
    """
    euler = []       # 欧拉序（节点）
    depth_arr = []   # 对应深度
    first = {}       # 每个节点首次出现位置
    
    def dfs(node, par, d):
        first[node] = len(euler)
        euler.append(node)
        depth_arr.append(d)
        
        for child in [node.left, node.right]:
            if child and child != par:
                dfs(child, node, d + 1)
                euler.append(node)   # 回溯时再次记录
                depth_arr.append(d)
    
    dfs(root, None, 0)
    
    # 构建稀疏表（区间最小深度）
    m = len(euler)
    LOG = max(1, m.bit_length())
    sparse = [[float('inf')] * m for _ in range(LOG)]
    sparse[0] = depth_arr[:]
    
    for k in range(1, LOG):
        for i in range(m - (1 << k) + 1):
            if sparse[k-1][i] <= sparse[k-1][i + (1 << (k-1))]:
                sparse[k][i] = sparse[k-1][i]
            else:
                sparse[k][i] = sparse[k-1][i + (1 << (k-1))]
    
    def query_lca(u, v):
        l, r = first[u], first[v]
        if l > r:
            l, r = r, l
        length = r - l + 1
        k = length.bit_length() - 1
        # 找区间最小深度对应的节点
        if sparse[k][l] <= sparse[k][r - (1 << k) + 1]:
            min_depth_idx = l  # 简化，实际需追踪下标
        else:
            min_depth_idx = r - (1 << k) + 1
        return euler[min_depth_idx]
    
    return query_lca
```

## 小结

- LCA 是树上路径问题的关键
- 朴素算法 O(n)，倍增 O(log n) 查询，Tarjan 离线 O(1) 均摊
- **倍增 LCA 最常用**（在线，实现简洁）
- 路径长度 = depth(u) + depth(v) - 2·depth(LCA)
- 树上差分：高效处理路径批量修改

## 练习

1. 实现"树中两节点间的所有路径节点"的遍历
2. 用倍增 LCA 解决"k 步能否从 u 到达 v"
3. 实现树上差分，统计每条边被路径覆盖的次数

---

**上一章：** [树形 DP](04-tree-dp.md)

---

## 结语：二叉树学习路线回顾

```
Part 1 基础        → 节点/术语/分类/存储
Part 2 遍历        → DFS递归/迭代/BFS/Morris
Part 3 BST         → 概念/增删查/应用变体
Part 4 平衡树      → AVL/红黑树/B树
Part 5 高级专题    → 线段树/BIT/Trie/树形DP/LCA
```

**竞赛常用组合**：
- 区间问题 → 线段树 / BIT
- 字符串前缀 → Trie
- 树上路径 → LCA + 树上差分
- 树上统计 → 树形 DP + 换根

*恭喜完成全部 20 章！*
