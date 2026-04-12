# 第20章：计数DP与组合数学

## 20.1 计数DP的本质

**计数DP** 解决的问题形如：

> 满足某种条件的方案（组合、排列、划分）共有多少种？

与最优化DP的区别：
- 最优化DP：`dp[state] = min/max 某个量`
- 计数DP：`dp[state] = 方案数`，转移时用加法而非 min/max

---

## 20.2 组合数的动态规划求法

### 帕斯卡三角形（杨辉三角）

```python
def generate_pascal(n):
    # C[i][j] = C(i, j) = 从 i 个中选 j 个的组合数
    C = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        C[i][0] = 1
        for j in range(1, i + 1):
            C[i][j] = C[i-1][j-1] + C[i-1][j]  # 帕斯卡恒等式
    return C

C = generate_pascal(20)
print(C[6][2])   # 15 = C(6,2)
print(C[10][5])  # 252
```

### 带模运算的大组合数

当 n 很大时（如 n ≤ 10^6），用费马小定理求逆元：

```python
MOD = 10**9 + 7

def precompute_factorials(max_n):
    fact = [1] * (max_n + 1)
    inv_fact = [1] * (max_n + 1)
    
    for i in range(1, max_n + 1):
        fact[i] = fact[i-1] * i % MOD
    
    # 费马小定理：a^(p-1) ≡ 1 (mod p)，故 a^(-1) ≡ a^(p-2) (mod p)
    inv_fact[max_n] = pow(fact[max_n], MOD - 2, MOD)
    for i in range(max_n - 1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
    
    return fact, inv_fact

fact, inv_fact = precompute_factorials(10**6)

def comb(n, k):
    if k < 0 or k > n:
        return 0
    return fact[n] * inv_fact[k] % MOD * inv_fact[n-k] % MOD
```

---

## 20.3 划分问题

### 整数划分

**题目**：将整数 n 拆分为若干正整数之和，有多少种方式？（顺序不同视为相同）

```
n=4: 4 = 3+1 = 2+2 = 2+1+1 = 1+1+1+1，共5种
```

这是**完全背包的计数版**：

```python
def partition_count(n):
    # dp[i] = 整数 i 的划分数
    dp = [0] * (n + 1)
    dp[0] = 1
    
    for k in range(1, n + 1):  # 每个正整数作为"物品"（完全背包）
        for i in range(k, n + 1):
            dp[i] += dp[i - k]
    
    return dp[n]

for i in range(1, 11):
    print(f"p({i}) = {partition_count(i)}")
# 1, 2, 3, 5, 7, 11, 15, 22, 30, 42
```

### 有约束的划分

```python
def partition_into_k_parts(n, k):
    """将 n 划分为恰好 k 个正整数之和的方案数"""
    # dp[i][j] = 将 i 划分为 j 个正整数的方案数
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            for part in range(1, i - j + 2):  # 最小部分为1
                dp[i][j] += dp[i - part][j - 1]
    
    return dp[n][k]
```

---

## 20.4 卡特兰数

卡特兰数是计数DP中最重要的序列之一：

**C(0)=1, C(1)=1, C(2)=2, C(3)=5, C(4)=14, C(5)=42, ...**

**递推关系**：`C(n) = Σ C(k) * C(n-1-k)，k=0..n-1`

**封闭形式**：`C(n) = C(2n, n) / (n+1)`

**数十种组合意义**（均等于 C(n)）：
- n+1 个节点的二叉树数量
- n 个括号对的合法括号序列数
- n×n 格子从左下到右上不超过对角线的路径数
- 凸 (n+2) 边形的三角剖分方案数

```python
def catalan(n, mod=10**9+7):
    # C(n) = C(2n, n) / (n+1)
    return comb(2*n, n) * pow(n+1, mod-2, mod) % mod

# 验证
for i in range(8):
    print(f"C({i}) = {catalan(i)}")
# 1, 1, 2, 5, 14, 42, 132, 429
```

**应用：合法括号序列数（LeetCode 22）**

```python
def generate_parenthesis(n):
    """生成所有合法括号序列（回溯 + 计数）"""
    result = []
    
    def backtrack(s, open_count, close_count):
        if len(s) == 2 * n:
            result.append(s)
            return
        if open_count < n:
            backtrack(s + '(', open_count + 1, close_count)
        if close_count < open_count:
            backtrack(s + ')', open_count, close_count + 1)
    
    backtrack('', 0, 0)
    return result  # len(result) == catalan(n)
```

---

## 20.5 斯特林数

**第二类斯特林数** S(n, k)：将 n 个不同元素划分为 k 个非空子集的方案数。

```
递推：S(n, k) = k * S(n-1, k) + S(n-1, k-1)
含义：第 n 个元素要么加入已有 k 个子集之一（k 种选择），
      要么自己单独成为新子集（从 n-1 个元素分成 k-1 组的基础上）
```

```python
def stirling_second(max_n):
    S = [[0] * (max_n + 1) for _ in range(max_n + 1)]
    S[0][0] = 1
    
    for n in range(1, max_n + 1):
        for k in range(1, n + 1):
            S[n][k] = k * S[n-1][k] + S[n-1][k-1]
    
    return S

S = stirling_second(6)
print(S[4][2])  # 7（4个元素分成2组的方式）
```

**贝尔数** B(n)：n 个元素的所有划分方案总数：
`B(n) = Σ S(n, k)，k=0..n`

---

## 20.6 容斥原理与DP

**容斥原理**：|A∪B∪C| = |A|+|B|+|C| - |A∩B| - |A∩C| - |B∩C| + |A∩B∩C|

在计数DP中，容斥原理用于处理"禁止某些情况"的问题。

### 错位排列

**题目**：n 封信放入 n 个信封，没有信放入对应信封的方案数（错位排列 D(n)）。

```python
def derangement(n):
    if n == 0: return 1
    if n == 1: return 0
    if n == 2: return 1
    
    # D(n) = (n-1) * (D(n-1) + D(n-2))
    # 含义：第n封信放在第k个信封（k != n，有n-1种选择）
    # 之后第n封信与第k个位置的信互换问题：
    # - 如果第k封信不放第n个信封：D(n-1)
    # - 如果第k封信放第n个信封：D(n-2)
    
    a, b = 1, 0  # D(0), D(1)
    for i in range(2, n + 1):
        a, b = b, (i - 1) * (a + b)
    return b

for i in range(8):
    print(f"D({i}) = {derangement(i)}")
# 1, 0, 1, 2, 9, 44, 265, 1854
```

### 染色问题（容斥DP）

**题目**（LeetCode 1349）：学生坐在椅子上，每行最多一个学生坐，相邻椅子不能同坐，求最多能安排多少学生。

这是一个状压DP + 计数问题，详见第12章。

---

## 20.7 格路计数

**题目**：从 (0,0) 到 (m,n) 的路径，只能向右或向上，且不能经过某些禁止点，有多少条？

```python
def grid_paths_with_obstacles(m, n, obstacles):
    # dp[i][j] = 到达 (i,j) 的路径数
    obstacle_set = set(map(tuple, obstacles))
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 1
    
    for i in range(m + 1):
        for j in range(n + 1):
            if (i, j) in obstacle_set:
                dp[i][j] = 0
                continue
            if i > 0: dp[i][j] += dp[i-1][j]
            if j > 0: dp[i][j] += dp[i][j-1]
    
    return dp[m][n]
```

**Lindström-Gessel-Viennot 引理**：非交叉路径的计数，用行列式计算。

---

## 20.8 综合：统计特殊子序列数（LeetCode 940）

**题目**：给定字符串，统计不同的非空子序列个数。

```python
def distinct_subsequences_ii(s):
    MOD = 10**9 + 7
    
    # dp[i] = 以 s[i] 结尾的不同子序列数
    dp = {}
    total = 0
    
    for c in s:
        # 以 c 结尾的子序列 = 在所有已有子序列后加 c + 单独的 c
        # 但要减去之前已有的以 c 结尾的子序列（避免重复）
        new = (total + 1) % MOD  # total 个已有子序列各加一个c，再加c本身
        total = (total + new - dp.get(c, 0)) % MOD
        dp[c] = new
    
    return total

print(distinct_subsequences_ii("abc"))  # 7
print(distinct_subsequences_ii("aba"))  # 6
```

---

## 20.9 综合提升：LeetCode 1987

**题目**：你有 n 类不同的球，每类 k 个，从中选恰好 m 个球的方案数。

这涉及**组合数的容斥**：

```python
def count_balls(n, k, m, mod=10**9+7):
    """
    n 类球，每类最多 k 个，选 m 个的方案数
    = C(n+m-1, m) - n*C(n+m-k-2, m-k-1) + C(n,2)*C(n+m-2k-3, m-2k-2) - ...
    """
    # 容斥：先不限制每类数量，再减去违规的
    # 复杂度取决于具体约束
    pass
```

---

## 20.10 本章小结与全书回顾

**计数DP的核心工具**：

| 工具 | 适用场景 |
|------|---------|
| 帕斯卡三角形 | 组合数，C(n,k) |
| 完全背包计数 | 整数划分，无序选择 |
| 卡特兰数 | 括号、树、路径等结构计数 |
| 斯特林数 | 集合划分 |
| 容斥原理 | 禁止某些情况的计数 |
| 数位DP | 按位构造满足条件的数 |

---

## 全书总结：动态规划的思维框架

```
遇到DP问题，按以下步骤思考：

1. 识别DP特征
   → 最优子结构？重叠子问题？无后效性？

2. 设计状态
   → dp[i] 或 dp[i][j] 表示什么？
   → 维度是否足够？是否满足无后效性？

3. 推导转移方程
   → "最后一步"是什么？
   → 枚举所有可能的来源

4. 确定base case
   → 最小子问题的答案是什么？

5. 确定填表顺序
   → 依赖关系是否满足？

6. 优化（可选）
   → 空间压缩？单调队列？斜率优化？矩阵快速幂？

7. 提取答案
   → 目标状态在哪里？
```

**各类DP的应用场景速查**：

| 类型 | 典型特征 | 代表题目 |
|------|---------|---------|
| 线性DP | 数组/字符串，一维状态 | 打家劫舍、最大子数组 |
| 背包DP | 选或不选，容量约束 | 零钱兑换、分割子集 |
| 序列DP | 两个序列，子序列/子串 | LCS、编辑距离 |
| 区间DP | 合并/分割区间 | 矩阵链乘、戳气球 |
| 树形DP | 树结构，子树信息合并 | 打家劫舍III、换根 |
| 状压DP | 集合状态，n≤20 | TSP、优美排列 |
| 数位DP | 按位构造数字 | 数字1的个数 |
| 博弈DP | 双方最优策略 | 石子游戏 |
| 概率DP | 随机过程，概率/期望 | 骑士概率 |
| 计数DP | 方案计数 | 不同路径数 |

---

## LeetCode 推荐题目

- [96. 不同的二叉搜索树](https://leetcode.cn/problems/unique-binary-search-trees/) ⭐⭐（卡特兰数）
- [940. 不同的子序列 II](https://leetcode.cn/problems/distinct-subsequences-ii/) ⭐⭐⭐
- [1155. 掷骰子的N种方法](https://leetcode.cn/problems/number-of-dice-rolls-with-target-sum/) ⭐⭐
- [2338. 统计理想数组的数目](https://leetcode.cn/problems/count-the-number-of-ideal-arrays/) ⭐⭐⭐⭐
- [730. 统计不同回文子序列](https://leetcode.cn/problems/count-different-palindromic-subsequences/) ⭐⭐⭐

---

*至此，本教程全部20章内容完结。从斐波那契到斜率优化，从基础线性DP到博弈论与概率DP，你已经掌握了动态规划的完整知识体系。继续在LeetCode上刷题，将这些知识内化为直觉！*
