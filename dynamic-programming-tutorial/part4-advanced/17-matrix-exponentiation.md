# 第17章：矩阵快速幂优化线性递推

## 17.1 问题动机

某些线性递推关系：

```
f(n) = a₁·f(n-1) + a₂·f(n-2) + ... + aₖ·f(n-k)
```

用普通迭代求 f(n) 需要 O(n) 时间。但当 n 高达 10^18 时，O(n) 完全不可行，需要 **O(k³ log n)** 的矩阵快速幂。

---

## 17.2 矩阵乘法基础

**矩阵乘法**：C = A × B，其中 `C[i][j] = Σ A[i][k] * B[k][j]`

```python
def mat_mul(A, B, mod=None):
    n = len(A)
    m = len(B[0])
    k = len(B)
    C = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for p in range(k):
                C[i][j] += A[i][p] * B[p][j]
            if mod:
                C[i][j] %= mod
    return C
```

**矩阵快速幂**：计算 A^n

```python
def mat_pow(A, n, mod=None):
    size = len(A)
    # 单位矩阵
    result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
    
    while n > 0:
        if n & 1:
            result = mat_mul(result, A, mod)
        A = mat_mul(A, A, mod)
        n >>= 1
    
    return result
```

**时间复杂度**：O(k³ log n)，其中 k 是矩阵大小。

---

## 17.3 斐波那契的矩阵表达

将递推关系转化为矩阵乘法：

```
[F(n+1)]   [1 1] [F(n)  ]
[F(n)  ] = [1 0] [F(n-1)]
```

即：

```
v(n) = M · v(n-1) = M² · v(n-2) = ... = M^(n-1) · v(1)
```

```python
def fib_matrix(n, mod=10**9+7):
    if n <= 1:
        return n
    
    M = [[1, 1], [1, 0]]
    result = mat_pow(M, n - 1, mod)
    
    # v(1) = [F(1), F(0)] = [1, 0]
    return result[0][0]  # F(n) = M^(n-1)[0][0] * 1 + M^(n-1)[0][1] * 0

print(fib_matrix(10))   # 55
print(fib_matrix(50))   # 12586269025
```

---

## 17.4 一般线性递推的矩阵化

对于 k 阶线性递推：

```
f(n) = a₁·f(n-1) + a₂·f(n-2) + ... + aₖ·f(n-k)
```

构造 **伴随矩阵（Companion Matrix）**：

```
[f(n)  ]   [a₁ a₂ a₃ ... aₖ] [f(n-1)]
[f(n-1)]   [1  0  0  ...  0] [f(n-2)]
[f(n-2)] = [0  1  0  ...  0] [f(n-3)]
[  ...  ]   [0  0  1  ...  0] [  ...  ]
[f(n-k+1)] [0  0  0  ...  0] [f(n-k)]
```

```python
def linear_recurrence(coeffs, init_vals, n, mod=10**9+7):
    """
    coeffs: [a₁, a₂, ..., aₖ]（从高阶到低阶）
    init_vals: [f(0), f(1), ..., f(k-1)]
    求 f(n)
    """
    k = len(coeffs)
    
    if n < k:
        return init_vals[n]
    
    # 构造伴随矩阵
    M = [[0] * k for _ in range(k)]
    for j in range(k):
        M[0][j] = coeffs[j]  # 第一行是系数
    for i in range(1, k):
        M[i][i-1] = 1  # 次对角线
    
    # 计算 M^(n-k+1)
    Mn = mat_pow(M, n - k + 1, mod)
    
    # 初始向量 v = [f(k-1), f(k-2), ..., f(0)]
    result = 0
    for j in range(k):
        result = (result + Mn[0][j] * init_vals[k-1-j]) % mod
    
    return result

# 斐波那契：f(n) = f(n-1) + f(n-2)，初始值 f(0)=0, f(1)=1
print(linear_recurrence([1, 1], [0, 1], 10))  # 55
```

---

## 17.5 实战：铺砖问题

**题目**：用 1×2 的砖块铺满 2×n 的地板，有多少种方法？

递推关系：`f(n) = f(n-1) + f(n-2)`（就是斐波那契！）

```python
def tiling(n, mod=10**9+7):
    return linear_recurrence([1, 1], [1, 1], n, mod)

print(tiling(10))  # 89
```

---

## 17.6 实战：带状态的DP加速

某些DP问题，每一步的状态转移是固定的线性变换，可以用矩阵快速幂加速。

**例：恰好走 k 步从节点 s 到节点 t 的路径数**

设邻接矩阵为 A，则走恰好 k 步的路径数 = `A^k[s][t]`。

```python
def count_paths(adj_matrix, s, t, k, mod=10**9+7):
    """
    adj_matrix: 邻接矩阵
    求从 s 到 t 恰好走 k 步的路径数
    """
    Ak = mat_pow(adj_matrix, k, mod)
    return Ak[s][t]

# 例：4节点完全图，走3步从节点0到节点0的路径数
adj = [[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]]
print(count_paths(adj, 0, 0, 3, 10**9+7))
```

---

## 17.7 实战：字符串计数（LeetCode 1220）

**题目**：长度为 n 的元音字符串（a,e,i,o,u），相邻字符满足顺序约束，有多少种？

```python
def count_vowel_permutation(n):
    MOD = 10**9 + 7
    # a 后面只能是 e
    # e 后面只能是 a 或 i
    # i 后面不能跟 i
    # o 后面只能是 i 或 u
    # u 后面只能是 a
    
    # 转移矩阵：M[i][j] = 从 j 可以转移到 i（方便矩阵乘法）
    # 顺序：a=0, e=1, i=2, o=3, u=4
    M = [
        [0, 1, 1, 0, 1],  # a 可以由 e, i, u 转来
        [1, 0, 1, 0, 0],  # e 可以由 a, i 转来
        [0, 1, 0, 1, 0],  # i 可以由 e, o 转来
        [0, 0, 1, 0, 0],  # o 可以由 i 转来
        [0, 0, 1, 1, 0],  # u 可以由 i, o 转来
    ]
    
    if n == 1:
        return 5
    
    Mn = mat_pow(M, n - 1, MOD)
    
    # 初始：每个元音各1个
    result = 0
    for i in range(5):
        for j in range(5):
            result = (result + Mn[i][j]) % MOD
    
    return result

print(count_vowel_permutation(1))   # 5
print(count_vowel_permutation(2))   # 10
print(count_vowel_permutation(5))   # 68
```

---

## 17.8 矩阵快速幂的本质

矩阵快速幂 = **将线性变换用矩阵表示** + **快速幂（重复平方）**

适用条件：
1. 状态转移是**线性的**（可以写成矩阵乘法）
2. n 很大（如 10^18），需要 O(log n)
3. 状态维数 k 不太大（k ≤ 100，因为矩阵乘法是 O(k³)）

---

## 17.9 本章小结

| 问题类型 | 矩阵表示 | 复杂度 |
|---------|---------|-------|
| k阶线性递推 | k×k伴随矩阵 | O(k³ log n) |
| 图上路径计数 | 邻接矩阵 | O(V³ log k) |
| 有限状态机DP | 转移矩阵 | O(S³ log n) |

---

## LeetCode 推荐题目

- [509. 斐波那契数](https://leetcode.cn/problems/fibonacci-number/) ⭐（用矩阵快速幂实现）
- [1220. 统计元音字母序列的数目](https://leetcode.cn/problems/count-vowels-permutation/) ⭐⭐⭐
- [790. 多米诺和托米诺平铺](https://leetcode.cn/problems/domino-and-tromino-tiling/) ⭐⭐⭐
- [935. 骑士拨号器](https://leetcode.cn/problems/knight-dialer/) ⭐⭐
