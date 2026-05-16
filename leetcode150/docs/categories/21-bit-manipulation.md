# 21 — Bit Manipulation（融合版）

> **难度**：★★☆☆☆
> **题数**：6
> **核心套路**：XOR 消消乐（单数）、位计数（hamming weight / popcount）、位运算技巧（最低位 / 反转 / 区间公共前缀）
> **本文件**：覆盖 bit_manipulation 6 题的算法套路总结 + 典型题精讲 + 自测

---

## 一例速记

> **XOR 消消乐（Single Number I）**：全部异或，相同数对消为 0，结果即为出现奇数次的数（136）
> **Single Number II（每数出现 3 次）**：对每一位统计 `1` 的个数，模 3 取余，余数即为目标数对应位（137）；或用两个变量 `ones`/`twos` 模拟三进制计数器
> **Hamming Weight（位计数）**：`n & (n-1)` 消去最低有效位，循环次数 = 1 的个数（191）；或 `bin(n).count('1')`；或 Brian Kernighan 算法
> **Number of 1 Bits 进阶**：338 Counting Bits，$O(n)$ 用 DP：`dp[i] = dp[i >> 1] + (i & 1)`（右移一位的结果已知，加上当前最低位）
> **Reverse Bits**：逐位取最低位，左移到目标位置，累计拼接（190）；或分治翻转（16→8→4→2→1 位块交换）
> **Bitwise AND of Range**：两数一直右移到相等（即找公共前缀），左移回去即为答案（201）；等价于 Brian Kernighan：不断用 `right & (right-1)` 消去 right 最低位直到 `right <= left`
> **Add Binary**：从低位逐位模拟加法进位（67）；或 Python 直接 `bin(int(a,2) + int(b,2))[2:]`
> **AI 关联**：哈希压缩（MinHash / SimHash 用位运算加速）/ 集合编码（bitmap / bitset）/ CUDA 原子操作 / 量化模型中的整数位运算（INT8 / INT4 推理）

---

## 思维路径还原

> "看到 **'136 只有一个数出现一次，其余出现两次'** → XOR 消消乐：
> `result = 0; for x in nums: result ^= x`，$O(n)$ 时间 $O(1)$ 空间。
> 性质：`x ^ x = 0`，`x ^ 0 = x`，XOR 满足交换律和结合律。
>
> 看到 **'137 只有一个数出现一次，其余出现三次'** → 不能直接用 XOR（3 次不对消）。
> 方法 1：对 32 位每一位统计 1 的个数，`count[bit] % 3` 就是答案对应位（$O(32n)$）。
> 方法 2：`ones`/`twos` 两个变量，模拟每一位的三进制计数器：
>   `ones = (ones ^ x) & ~twos`
>   `twos = (twos ^ x) & ~ones`
> 最终 `ones` 即为仅出现一次的数。
>
> 看到 **'191 位 1 的个数'** → Brian Kernighan：`while n: n &= n-1; count += 1`，
> 每次操作消去 n 最低有效位（`n & (n-1)` 将最低 1 变为 0）。
> 循环次数 = 1 的个数，比逐位移位快（跳过所有 0 位）。
>
> 看到 **'338 位计数（0~n 每个数的 1 个数）'** → DP：
> `bits[i] = bits[i >> 1] + (i & 1)`：i 右移一位得 i//2（已知），当前最低位贡献 `i & 1`。
> $O(n)$ 时间，$O(n)$ 空间（输出数组，无额外空间）。
>
> 看到 **'190 反转 32 位整数的位'** → 逐位操作：
> 右移 n 取最低位，左移到目标位置，32 轮循环。
> 注意 Python 整数无溢出，最后用 `& 0xFFFFFFFF` 截取 32 位。
>
> 看到 **'201 区间位与（left AND left+1 AND ... AND right）'** → 公共前缀法：
> left 和 right 同时右移，直到 left == right，记录右移次数 shift，结果为 `right << shift`。
> 直觉：若 left != right，则区间内必然包含相差 1 的相邻数，它们在某一位上一个为 0 一个为 1，AND 后该位为 0；一直右移到 left == right，即找到公共前缀。"

---

## 学习目标

- 掌握 XOR 消消乐原理及其在 136 / 137 中的应用与局限
- 熟练使用 `n & (n-1)` 消去最低有效位，实现 $O(k)$ 的位计数（k = 1 的个数）
- 理解 338 的 DP 位计数：右移 + 最低位，$O(n)$ 预处理所有数
- 掌握 190 的位反转：32 轮逐位提取 + 拼接，注意 Python 整数截断
- 理解 201 的公共前缀思路（同步右移直到相等）
- 能识别"位运算 / 奇偶性 / 区间 AND"题型并直接套对应模板

---

## 抽象成方法（标准模板代码）

### 套路 1：XOR 消消乐（Single Number I）

适用题：136

```python
from typing import List


def single_number(nums: List[int]) -> int:
    """
    136: 找出只出现一次的数，其余数出现恰好两次。
    XOR 性质：x ^ x = 0，x ^ 0 = x。时间 O(n)，空间 O(1)。
    """
    result = 0
    for x in nums:
        result ^= x
    return result

# 等价的函数式写法（Python）
import functools, operator
def single_number_functional(nums: List[int]) -> int:
    return functools.reduce(operator.xor, nums)
```

---

### 套路 2：三进制位计数（Single Number II）

适用题：137

```python
def single_number_ii(nums: List[int]) -> int:
    """
    137: 找出只出现一次的数，其余数出现恰好三次。
    方法 1（直观）：对每一位统计 1 的个数，mod 3 取余。时间 O(32n)，空间 O(1)。
    """
    result = 0
    for bit in range(32):
        total = sum((x >> bit) & 1 for x in nums)
        if total % 3:
            result |= 1 << bit
    # Python 需处理负数（32 位有符号）
    if result >= (1 << 31):
        result -= (1 << 32)
    return result


def single_number_ii_fast(nums: List[int]) -> int:
    """
    137: ones/twos 模拟三进制计数器，时间 O(n)，空间 O(1)。
    ones[bit] = 1 表示该位累计 1 次，twos[bit] = 1 表示累计 2 次，3 次时两者归零。
    """
    ones, twos = 0, 0
    for x in nums:
        ones = (ones ^ x) & ~twos
        twos = (twos ^ x) & ~ones
    return ones
```

---

### 套路 3：Brian Kernighan 位计数

适用题：191

```python
def hamming_weight(n: int) -> int:
    """
    191: 统计 n 的二进制表示中 1 的个数（popcount）。
    n & (n-1) 消去最低有效位，时间 O(k)，k = 1 的个数 ≤ 32。
    """
    count = 0
    while n:
        n &= n - 1     # 消去最低有效位
        count += 1
    return count


# 等价写法 1：逐位移位（O(32)，固定 32 轮）
def hamming_weight_shift(n: int) -> int:
    count = 0
    for _ in range(32):
        count += n & 1
        n >>= 1
    return count


# 等价写法 2：Python 内置（推荐实际使用）
def hamming_weight_builtin(n: int) -> int:
    return bin(n).count('1')
```

---

### 套路 4：DP 位计数（0 ~ n 所有数）

适用题：338

```python
def count_bits(n: int) -> List[int]:
    """
    338: 返回 0..n 每个数的 1 的个数。时间 O(n)，空间 O(n)（即输出数组）。
    状态转移：bits[i] = bits[i >> 1] + (i & 1)
    直觉：i 右移一位得 i//2（1 的个数已知），再加上当前最低位是否为 1。
    """
    bits = [0] * (n + 1)
    for i in range(1, n + 1):
        bits[i] = bits[i >> 1] + (i & 1)
    return bits
```

> 关键：`i >> 1 = i // 2`，对应 i 去掉最低位后的数，其 popcount 已被计算。
> 这是最优解：$O(n)$ 时间，$O(1)$ 额外空间（不含输出）。

---

### 套路 5：位反转（Reverse Bits）

适用题：190

```python
def reverse_bits(n: int) -> int:
    """
    190: 反转 32 位无符号整数的二进制位。时间 O(32)，空间 O(1)。
    """
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)   # 取 n 最低位，追加到 result 末尾
        n >>= 1
    return result & 0xFFFFFFFF              # Python 截取 32 位


# 分治翻转（高级，可缓存 16 位半字）
def reverse_bits_cache():
    """若需多次反转同一 32 位整数，可缓存 16 位半字的反转结果，O(1) 完成。"""
    cache: dict[int, int] = {}

    def _reverse_16(x: int) -> int:
        if x not in cache:
            res = 0
            for _ in range(16):
                res = (res << 1) | (x & 1)
                x >>= 1
            cache[x] = res
        return cache[x]

    def reverse(n: int) -> int:
        lo = n & 0xFFFF
        hi = (n >> 16) & 0xFFFF
        return (_reverse_16(lo) << 16) | _reverse_16(hi)

    return reverse
```

---

### 套路 6：区间位与（公共前缀）

适用题：201

```python
def range_bitwise_and(left: int, right: int) -> int:
    """
    201: 计算 [left, right] 内所有整数的位与。
    同步右移直到 left == right，即找公共二进制前缀。时间 O(log n)，空间 O(1)。
    """
    shift = 0
    while left != right:
        left >>= 1
        right >>= 1
        shift += 1
    return right << shift


# Brian Kernighan 变体：不断消去 right 最低位直到 right <= left
def range_bitwise_and_bk(left: int, right: int) -> int:
    while right > left:
        right &= right - 1     # 消去 right 最低有效位
    return right
```

---

### 速查表

| 题型特征 | 套路 | 时间 | 空间 |
|---|---|---|---|
| 找出现奇数次的数（其余偶数次）| XOR 消消乐 | $O(n)$ | $O(1)$ |
| 找出现 1 次的数（其余 3 次）| 按位 mod 3 或 ones/twos | $O(n)$ | $O(1)$ |
| 单个数的 popcount | Brian Kernighan `n & (n-1)` | $O(k)$ | $O(1)$ |
| 0~n 所有数的 popcount | DP `bits[i>>1]+(i&1)` | $O(n)$ | $O(n)$ |
| 反转 32 位整数的位 | 逐位取低位拼接 | $O(32)$ | $O(1)$ |
| 区间 AND | 同步右移找公共前缀 | $O(\log n)$ | $O(1)$ |
| 二进制字符串相加 | 逐位模拟进位 | $O(\max(L_a,L_b))$ | $O(\max(L_a,L_b))$ |

---

## 方法变形（3 类）

### 变形 1：XOR 扩展

- **136**（单个奇数次）→ 基础 XOR。
- **137**（单个，其余 3 次）→ 按位 mod 3（可泛化为任意 k 次）。
- **260**（两个奇数次，非本 category）：XOR 得到两数之 XOR，找任一非零位将数组分组，分组后分别对每组 XOR 即得两个数。
- 泛化规律：若其余数出现 k 次，对每一位统计 1 的个数 mod k 即为目标数对应位。

### 变形 2：位运算技巧速查

- `x & (x-1)`：消去最低有效位（1）
- `x & (-x)` 或 `x & ~(x-1)`：提取最低有效位（Lowest Set Bit）
- `x | (x-1)`：将最低有效位及以下全置 1
- `x ^ (x-1)`：将最低有效位及以下全置 1（另一形式）
- `~x + 1` 或 `-x`（Python 直接用负号）：二进制补码取反
- `(x >> k) & 1`：取第 k 位
- `x | (1 << k)`：将第 k 位置 1
- `x & ~(1 << k)`：将第 k 位置 0

### 变形 3：338 DP 变体

- `bits[i] = bits[i & (i-1)] + 1`：另一种等价 DP（消去最低位后 +1），与 Brian Kernighan 对应。
- `bits[i] = bits[i // 2] + (i % 2)`：同 `bits[i>>1] + (i&1)`，仅写法不同。
- 集合枚举：对 0 到 $2^n - 1$ 的所有子集，338 的预处理可为子集枚举时的位计数提供 $O(1)$ 查表。

---

## 思考路标（条件反射）

1. 看到 **"出现偶数次 / 其余出现 2 次"** → XOR 消消乐，$O(n)$ $O(1)$
2. 看到 **"其余出现 3 次 / k 次"** → 按位 mod k，或 ones/twos 三进制计数器
3. 看到 **"统计单个数的 1 位个数"** → `n & (n-1)` 循环，或 `bin(n).count('1')`
4. 看到 **"统计 0~n 所有数的 1 位个数"** → DP，`bits[i] = bits[i>>1] + (i&1)`
5. 看到 **"反转 32 位整数"** → 32 轮循环，每轮取最低位追加到结果，末尾 `& 0xFFFFFFFF`
6. 看到 **"区间 AND / 区间公共前缀"** → 同步右移直到相等，或 Brian Kernighan 消低位
7. 看到 **"二进制字符串相加"** → 逐位模拟进位（类比十进制加法），或 Python int 转换
8. 看到 **"集合编码 / 子集枚举"** → 位运算：i 的所有非空子集 `s = i; while s: ... s = (s-1)&i`
9. 看到 **"INT8 / INT4 推理加速"** → 量化模型中，矩阵乘法用整数位运算代替浮点（AI 关联）

---

## 易错点

1. **136 XOR 顺序无关**：XOR 满足交换律和结合律，可以任意顺序异或，结果不变；不需要先排序。
2. **137 Python 负数处理**：Python 整数不溢出，但题目语境是 32 位有符号整数；按位统计后若结果 `>= 2^31` 需减去 `2^32` 转为负数表示。ones/twos 方法在 Python 中天然正确（无溢出问题）。
3. **190 Python 截断**：Python 无符号整数概念，`n >>= 1` 对负数（Python 表示为任意精度负整数）会有符号扩展；题目给的是 32 位无符号整数，逐位操作后最后 `& 0xFFFFFFFF` 截取 32 位。
4. **191 vs 338 的区别**：191 对单个数求 popcount；338 对 0~n 每个数求 popcount，用 DP 而非循环重复计算。
5. **201 右移到相等**：`while left != right` 而非 `left < right`；若 `left == right` 则区间内只有一个数，AND 就是它本身，不需要右移。
6. **67 Add Binary 前导零**：用 int 转换时 `bin(...)` 结果前缀为 `'0b'`，用 `[2:]` 截取；逐位模拟时从末尾向前加，最后 `carry` 若为 1 需补到最前。
7. **ones/twos 顺序**：137 中必须先更新 `ones` 再更新 `twos`，顺序反了会导致逻辑错误（两者互依赖）。

---

## 典型应用例题

### 例 1：136. Single Number

**题目**：数组中除某个数外，其余每个数都恰好出现两次，找出那个出现一次的数。要求 $O(n)$ 时间，$O(1)$ 空间。

**思路**：XOR 消消乐。`x ^ x = 0`（相同数对消），`x ^ 0 = x`（零不改变）。全部异或后，出现两次的数对消为 0，只剩出现一次的数。

**解**：

```python
# 参考：solutions/bit_manipulation/p136_single_number.py
def singleNumber(nums: List[int]) -> int:
    result = 0
    for x in nums:
        result ^= x
    return result
```

**分析**：一次遍历，时间 $O(n)$，空间 $O(1)$。若用哈希表则 $O(n)$ 空间；用排序则 $O(n \log n)$ 时间。XOR 是最优解。

---

### 例 2：191. Number of 1 Bits

**题目**：返回正整数 `n` 的 32 位二进制表示中 1 的个数（又称 Hamming Weight）。

**思路**：Brian Kernighan 算法：`n & (n-1)` 每次消去最低有效位（最右边的 1），循环次数等于 1 的个数，比逐位移位（固定 32 轮）更快。

**解**：

```python
# 参考：solutions/bit_manipulation/p191_number_of_1_bits.py
def hammingWeight(n: int) -> int:
    count = 0
    while n:
        n &= n - 1    # 消去最低有效位
        count += 1
    return count
```

**分析**：时间 $O(k)$，k 为 1 的个数（最多 32），在稀疏位（1 较少）时比逐位移位快。`n & (n-1)` 的原理：$n-1$ 将 n 的最低有效位（比如第 k 位）变为 0，其下方所有位变为 1；AND 后第 k 位以下全为 0，第 k 位以上不变。

---

### 例 3：201. Bitwise AND of Numbers Range

**题目**：给定区间 `[left, right]`，计算该区间内所有整数的位与结果。

**思路**：若 `left != right`，区间内相邻整数必然在某位上一个为 0 一个为 1，该位 AND 后为 0。同步右移直到 `left == right`，即找到两端共同的二进制前缀，再左移回去即为答案。

**解**：

```python
# 参考：solutions/bit_manipulation/p201_bitwise_and_of_numbers_range.py
def rangeBitwiseAnd(left: int, right: int) -> int:
    shift = 0
    while left != right:
        left >>= 1
        right >>= 1
        shift += 1
    return right << shift
```

**分析**：时间 $O(\log n)$（最多右移 32 次），空间 $O(1)$。当 `left = 0` 时区间包含 0，AND 结果必为 0；当 `left == right` 时结果即为 left 本身。

---

## 自测题

**自测 1**（136 Single Number）—— `nums=[2,2,1]` 返回 1；`nums=[4,1,2,1,2]` 返回 4。提示：`result = 0; for x in nums: result ^= x`，一行搞定。参考 `solutions/bit_manipulation/p136_single_number.py`。

**自测 2**（137 Single Number II）—— `nums=[2,2,3,2]` 返回 3；`nums=[0,1,0,1,0,1,99]` 返回 99。提示：按位统计，`count[bit] % 3` 即目标数该位；或 ones/twos 三进制计数器（先更新 ones 再更新 twos）。参考 `solutions/bit_manipulation/p137_single_number_ii.py`。

**自测 3**（190 Reverse Bits）—— `n=43261596`（二进制 `00000010100101000001111010011100`），反转后应为 `964176192`（`00111001011110000010100101000000`）。提示：32 轮循环，`result = (result << 1) | (n & 1); n >>= 1`，最后 `& 0xFFFFFFFF`。参考 `solutions/bit_manipulation/p190_reverse_bits.py`。

**自测 4**（191 Number of 1 Bits）—— `n=11`（二进制 `1011`）返回 3；`n=128`（二进制 `10000000`）返回 1。提示：`while n: n &= n-1; count += 1`（Brian Kernighan）或 `bin(n).count('1')`。参考 `solutions/bit_manipulation/p191_number_of_1_bits.py`。

**自测 5**（201 Bitwise AND Range）—— `left=5, right=7`（二进制 `101`, `110`, `111`），AND 结果为 `100 = 4`。`left=0, right=0` 返回 0。提示：同步右移直到 left==right，shift 次数记录，结果 `right << shift`。参考 `solutions/bit_manipulation/p201_bitwise_and_of_numbers_range.py`。

**自测 6**（67 Add Binary）—— `a='11', b='1'` 返回 `'100'`；`a='1010', b='1011'` 返回 `'10101'`。提示：从末尾逐位加，维护 carry，最后若 carry=1 则在最前加 '1'；或 `bin(int(a,2)+int(b,2))[2:]`。参考 `solutions/bit_manipulation/p067_add_binary.py`。

---

## 题目全览（6 题）

| # | 题目 | 套路分类 | 难度 |
|---|---|---|---|
| 136 | Single Number | XOR 消消乐 | Easy |
| 137 | Single Number II | 按位 mod 3 / ones-twos 计数 | Medium |
| 190 | Reverse Bits | 32 轮逐位反转 | Easy |
| 191 | Number of 1 Bits | Brian Kernighan `n & (n-1)` | Easy |
| 201 | Bitwise AND of Numbers Range | 同步右移公共前缀 | Medium |
| 67 | Add Binary | 逐位模拟进位 | Easy |

---

## 融合版说明

| 段 | 来源 | 价值 |
|---|---|---|
| 一例速记 | 本文件 | 6 题套路一览 + AI（量化推理 / 哈希压缩）关联 |
| 思维路径还原 | 本文件 | 6 道题的解题独白，含关键公式 |
| 抽象成方法 | 本文件 | 6 个标准模板（XOR / 三进制 / Brian Kernighan / DP 位计数 / 位反转 / 公共前缀）+ 速查表 |
| 方法变形 | 本文件 | 3 类变体（XOR 扩展 / 位运算技巧速查 / DP 变体） |
| 思考路标 | 本文件 | 9 条题型识别条件反射 |
| 易错点 | 本文件 | 7 条高频踩坑（Python 截断 / 负数 / 顺序依赖） |
| 典型应用例题 | solutions/ | 3 道精讲（136、191、201），代码 + 分析 |
| 自测题 | leetcode | 6 题带提示，链接 solutions 文件 |
| 题目全览 | 本文件 | 6 题完整列表 |

---

> **跨 category 导航**：
> - 哈希表 → `14-hash-table.md`（位运算常作为快速哈希的构件）
> - 数学 → `22-math.md`（进制转换、快速幂与位运算紧密相关）
> - 子集枚举 → `08-backtracking.md`（位掩码枚举所有子集）
> - CUDA Warp 级原子操作（`atomicOr`, `atomicAnd`）和 INT8 推理中的矩阵量化均大量使用本 category 的位运算技巧
