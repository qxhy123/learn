# 第13章：数位DP

## 13.1 什么是数位DP

**数位DP** 处理的问题形如：

> 统计 `[1, n]` 中满足某种条件的整数个数。

这类条件通常与数字的**各个位**有关，如：
- 各位之和等于某个值
- 不包含某个数字
- 任意两位之差不小于某个值
- ...

**朴素做法**：遍历 1 到 n，逐个检查。但 n 可能高达 10^18，O(n) 完全不可行。

**数位DP做法**：按位构造数字，用 DP 统计合法方案数。

---

## 13.2 核心思路：按位构造

将数字 n 按位拆解（如 n=326 → [3, 2, 6]），从高位到低位逐位确定每一位的值。

**关键状态**：
- `pos`：当前填到第几位（从高位开始）
- `tight`：当前前缀是否已经"贴紧"上界（即前面的位都与 n 对应位相同）
  - 若 `tight=True`，当前位最多填 `digits[pos]`
  - 若 `tight=False`，当前位可以填 0~9
- `leading_zero`：是否还有前导零（影响某些问题，如计算数字之和时前导0不应计入位数）
- **其他问题相关状态**：如当前各位之和、是否满足特定条件等

---

## 13.3 模板代码

```python
from functools import lru_cache

def count_numbers(n, condition):
    """
    统计 [0, n] 中满足 condition 的数字数量。
    condition 需要在 dp 函数中实现。
    """
    digits = [int(d) for d in str(n)]
    
    @lru_cache(maxsize=None)
    def dp(pos, tight, *extra_state):
        """
        pos:       当前处理到第几位（0-indexed）
        tight:     是否受上界约束
        extra_state: 问题相关的额外状态
        """
        if pos == len(digits):
            # 所有位已填完，判断是否满足条件
            return 1 if is_valid(*extra_state) else 0
        
        limit = digits[pos] if tight else 9
        result = 0
        
        for d in range(0, limit + 1):
            new_tight = tight and (d == limit)
            new_extra = update_state(*extra_state, d)
            result += dp(pos + 1, new_tight, *new_extra)
        
        return result
    
    return dp(0, True, *initial_state)
```

---

## 13.4 实战：数字1的个数（LeetCode 233）

**题目**：统计 `[1, n]` 中数字 1 出现的总次数。

```python
from functools import lru_cache

def count_digit_one(n):
    digits = [int(d) for d in str(n)]
    
    @lru_cache(maxsize=None)
    def dp(pos, tight, count):
        # count: 已经出现了多少个 1
        if pos == len(digits):
            return count  # 返回当前路径中1的个数（贡献给总和）
        
        limit = digits[pos] if tight else 9
        result = 0
        
        for d in range(0, limit + 1):
            result += dp(pos + 1, tight and (d == limit), count + (d == 1))
        
        return result
    
    return dp(0, True, 0)

print(count_digit_one(13))   # 6
print(count_digit_one(100))  # 21
```

---

## 13.5 实战：不含连续1的非负整数（LeetCode 600）

**题目**：统计 `[0, n]` 中二进制表示不含连续 1 的整数数量。

```python
def find_integers(n):
    bits = [int(b) for b in bin(n)[2:]]  # 二进制位
    
    @lru_cache(maxsize=None)
    def dp(pos, tight, prev_bit):
        # prev_bit: 上一位是否为1（用于判断连续1）
        if pos == len(bits):
            return 1
        
        limit = bits[pos] if tight else 1  # 二进制，每位只能是0或1
        result = 0
        
        for d in range(0, limit + 1):
            if prev_bit == 1 and d == 1:
                continue  # 连续1，跳过
            result += dp(pos + 1, tight and (d == limit), d)
        
        return result
    
    return dp(0, True, 0)

print(find_integers(5))   # 5 (0,1,2,3,5 不含连续1，4=100可以)
print(find_integers(1))   # 2
```

---

## 13.6 实战：各位数字之和可被整除（LeetCode 1015）

**题目**：统计 `[1, n]` 中满足"各位数字之和能被 k 整除"的数字数量。

```python
def divisor_game(n, k):
    digits = [int(d) for d in str(n)]
    
    @lru_cache(maxsize=None)
    def dp(pos, tight, digit_sum_mod, is_positive):
        """
        digit_sum_mod: 当前各位数字之和 mod k
        is_positive: 是否至少有一位非零（排除0本身）
        """
        if pos == len(digits):
            return 1 if (is_positive and digit_sum_mod == 0) else 0
        
        limit = digits[pos] if tight else 9
        result = 0
        
        for d in range(0, limit + 1):
            result += dp(
                pos + 1,
                tight and (d == limit),
                (digit_sum_mod + d) % k,
                is_positive or (d > 0)
            )
        
        return result
    
    return dp(0, True, 0, False)
```

---

## 13.7 实战：数字范围按位与（LeetCode 201）

**题目**：`[m, n]` 范围内所有整数的按位与结果。

这道题不用数位DP，但展示了"按位思维"：

```python
def range_bitwise_and(m, n):
    # 不断右移直到 m == n，找公共前缀
    shift = 0
    while m != n:
        m >>= 1
        n >>= 1
        shift += 1
    return m << shift
```

---

## 13.8 实战：至少有1位重复的数字（LeetCode 1012）

**题目**：统计 `[1, n]` 中至少有一个重复数字的整数数量。

**技巧：正难则反**

```
答案 = n - 没有重复数字的整数数量
```

统计没有重复数字的整数更容易：

```python
def num_dup_digits_at_most_n(n):
    digits = [int(d) for d in str(n)]
    
    @lru_cache(maxsize=None)
    def dp(pos, tight, used_mask, started):
        """
        used_mask: 已用数字的集合（10个数字，用10位mask）
        started: 是否已经开始（非前导零）
        """
        if pos == len(digits):
            return 1 if started else 0  # 不计0本身
        
        limit = digits[pos] if tight else 9
        result = 0
        
        for d in range(0, limit + 1):
            if started and (used_mask & (1 << d)):
                continue  # 数字已使用
            
            new_started = started or (d > 0)
            new_mask = (used_mask | (1 << d)) if new_started else 0
            
            result += dp(pos + 1, tight and (d == limit), new_mask, new_started)
        
        return result
    
    no_dup = dp(0, True, 0, False)
    return n - no_dup

print(num_dup_digits_at_most_n(20))   # 1 (11)
print(num_dup_digits_at_most_n(100))  # 10
```

---

## 13.9 数位DP的通用框架总结

```python
from functools import lru_cache

def digit_dp_template(n):
    digits = list(map(int, str(n)))
    
    @lru_cache(maxsize=None)
    def dp(pos, tight, leading_zero, *state):
        # pos:          当前位（0-indexed）
        # tight:        是否受上界限制
        # leading_zero: 是否还是前导零（影响数字长度计算）
        # *state:       问题相关的额外状态
        
        if pos == len(digits):
            return base_case(*state, leading_zero)
        
        up = digits[pos] if tight else 9
        res = 0
        
        for d in range(0, up + 1):
            if leading_zero and d == 0:
                # 前导零继续
                res += dp(pos+1, tight and d==up, True, *initial_state)
            else:
                new_state = transition(*state, d, leading_zero and d==0)
                if is_prunable(new_state):
                    continue
                res += dp(pos+1, tight and d==up, False, *new_state)
        
        return res
    
    return dp(0, True, True, *initial_state)
```

**设计数位DP状态的清单**：

| 状态 | 是否必须 | 说明 |
|------|---------|------|
| `pos` | ✅ 必须 | 当前处理的位 |
| `tight` | ✅ 必须 | 是否受上界约束 |
| `leading_zero` | 视情况 | 需要区分前导零时使用 |
| 问题相关状态 | ✅ 必须 | 如数字和、使用过的数字等 |

---

## LeetCode 推荐题目

- [233. 数字 1 的个数](https://leetcode.cn/problems/number-of-digit-one/) ⭐⭐⭐
- [600. 不含连续1的非负整数](https://leetcode.cn/problems/non-negative-integers-without-consecutive-ones/) ⭐⭐⭐
- [1012. 至少有 1 位重复的数字](https://leetcode.cn/problems/numbers-with-repeated-digits/) ⭐⭐⭐
- [357. 统计各位数字都不同的数字个数](https://leetcode.cn/problems/count-numbers-with-unique-digits/) ⭐⭐
- [2719. 统计整数数目](https://leetcode.cn/problems/count-of-integers/) ⭐⭐⭐
