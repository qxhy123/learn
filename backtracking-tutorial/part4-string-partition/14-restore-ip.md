# 第14章：复原 IP 地址

## 14.1 问题描述

**复原 IP 地址**：给定只含数字的字符串，返回所有有效的 IPv4 地址。（LeetCode 93）

```
输入：s = "25525511135"
输出：["255.255.11.135","255.255.111.35"]

输入：s = "0000"
输出：["0.0.0.0"]

输入：s = "1111111111111111"
输出：[]（太长，无法构成有效 IP）
```

**有效段的规则**：
- 值在 0~255 之间
- 不能有前导零（"01" 无效，"0" 有效）

## 14.2 标准回溯解法

```python
def restoreIpAddresses(s):
    """
    将字符串分成 4 段，每段满足 IP 地址规则
    时间 O(3^4) = O(81)，实际更小（有大量剪枝）
    """
    result = []
    n = len(s)
    
    def is_valid(segment):
        # 长度检查
        if len(segment) > 3 or len(segment) == 0:
            return False
        # 前导零检查
        if len(segment) > 1 and segment[0] == '0':
            return False
        # 数值范围检查
        return int(segment) <= 255
    
    def backtrack(start, parts):
        if len(parts) == 4:
            if start == n:  # 恰好用完所有字符
                result.append('.'.join(parts))
            return
        
        # 剪枝：剩余字符数不合法
        remaining = n - start
        needed = 4 - len(parts)
        if remaining < needed or remaining > needed * 3:
            return
        
        for length in range(1, 4):  # 每段 1~3 个字符
            if start + length > n:
                break
            segment = s[start:start + length]
            if is_valid(segment):
                parts.append(segment)
                backtrack(start + length, parts)
                parts.pop()
    
    backtrack(0, [])
    return result

# 测试
print(restoreIpAddresses("25525511135"))
# ['255.255.11.135', '255.255.111.35']
print(restoreIpAddresses("0000"))
# ['0.0.0.0']
print(restoreIpAddresses("101023"))
# ['1.0.10.23', '1.0.102.3', '10.1.0.23', '10.10.2.3', '101.0.2.3']
```

## 14.3 剪枝分析

```
字符串 s 长度为 n，IP 需要 4 段，每段 1~3 字符。
有效约束：
  4 <= n <= 12（否则无解）
  每步选 length 时：remaining_chars / remaining_parts 给出上下界

剪枝条件：
  remaining < needed       → 字符不够，提前返回
  remaining > needed * 3   → 字符太多，提前返回
  前导零               → 长度>1 且首字符为 '0' 直接跳过
  超范围               → int(segment) > 255 直接跳过
```

## 14.4 IPv6 扩展

```python
def restoreIpv6Addresses(s):
    """
    复原 IPv6 地址：8 组，每组 1~4 个十六进制字符
    """
    result = []
    n = len(s)
    
    def is_valid_hex(segment):
        if len(segment) == 0 or len(segment) > 4:
            return False
        # IPv6 允许前导零（如 "0001" 有效）
        return all(c in '0123456789abcdefABCDEF' for c in segment)
    
    def backtrack(start, parts):
        if len(parts) == 8:
            if start == n:
                result.append(':'.join(parts))
            return
        
        remaining = n - start
        needed = 8 - len(parts)
        if remaining < needed or remaining > needed * 4:
            return
        
        for length in range(1, 5):
            if start + length > n:
                break
            segment = s[start:start + length]
            if is_valid_hex(segment):
                parts.append(segment)
                backtrack(start + length, parts)
                parts.pop()
    
    backtrack(0, [])
    return result
```

## 14.5 通用字符串分段框架

IP 地址问题是字符串分段的典型，可以抽象为通用框架：

```python
def split_string(s, num_parts, is_valid_part, separator='.'):
    """
    通用字符串分段回溯
    
    参数：
        s           - 待分割字符串
        num_parts   - 需要分成几段
        is_valid_part(segment) - 判断一段是否有效
        separator   - 分隔符
    """
    result = []
    n = len(s)
    
    def backtrack(start, parts):
        if len(parts) == num_parts:
            if start == n:
                result.append(separator.join(parts))
            return
        
        remaining = n - start
        needed = num_parts - len(parts)
        
        for length in range(1, n - start + 1):
            segment = s[start:start + length]
            if not is_valid_part(segment, remaining, needed):
                continue
            parts.append(segment)
            backtrack(start + length, parts)
            parts.pop()
    
    backtrack(0, [])
    return result

# 用通用框架实现 IP 地址
def is_valid_ip_part(segment, remaining, needed):
    if len(segment) > 3:
        return False
    if remaining < needed or remaining > needed * 3:
        return False
    if len(segment) > 1 and segment[0] == '0':
        return False
    return int(segment) <= 255

result = split_string("25525511135", 4, is_valid_ip_part)
print(result)  # ['255.255.11.135', '255.255.111.35']
```

## 14.6 迭代解法（四重循环）

由于 IP 恰好 4 段，可以用四重循环替代递归（更直观）：

```python
def restoreIpAddresses_iterative(s):
    """
    四重循环枚举 3 个分割点
    等价于回溯但更直观
    """
    n = len(s)
    result = []
    
    def valid(seg):
        return len(seg) <= 3 and (len(seg) == 1 or seg[0] != '0') and int(seg) <= 255
    
    for i in range(1, min(4, n)):
        for j in range(i+1, min(i+4, n)):
            for k in range(j+1, min(j+4, n)):
                a, b, c, d = s[:i], s[i:j], s[j:k], s[k:]
                if all(valid(x) for x in [a, b, c, d]):
                    result.append(f"{a}.{b}.{c}.{d}")
    
    return result
```

## 小结

| 方法 | 优点 | 适用场景 |
|------|------|---------|
| 回溯 + 剪枝 | 通用，易扩展 | 任意段数 |
| 四重循环 | 代码简单 | 固定 4 段 |
| 通用框架 | 高复用性 | 类似分段问题 |

**关键剪枝总结**：
1. 字符总数越界剪枝：`n < needed` 或 `n > needed * 3`
2. 前导零剪枝
3. 数值范围剪枝

## 练习

1. 解决"IP 地址无效化"（LeetCode 1108）：将 IP 中的 '.' 替换为 '[.]'
2. 统计字符串能复原出多少个有效 IP 地址（不返回具体列表）
3. 扩展为 CIDR 表示法（IP/前缀长度），验证合法性

---

**上一章：** [回文分割](13-palindrome-partition.md) | **下一章：** [电话号码字母组合](15-letter-combinations.md)
