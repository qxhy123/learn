# 191. Number of 1 Bits

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/number-of-1-bits/
- Official Group: Bit Manipulation
- Pattern Group: Bit Manipulation
- Patterns: bit-manipulation

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given an unsigned integer `n`, return how many `1` bits appear in its binary representation.

This count is often called the Hamming weight.

For example, the decimal number `11` is:

```text
11 = 8 + 2 + 1
   = 1011₂
```

Its binary representation has three `1` bits:

```text
1011
^ ^^  three 1s
```

So the answer is:

```text
3
```

The problem is not asking for the value of the number in decimal form. It is asking for a structural property of the number's binary form:

> How many bit positions are turned on?

A bit position is turned on if that bit equals `1`. It is turned off if that bit equals `0`.

---

### 2. Start From the Most Direct Idea

If we wrote the number out in binary, we could simply count the characters equal to `'1'`.

Conceptually:

```python
binary = bin(n)
answer = binary.count("1")
```

This is easy to understand, but it avoids the main point of the problem: using bit operations directly.

A lower-level baseline is to inspect each bit from right to left.

The rightmost bit tells us whether the number is odd:

```text
n & 1 == 1  means the last bit is 1
n & 1 == 0  means the last bit is 0
```

After checking the last bit, shift the number right by one position:

```text
n >>= 1
```

That discards the bit we just counted and makes the next bit become the new last bit.

Baseline algorithm:

```python
count = 0

while n != 0:
    if n & 1:
        count += 1
    n >>= 1

return count
```

This is correct. For a fixed 32-bit unsigned integer it checks at most `32` bits, so it is already constant time in LeetCode's model.

But there is a sharper observation that lets us skip all zero bits.

---

### 3. Key Observation: `n & (n - 1)` Removes One `1` Bit

The most important identity for this problem is:

```text
n & (n - 1)
```

This expression clears the lowest set bit of `n`.

"Lowest set bit" means the rightmost bit whose value is `1`.

Why does this work?

Look at a number ending with some zeros after its rightmost `1`:

```text
n     = ... 1 0 0 0
```

Subtracting `1` changes that rightmost `1` to `0`, and changes all trailing zeros after it to `1`:

```text
n - 1 = ... 0 1 1 1
```

Now apply bitwise AND:

```text
n       = ... 1 0 0 0
n - 1   = ... 0 1 1 1
-----------------------
n&(n-1) = ... 0 0 0 0
```

Everything to the left of that bit stays the same. The rightmost `1` disappears. The lower bits become zero because `n` had zeros there.

So one operation turns:

```text
some number with k one-bits
```

into:

```text
some smaller number with k - 1 one-bits
```

That is exactly the operation this problem wants.

---

### 4. The Bit-Count Invariant

Maintain two pieces of state:

```text
n      = the remaining bits not yet removed
count  = how many 1-bits have already been removed
```

The invariant is:

```text
original_number_of_1_bits = count + number_of_1_bits_still_in_n
```

At the beginning:

```text
count = 0
n = original input
```

So the invariant is true.

Each loop iteration does this:

```text
n = n & (n - 1)
count += 1
```

The expression `n & (n - 1)` removes exactly one `1` bit from `n`. Since one remaining `1` bit was removed, increasing `count` by one keeps the invariant balanced.

Eventually, `n` becomes `0`.

At that moment:

```text
number_of_1_bits_still_in_n = 0
```

So the invariant becomes:

```text
original_number_of_1_bits = count
```

That is the answer.

---

### 5. Detailed Algorithm

1. Initialize:

```text
count = 0
```

2. While `n` is not zero:

```text
while n != 0:
```

3. Remove the lowest set bit:

```text
n = n & (n - 1)
```

4. Record that one `1` bit was removed:

```text
count += 1
```

5. When no set bits remain, return `count`.

The loop does not run once per binary position. It runs once per `1` bit.

So for a number like:

```text
10000000000000000000000000000000
```

there is only one `1` bit, and the loop runs once.

For a number like:

```text
11111111111111111111111111111111
```

there are thirty-two `1` bits, and the loop runs thirty-two times.

---

### 6. Example Walkthrough: `n = 11`

Start with:

```text
n = 11
binary n = 1011
count = 0
```

#### Iteration 1

The lowest set bit is the rightmost `1`:

```text
n       = 1011
n - 1   = 1010
----------------
n&(n-1) = 1010
```

Update:

```text
n = 1010₂ = 10
count = 1
```

We have removed one `1` bit.

#### Iteration 2

Now:

```text
n       = 1010
n - 1   = 1001
----------------
n&(n-1) = 1000
```

Update:

```text
n = 1000₂ = 8
count = 2
```

We have removed a second `1` bit.

#### Iteration 3

Now:

```text
n       = 1000
n - 1   = 0111
----------------
n&(n-1) = 0000
```

Update:

```text
n = 0
count = 3
```

The loop stops because no `1` bits remain.

Final answer:

```text
3
```

---

### 7. Example Walkthrough: `n = 128`

The number `128` is a power of two:

```text
128 = 10000000₂
```

It has exactly one `1` bit.

Run one iteration:

```text
n       = 10000000
n - 1   = 01111111
------------------
n&(n-1) = 00000000
```

Update:

```text
count = 1
n = 0
```

The loop stops immediately.

Final answer:

```text
1
```

This example shows why this method is better than blindly scanning every bit: all trailing zeros are skipped in one operation.

---

### 8. Code

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0

        while n != 0:
            n &= n - 1
            count += 1

        return count
```

Equivalent pseudocode:

```text
count = 0

while n is not zero:
    remove the lowest 1-bit from n
    count one removed bit

return count
```

---

### 9. Why This Code Is Correct

We prove that the algorithm returns the number of `1` bits in the original input.

The algorithm maintains this invariant:

```text
count + number_of_1_bits(n) = number_of_1_bits(original input)
```

At initialization, `count` is `0` and `n` is the original input, so the invariant is true.

During each loop iteration, `n & (n - 1)` removes exactly one `1` bit from `n` and does not create any new `1` bits. Therefore `number_of_1_bits(n)` decreases by exactly one. The algorithm also increases `count` by exactly one. The sum of these two quantities stays unchanged, so the invariant remains true.

The loop terminates when `n == 0`. A zero value has no `1` bits, so `number_of_1_bits(n) = 0`. Substituting this into the invariant gives:

```text
count = number_of_1_bits(original input)
```

Thus the returned value is exactly the Hamming weight of the input.

---

### 10. Complexity

Let `k` be the number of `1` bits in `n`.

The loop removes exactly one `1` bit per iteration, so it runs exactly `k` times.

For LeetCode's unsigned 32-bit input model:

```text
0 <= k <= 32
```

So the complexity is:

```text
Time:  O(k), which is O(1) for a fixed 32-bit integer
Space: O(1)
```

If we describe the input width as `w` bits, the worst-case time is `O(w)`.

---

### 11. Common Pitfalls

#### Pitfall 1: Confusing decimal digits with binary bits

The number `11` in decimal is not two ones. Its binary representation is:

```text
1011
```

So the answer is `3`, not `2`.

#### Pitfall 2: Forgetting what `n & 1` checks

The expression:

```python
n & 1
```

only checks the current lowest bit. It does not count all bits by itself. If using the shift-based baseline, you must shift after checking:

```python
count += n & 1
n >>= 1
```

#### Pitfall 3: Using `n & (n + 1)` instead of `n & (n - 1)`

The clearing identity depends on subtracting one:

```text
n & (n - 1)
```

Using `n + 1` has a different meaning and does not reliably remove a set bit.

#### Pitfall 4: Incrementing the count in the wrong place

For Brian Kernighan's algorithm, every loop iteration removes exactly one `1` bit. Therefore the count should increase exactly once per iteration.

Do not add the numeric value of a bitmask such as `n & (n - 1)` to the count.

#### Pitfall 5: Ignoring integer width in languages with signed integers

The LeetCode problem defines the input as unsigned. In languages with fixed-width signed integers, be careful with right shifts if you use the baseline shift method.

The `n & (n - 1)` method avoids most shift-sign confusion because it only uses subtraction and AND on the integer value provided by the platform/problem interface.

In Python, integers are unbounded, but LeetCode passes a non-negative value for this problem, so the loop terminates normally.

---

### 12. First-Principles Summary

This problem follows from these basic facts:

```text
1. An integer can be viewed as a sequence of binary bits.
2. The answer is the number of positions whose bit is 1.
3. Subtracting 1 flips the lowest 1-bit to 0 and flips lower trailing zeros to 1.
4. AND-ing n with n - 1 keeps the higher bits but clears that lowest 1-bit.
5. Therefore each application of n & (n - 1) removes exactly one counted bit.
6. Counting how many removals are needed to reach 0 gives the number of 1 bits.
```

In one sentence:

> Repeatedly clear the lowest set bit with `n &= n - 1`, and count how many times this can be done before the number becomes zero.

## Implementation

See `solutions/bit_manipulation/p191_number_of_1_bits.py`.

## Tests

See `tests/bit_manipulation/test_p191_number_of_1_bits.py`.

## Examples

### Example 1
- Input: `{'raw': '11\n128\n2147483645'}`
- Output: `'See official examples'`

### Official Example: `n = 11`
- Input: `n = 11`
- Binary: `1011`
- Output: `3`

### Official Example: `n = 128`
- Input: `n = 128`
- Binary: `10000000`
- Output: `1`

### Official Example: `n = 2147483645`
- Input: `n = 2147483645`
- Binary: `1111111111111111111111111111101`
- Output: `30`

## Follow-up Practice
- Trace `n & (n - 1)` on `n = 7`, `n = 8`, and `n = 15`.
- Compare the shift-based baseline with the lowest-set-bit removal method.
- Explain why a power of two always returns `1`.
