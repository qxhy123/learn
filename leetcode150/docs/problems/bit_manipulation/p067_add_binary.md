# 67. Add Binary

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/add-binary/
- Official Group: Bit Manipulation
- Pattern Group: Bit Manipulation
- Patterns: bit-manipulation

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given two strings, `a` and `b`.

Each string contains only the characters `'0'` and `'1'`, and each string represents a non-negative integer written in base 2.

The task is to return a new string representing their sum, also in base 2.

For example:

```text
a = "11"     # binary for 3
b = "1"      # binary for 1
sum = "100"  # binary for 4
```

The important detail is that the inputs are strings, not integer arrays. A good solution should add the numbers digit by digit, the same way we add decimal numbers by hand, instead of relying on language conversion helpers.

So the real problem is:

> Simulate addition in base 2 from right to left, carrying any overflow into the next bit position.

### 2. Start From the Baseline Idea

The most direct baseline is to convert both binary strings into ordinary integers, add them, and convert the result back to binary:

```python
value = int(a, 2) + int(b, 2)
return bin(value)[2:]
```

That is short, and it often works in Python.

But it avoids the actual algorithmic idea. It also depends on built-in parsing and formatting, and in languages with fixed-width integer types it can overflow when the strings are very long.

A slightly more manual baseline is:

1. Convert `a` from binary string to integer.
2. Convert `b` from binary string to integer.
3. Add the integers.
4. Repeatedly divide by 2 to build the answer bits.

This still does extra work: it first reconstructs the whole numeric values, even though binary addition only needs local information from one bit position at a time.

The first-principles question is:

> Can we produce each output bit using only the current bits and a carry?

Yes. That is exactly how positional addition works.

### 3. Binary Addition From First Principles

In base 10, each column can produce a digit from `0` to `9`, and anything `10` or larger creates a carry into the next column.

In base 2, each column can produce only a digit from `0` to `1`, and anything `2` or larger creates a carry into the next column.

For one bit position, there are three possible contributors:

```text
bit from a
bit from b
carry from the previous lower position
```

Their total can be:

```text
0, 1, 2, or 3
```

The output bit is the remainder after dividing by 2:

```text
output_bit = total % 2
```

The carry into the next position is the quotient after dividing by 2:

```text
next_carry = total // 2
```

That gives the entire local rule:

```text
total = current_bit_of_a + current_bit_of_b + carry
append total % 2 to the answer
carry = total // 2
```

Because binary numbers are positional, this local rule is enough for the whole string.

### 4. The Key Observation

The least significant bit is at the right end of each string.

For example:

```text
"1011"
   ^ rightmost bit is the 1s place
```

Addition must start there, because the carry flows from lower-value positions to higher-value positions:

```text
1s place -> 2s place -> 4s place -> 8s place -> ...
```

So we use two indices:

```text
i = len(a) - 1
j = len(b) - 1
```

At each step, we read `a[i]` if `i` is still in range, otherwise that side contributes `0`. We do the same for `b[j]`.

This naturally handles different lengths:

```text
  1010
+   11
```

The missing left-side positions of the shorter string are simply zeros:

```text
  1010
+ 0011
```

### 5. The Bit/Carry Invariant

The invariant is the heart of the solution.

After processing some suffix of `a` and some suffix of `b`, we maintain:

```text
answer holds the correct sum bits for all processed lower positions,
but in reverse order,
and carry is exactly the carry that must be added to the next unprocessed position.
```

Why reverse order?

Because we discover the least significant output bit first, then the next bit, and so on. The final binary string must be most-significant bit first, so we reverse the collected bits at the end.

For example, if the true answer is:

```text
10101
```

we will produce bits in this order:

```text
1, 0, 1, 0, 1
```

For palindromic examples that happens to look the same, but in general the produced list is the reverse of the final string.

The carry invariant is more important than the storage order:

```text
carry always contains the overflow from all lower positions already processed.
```

That means when we reach a new column, the only information from the entire lower suffix that still matters is `carry`. We never need to remember the whole lower suffix again.

### 6. Detailed Algorithm

Use three pieces of state:

```text
i      index into a, starting at the last character
j      index into b, starting at the last character
carry  overflow from the lower bit position
bits   output bits collected from right to left
```

Then repeat while there is still work left:

```text
while i >= 0 or j >= 0 or carry != 0:
```

The loop continues if any of these are true:

- `a` still has an unprocessed bit.
- `b` still has an unprocessed bit.
- There is a final carry that must become a new leading bit.

Inside the loop:

1. Start `total` with `carry`.
2. If `i >= 0`, add the integer value of `a[i]` and move `i` left.
3. If `j >= 0`, add the integer value of `b[j]` and move `j` left.
4. Append `total % 2` as the next output bit.
5. Set `carry = total // 2`.

At the end, reverse `bits` and join them into a string.

### 7. Example Walkthrough: `a = "11"`, `b = "1"`

Write the addition vertically:

```text
  11
+  1
----
```

Initialize:

```text
i = 1       # points to last '1' in a
j = 0       # points to last '1' in b
carry = 0
bits = []
```

#### Step 1: Add the rightmost bits

```text
a[i] = 1
b[j] = 1
carry = 0

total = 1 + 1 + 0 = 2
output bit = 2 % 2 = 0
new carry = 2 // 2 = 1
```

Now:

```text
bits = ["0"]
i = 0
j = -1
carry = 1
```

The rightmost result bit is correct: `1 + 1` in binary is `10`, so we write `0` and carry `1`.

#### Step 2: Add the next column

`b` has no bit left, so it contributes `0`.

```text
a[i] = 1
b contributes 0
carry = 1

total = 1 + 0 + 1 = 2
output bit = 0
new carry = 1
```

Now:

```text
bits = ["0", "0"]
i = -1
j = -1
carry = 1
```

#### Step 3: Flush the final carry

Both strings are exhausted, but `carry` is still `1`, so the loop must run one more time.

```text
a contributes 0
b contributes 0
carry = 1

total = 1
output bit = 1
new carry = 0
```

Now:

```text
bits = ["0", "0", "1"]
```

Reverse the collected bits:

```text
"100"
```

So:

```text
"11" + "1" = "100"
```

### 8. Example Walkthrough: `a = "1010"`, `b = "1011"`

Align the strings:

```text
  1010
+ 1011
------
```

Process from right to left:

| Position from right | a bit | b bit | incoming carry | total | output bit | outgoing carry |
| --- | --- | --- | --- | --- | --- | --- |
| 1s | 0 | 1 | 0 | 1 | 1 | 0 |
| 2s | 1 | 1 | 0 | 2 | 0 | 1 |
| 4s | 0 | 0 | 1 | 1 | 1 | 0 |
| 8s | 1 | 1 | 0 | 2 | 0 | 1 |
| 16s | 0 | 0 | 1 | 1 | 1 | 0 |

The bits are produced from low to high:

```text
1, 0, 1, 0, 1
```

Reversing gives:

```text
10101
```

So:

```text
"1010" + "1011" = "10101"
```

### 9. Code / Pseudocode

A direct Python implementation looks like this:

```python
def addBinary(a: str, b: str) -> str:
    i = len(a) - 1
    j = len(b) - 1
    carry = 0
    bits = []

    while i >= 0 or j >= 0 or carry:
        total = carry

        if i >= 0:
            total += ord(a[i]) - ord("0")
            i -= 1

        if j >= 0:
            total += ord(b[j]) - ord("0")
            j -= 1

        bits.append(str(total % 2))
        carry = total // 2

    return "".join(reversed(bits))
```

Using `int(a[i])` instead of `ord(a[i]) - ord("0")` is also fine:

```python
total += int(a[i])
```

The algorithmic idea is the same either way.

### 10. Correctness

We prove that the algorithm returns the binary representation of the sum of `a` and `b`.

#### Lemma 1: Each loop iteration writes the correct bit for the current position.

At the start of an iteration, `carry` is the overflow from all lower positions. The algorithm adds `carry`, the current bit from `a` if present, and the current bit from `b` if present.

This `total` is exactly the value that belongs to the current binary column before splitting it into an output bit and a carry.

In base 2, the bit written in the current column must be `total % 2`, and the carry into the next column must be `total // 2`.

Therefore, each iteration writes the correct bit for its position and computes the correct carry for the next position.

#### Lemma 2: The loop invariant is preserved.

Assume before an iteration that `bits` contains the correct lower result bits in reverse order, and `carry` is the correct carry into the next unprocessed position.

By Lemma 1, the algorithm appends the correct bit for that next position and updates `carry` to the correct carry for the following position.

So after the iteration, `bits` contains one more correct lower result bit, still in reverse order, and `carry` is still exactly the carry needed for the next unprocessed position.

Thus the invariant is preserved.

#### Lemma 3: When the loop stops, all necessary result bits have been produced.

The loop stops only when:

```text
i < 0
j < 0
carry == 0
```

That means both input strings have no unprocessed bits left, and there is no remaining carry to become a new leading bit.

By the invariant, all lower positions already stored in `bits` are correct. Since no positions or carry remain, those bits form the entire sum.

#### Theorem: The algorithm returns the correct answer.

By Lemma 2, the invariant holds through every iteration. By Lemma 3, when the loop terminates, `bits` contains exactly all bits of the correct sum, but in reverse order. The algorithm reverses `bits` before joining them, so it returns the correct binary string.

### 11. Complexity

Let:

```text
n = len(a)
m = len(b)
```

The loop processes one bit from `a`, one bit from `b`, or a final carry on each iteration.

So the number of iterations is at most:

```text
max(n, m) + 1
```

Therefore:

- Time: `O(max(n, m))`
- Space: `O(max(n, m))` for the output list

If output space is not counted as auxiliary space, the extra working state is `O(1)`.

### 12. Common Pitfalls

- Stopping when both indices are exhausted but forgetting a final `carry`.
  - Example: `"1" + "1"` must produce `"10"`, not `"0"`.
- Building the answer left-to-right while processing right-to-left.
  - If you append bits as you compute them, remember to reverse at the end.
- Treating characters as numbers without converting them.
  - `'1' + '1'` is string concatenation in many languages, not numeric addition.
- Mishandling different lengths.
  - Once one string runs out, its missing bits should behave like zeros.
- Using decimal intuition for the carry.
  - In binary, `total // 2` is the carry and `total % 2` is the output bit.
- Removing leading zeros incorrectly.
  - LeetCode inputs for this problem are normal binary strings, and the addition algorithm naturally returns the correct representation. Do not strip the only zero from result `"0"`.

### 13. First-Principles Summary

Binary addition is not a special trick. It is ordinary positional addition with base `2` instead of base `10`.

At each column, only three values matter:

```text
current bit from a
current bit from b
carry from lower columns
```

Their sum determines everything:

```text
result bit = total % 2
next carry = total // 2
```

Because carry flows from right to left, we scan both strings from the end. Because output bits are discovered from least significant to most significant, we collect them and reverse them at the end.

The whole solution is just the repeated preservation of one invariant:

```text
processed lower bits are already correct,
and carry is exactly what must be added to the next higher bit.
```

## Implementation
See `solutions/bit_manipulation/p067_add_binary.py`.

## Tests
See `tests/bit_manipulation/test_p067_add_binary.py`.

## Examples

### Example 1
- Input: `{'a': '11', 'b': '1'}`
- Output: `'100'`

### Example 2
- Input: `{'a': '1010', 'b': '1011'}`
- Output: `'10101'`

## Follow-up Practice

- Trace `"1" + "1"` and identify why the final carry matters.
- Trace `"111" + "1"` and watch the carry propagate through several columns.
- Rewrite the loop using a single `total` variable, then again using explicit Boolean cases, and compare which version is easier to reason about.
- Explain why missing bits from the shorter string are equivalent to zeros.
