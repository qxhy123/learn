# 136. Single Number

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/single-number/
- Official Group: Bit Manipulation
- Pattern Group: Bit Manipulation
- Patterns: bit-manipulation, xor, parity

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an integer array `nums` with a very special guarantee:

```text
Every value appears exactly twice, except for one value that appears exactly once.
```

Return the value that appears once.

For example:

```text
nums = [2, 2, 1]
```

The value `2` appears twice.
The value `1` appears once.

So the answer is:

```text
1
```

Another example:

```text
nums = [4, 1, 2, 1, 2]
```

The duplicate pairs are:

```text
1 and 1
2 and 2
```

The value left over is:

```text
4
```

So the answer is:

```text
4
```

The real problem is not just to find a frequency of `1`. The follow-up constraint usually asks for:

```text
linear time
constant extra space
```

That means a hash map is a useful baseline, but not the final idea.

---

### 2. Start From the Brute Force Idea

The most direct approach is to test each number and count how often it appears.

Conceptually:

```python
for x in nums:
    count = 0
    for y in nums:
        if x == y:
            count += 1

    if count == 1:
        return x
```

This is correct because the problem promises exactly one single value.

But it is inefficient:

```text
For each of n values, scan up to n values again.
Time: O(n^2)
Space: O(1)
```

We can improve time by storing counts.

```python
counts = {}

for x in nums:
    counts[x] = counts.get(x, 0) + 1

for x, count in counts.items():
    if count == 1:
        return x
```

Now each number is processed a constant number of times:

```text
Time: O(n)
Space: O(n)
```

This is often the first good solution, but it uses extra memory proportional to the number of distinct values.

So the deeper question is:

> Can we keep only one running state and still make duplicate pairs disappear?

---

### 3. The Key Observation: Pairs Should Cancel

The input has only two kinds of values:

```text
values that appear twice
one value that appears once
```

If we had an operation where:

```text
x combined with x disappears
x combined with neutral_state leaves x unchanged
order does not matter
```

then the whole problem would become simple.

All duplicate pairs would cancel each other, and the single value would remain.

Addition almost works, but not quite:

```text
x + x = 2x
```

A duplicate does not disappear under addition.

Subtraction is tempting, but order matters:

```text
4 - 1 - 2 - 1 - 2
```

does not reliably leave `4`.

The operation we need is XOR.

---

### 4. What XOR Means From First Principles

XOR is a bitwise operation.

For one bit, XOR answers this question:

> Are the two bits different?

The truth table is:

```text
0 ^ 0 = 0
0 ^ 1 = 1
1 ^ 0 = 1
1 ^ 1 = 0
```

So equal bits cancel to `0`, and different bits produce `1`.

For integers, XOR applies that same rule independently at every bit position.

For example:

```text
5 = 101
5 = 101

5 ^ 5 = 000 = 0
```

Every bit is equal to itself, so every bit cancels.

Also:

```text
0 ^ 5 = 5
```

because XOR with `0` leaves each bit unchanged:

```text
0 ^ 1 = 1
0 ^ 0 = 0
```

These two identities are exactly what this problem needs:

```text
x ^ x = 0
0 ^ x = x
```

There is one more important property:

```text
XOR is associative and commutative.
```

That means grouping and order do not matter:

```text
(a ^ b) ^ c = a ^ (b ^ c)
a ^ b = b ^ a
```

So if the array is:

```text
[4, 1, 2, 1, 2]
```

then XORing all values is equivalent to rearranging them as:

```text
4 ^ (1 ^ 1) ^ (2 ^ 2)
```

The pairs cancel:

```text
4 ^ 0 ^ 0 = 4
```

The single number remains.

---

### 5. The XOR Invariant

Maintain one variable:

```text
answer = XOR of all numbers processed so far
```

At the start, before processing anything:

```text
answer = 0
```

That is correct because the XOR of an empty set of numbers should be the neutral state.

After reading a new number `x`, update:

```text
answer = answer ^ x
```

The invariant becomes:

```text
After processing nums[0:i], answer equals nums[0] ^ nums[1] ^ ... ^ nums[i].
```

Why is that useful?

Because once all numbers have been processed, `answer` equals the XOR of the entire array.

The problem guarantee says every non-answer value appears exactly twice. Since each duplicate pair contributes:

```text
x ^ x = 0
```

all pairs vanish from the final XOR.

Only the single number contributes once, so it is the final value of `answer`.

---

### 6. Detailed Algorithm

1. Initialize `answer` to `0`.
2. Iterate through every number `num` in `nums`.
3. Replace `answer` with `answer ^ num`.
4. After the loop, return `answer`.

In pseudocode:

```text
answer = 0

for num in nums:
    answer = answer XOR num

return answer
```

In Python-style code:

```python
def singleNumber(nums: list[int]) -> int:
    answer = 0

    for num in nums:
        answer ^= num

    return answer
```

This uses `answer` as a compact record of parity.

A value that has appeared an even number of times contributes nothing to the final XOR.
A value that has appeared an odd number of times contributes itself.

In this problem, every duplicate appears exactly twice, and the target appears once, so the odd-parity value is the answer.

---

### 7. Walkthrough: `[2, 2, 1]`

Start:

```text
answer = 0
```

Process `2`:

```text
answer = 0 ^ 2 = 2
```

The running XOR says: among the processed values, `2` has appeared odd times.

Process the second `2`:

```text
answer = 2 ^ 2 = 0
```

The two `2`s cancel.

Process `1`:

```text
answer = 0 ^ 1 = 1
```

End:

```text
return 1
```

---

### 8. Walkthrough: `[4, 1, 2, 1, 2]`

Use decimal values for readability:

```text
answer = 0
```

Process `4`:

```text
answer = 0 ^ 4 = 4
```

Process `1`:

```text
answer = 4 ^ 1 = 5
```

This intermediate value does not need to be meaningful as a candidate answer. It is just the XOR of everything seen so far.

Process `2`:

```text
answer = 5 ^ 2 = 7
```

Process the second `1`:

```text
answer = 7 ^ 1 = 6
```

Process the second `2`:

```text
answer = 6 ^ 2 = 4
```

End:

```text
return 4
```

To see the cancellation more directly, ignore the scan order and group equal values:

```text
4 ^ 1 ^ 2 ^ 1 ^ 2
= 4 ^ (1 ^ 1) ^ (2 ^ 2)
= 4 ^ 0 ^ 0
= 4
```

The loop is just a left-to-right way of computing the same expression.

---

### 9. Correctness

We prove that the algorithm returns the single number.

#### Invariant

After processing the first `k` numbers, `answer` equals the XOR of exactly those first `k` numbers.

#### Initialization

Before the loop, no numbers have been processed.

```text
answer = 0
```

This is the correct XOR value for an empty processed prefix, because `0` is the neutral value for XOR.

#### Maintenance

Assume the invariant is true before processing the next number `num`.

So `answer` is the XOR of all previously processed numbers.

The algorithm updates:

```text
answer = answer ^ num
```

Therefore, the new `answer` is the XOR of all previously processed numbers plus the new number.

So the invariant remains true.

#### Termination

When the loop ends, every number in `nums` has been processed.

By the invariant, `answer` equals the XOR of the entire array.

Every value except the single number appears exactly twice. Each duplicate pair cancels because:

```text
x ^ x = 0
```

The remaining single number is not canceled. XOR with `0` leaves it unchanged.

Therefore, the final `answer` is exactly the number that appears once.

So the algorithm is correct.

---

### 10. Complexity

Let `n` be the length of `nums`.

The algorithm scans the array once:

```text
Time: O(n)
```

It stores only one integer variable, regardless of input size:

```text
Space: O(1)
```

This satisfies the usual follow-up requirement: linear runtime with constant extra memory.

---

### 11. Common Pitfalls

#### Pitfall 1: Using a hash map when constant space is required

A frequency map is correct, but it uses `O(n)` extra space in the worst case.

For this problem, XOR is preferred because the input guarantee is stronger than a general frequency-counting problem.

#### Pitfall 2: Thinking the running XOR must always be a candidate answer

During the scan, intermediate values may not appear in the array.

For example:

```text
4 ^ 1 = 5
```

The value `5` is not a candidate answer in `[4, 1, 2, 1, 2]`.

That is fine. The invariant is not “`answer` is currently the single number.”

The invariant is:

```text
answer is the XOR of all processed values.
```

Only after all duplicate pairs have been included does the final answer emerge.

#### Pitfall 3: Replacing XOR with OR or AND

OR does not cancel duplicates:

```text
x | x = x
```

AND does not cancel duplicates either:

```text
x & x = x
```

The cancellation identity is specific to XOR:

```text
x ^ x = 0
```

#### Pitfall 4: Worrying unnecessarily about negative numbers in Python

Python integers are unbounded, but this particular algorithm does not need a fixed-width mask.

Because every duplicated value is exactly equal to itself, even negative duplicates cancel correctly:

```text
(-3) ^ (-3) = 0
```

The same XOR identities still hold.

Fixed-width masking becomes important in other bit-manipulation problems, especially ones involving complements or manual bit reconstruction. It is not needed here.

#### Pitfall 5: Forgetting the problem guarantee

This algorithm depends on the exact promise:

```text
one value appears once
all other values appear exactly twice
```

If values could appear three times, or if two different values appeared once, this exact solution would not solve the modified problem.

---

### 12. First-Principles Summary

The problem asks us to remove all duplicate pairs and keep the one unpaired value.

XOR is the natural operation because it has exactly the cancellation behavior we need:

```text
x ^ x = 0
0 ^ x = x
```

Since XOR is associative and commutative, the order of the array does not matter mathematically. The loop simply computes the XOR of all values from left to right.

Every paired value cancels with its duplicate. The only value with no duplicate remains.

So the whole solution is:

```text
Keep a running XOR of the array.
Return it after the scan.
```

## Implementation
See `solutions/bit_manipulation/p136_single_number.py`.

## Tests
See `tests/bit_manipulation/test_p136_single_number.py`.

## Examples

### Example 1
- Input: `nums = [2, 2, 1]`
- Output: `1`
- Explanation: `2 ^ 2 = 0`, and `0 ^ 1 = 1`.

### Example 2
- Input: `nums = [4, 1, 2, 1, 2]`
- Output: `4`
- Explanation: The two `1`s cancel, the two `2`s cancel, and `4` remains.

### Example 3
- Input: `nums = [1]`
- Output: `1`
- Explanation: There are no duplicate pairs, so the only value is the single number.

## Follow-up Practice

- Trace `[7, 3, 7]` by writing the value of `answer` after each step.
- Rewrite `[4, 1, 2, 1, 2]` by grouping equal values before applying XOR.
- Explain why `answer = 0` is the correct starting value.
- Compare the hash-map solution with the XOR solution in both time and space.
