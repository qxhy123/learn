# 201. Bitwise AND of Numbers Range

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/bitwise-and-of-numbers-range/
- Official Group: Bit Manipulation
- Pattern Group: Bit Manipulation
- Patterns: bit-manipulation

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given two integers:

```text
left
right
```

return the bitwise AND of every integer in the inclusive range:

```text
left, left + 1, left + 2, ..., right
```

In mathematical shorthand, the answer is:

```text
left & (left + 1) & (left + 2) & ... & right
```

For example:

```text
left  = 5
right = 7
```

The numbers are:

```text
5, 6, 7
```

In binary:

```text
5 = 101
6 = 110
7 = 111
```

Their bitwise AND is:

```text
  101
& 110
& 111
= 100
```

So the answer is:

```text
4
```

The important first-principles question is:

> Which bit positions stay `1` in every number from `left` through `right`?

Bitwise AND is strict. A bit in the final answer is `1` only if that bit is `1` in every single number in the range. If even one number has `0` at that position, the final answer has `0` there.

---

### 2. Start From the Brute Force Baseline

The most direct solution is to AND all numbers one by one:

```python
answer = left

for value in range(left + 1, right + 1):
    answer &= value

return answer
```

This is correct because it follows the definition exactly.

But it can be far too slow when the range is large. For example:

```text
left  = 1
right = 2147483647
```

The brute force loop would try to process more than two billion numbers.

So the deeper question is:

> Can we determine which bits survive without visiting every number?

Yes. The answer depends only on the binary structure shared by `left` and `right`.

---

### 3. Bitwise AND Removes Any Changing Bit

Consider one bit position independently.

For that bit to be `1` in the range AND, every number in the interval must have `1` there.

If the range contains both:

```text
a number with this bit = 0
and
a number with this bit = 1
```

then the AND at that position becomes `0`.

So a bit can survive only if it does not change anywhere between `left` and `right`.

This is why the high bits are special. High bits change slowly. Low bits change frequently.

For example, from `5` to `7`:

```text
5 = 101
6 = 110
7 = 111
```

The leftmost bit is always `1`:

```text
1__
```

The lower two bits change:

```text
_01
_10
_11
```

Since those lower positions are not constant across the whole range, they must become `0` in the final AND.

That leaves:

```text
100
```

---

### 4. The Key Observation: Only the Common Binary Prefix Survives

Write `left` and `right` in binary.

The range includes every integer between them. If `left` and `right` differ at some bit position, then all less significant positions to the right cannot be trusted to stay fixed across the entire range.

Why?

Because moving from `left` upward to `right` crosses binary increments. Once a higher differing bit changes, the lower bits cycle through combinations as counting proceeds.

So the answer is:

```text
common binary prefix of left and right
followed by zeros
```

For example:

```text
left  = 5 = 101
right = 7 = 111
```

Compare from the most significant side:

```text
101
111
^
```

The common prefix is:

```text
1
```

After that prefix, the numbers differ, so every lower bit in the answer becomes `0`:

```text
100
```

This gives `4`.

Another example:

```text
left  = 26 = 11010
right = 30 = 11110
```

Compare:

```text
11010
11110
^^
```

The common prefix is:

```text
11
```

All remaining lower bits become `0`:

```text
11000
```

So the range AND is:

```text
24
```

---

### 5. The Common-Prefix Invariant

The clean algorithm repeatedly removes the lowest bit from both `left` and `right` until they become equal.

Removing the lowest bit means shifting right by one:

```text
x >>= 1
```

Each shift asks:

> Do `left` and `right` already have the same remaining high-bit prefix?

Maintain this invariant:

```text
After shifting both numbers k times, any bit removed during those k shifts cannot survive in the final answer.
```

Why is that true?

If `left != right`, then the interval spans at least one change somewhere in the remaining binary representation. The least significant positions below that change are not guaranteed to remain constant across all numbers in the range. Those removed positions must be `0` in the final AND.

When the shifted values finally become equal, that value is exactly the common high-bit prefix shared by the original `left` and `right`.

Then we shift it back left by the same number of positions:

```text
common_prefix << k
```

Shifting back appends `k` zeros, representing the lower bits that were proven unable to survive.

---

### 6. Detailed Algorithm

1. Initialize:

```text
shift = 0
```

2. While `left` and `right` are different:

```text
left  >>= 1
right >>= 1
shift += 1
```

This repeatedly discards one low bit from each boundary.

3. When the loop stops, `left == right`.

At that moment, `left` is the common prefix.

4. Restore the prefix to its original position by shifting left:

```text
return left << shift
```

The restored low bits are zeros, which is exactly what the range AND requires.

---

### 7. Pseudocode

```python
def rangeBitwiseAnd(left: int, right: int) -> int:
    shift = 0

    while left != right:
        left >>= 1
        right >>= 1
        shift += 1

    return left << shift
```

This is the common-prefix version.

There is another popular equivalent version:

```python
def rangeBitwiseAnd(left: int, right: int) -> int:
    while right > left:
        right &= right - 1

    return right
```

That version repeatedly clears the lowest set bit of `right` until `right` no longer exceeds `left`. It works because any low set bit that can be crossed within the range cannot survive the AND.

Both approaches are based on the same idea:

```text
bits that change inside the range become 0
only the shared high prefix remains
```

The shift-based common-prefix algorithm makes that idea especially explicit.

---

### 8. Walkthrough of Example 1

Input:

```text
left  = 5
right = 7
```

Binary:

```text
left  = 101
right = 111
```

They are not equal, so remove the lowest bit from both:

```text
left  = 10   # 2
right = 11   # 3
shift = 1
```

They are still not equal, so shift again:

```text
left  = 1
right = 1
shift = 2
```

Now they are equal. The common prefix is:

```text
1
```

Shift it back by `2` positions:

```text
1 << 2 = 100
```

So the answer is:

```text
4
```

This matches the direct calculation:

```text
5 & 6 & 7 = 4
```

---

### 9. Walkthrough of a Larger Range

Consider:

```text
left  = 18
right = 23
```

Binary:

```text
18 = 10010
19 = 10011
20 = 10100
21 = 10101
22 = 10110
23 = 10111
```

The common prefix is:

```text
10___
```

The remaining three positions change somewhere in the range, so they become zeros:

```text
10000
```

The answer is `16`.

Now trace the algorithm:

```text
left = 10010, right = 10111, shift = 0
left = 1001,  right = 1011,  shift = 1
left = 100,   right = 101,   shift = 2
left = 10,    right = 10,    shift = 3
```

The common prefix is:

```text
10
```

Append three zeros:

```text
10 << 3 = 10000
```

So the answer is:

```text
16
```

---

### 10. Why Crossing a Power of Two Often Produces Zero

A useful edge case is a range that crosses a power-of-two boundary.

For example:

```text
left  = 7  = 0111
right = 8  = 1000
```

There is no shared leading `1` within the relevant width:

```text
0111
1000
```

Every bit position is `0` in at least one number in the range.

So:

```text
7 & 8 = 0
```

This also explains the third official example:

```text
left  = 1
right = 2147483647
```

The range includes many powers of two and spans almost the entire positive 31-bit space. No positive bit can remain `1` across every number in that interval, so the answer is:

```text
0
```

---

### 11. Correctness

We prove that the common-prefix algorithm returns the bitwise AND of every number in `[left, right]`.

Let the original inputs be `L` and `R`.

The algorithm repeatedly shifts both boundaries right until they become equal. Suppose it performs `k` shifts.

After `k` shifts, the two shifted values are equal:

```text
L >> k == R >> k
```

This means the original `L` and `R` share the same binary prefix above the lowest `k` bit positions.

All bits above those `k` positions are fixed across the entire range. Since both endpoints have the same prefix there, every integer between them also has that same prefix. Therefore those prefix bits must appear in the final AND.

Now consider the `k` removed low positions. The loop removed them only while the shifted boundaries were still different. Those positions are at or below the part of the number that changes while counting from `L` to `R`. Across the full interval, each such position fails to remain `1` for every number. Since bitwise AND keeps a `1` only when every number has `1` in that position, all removed positions must be `0` in the final answer.

The algorithm returns:

```text
(L >> k) << k
```

That is exactly:

```text
the common prefix of L and R, followed by k zeros
```

The common prefix is precisely the set of bits that survive the range AND, and the lower changing bits are precisely the bits that become `0`.

Therefore the algorithm returns the correct range bitwise AND.

---

### 12. Complexity

The loop shifts both numbers until their common prefix remains.

For non-negative 32-bit LeetCode inputs, this takes at most `31` shifts.

More generally, if `W` is the number of bits needed to represent `right`, then:

```text
Time:  O(W)
Space: O(1)
```

Because LeetCode constrains the values to a fixed integer width, this is often described as:

```text
Time:  O(1)
Space: O(1)
```

The important practical point is that the runtime depends on the number of bits, not on the number of integers in the range.

---

### 13. Common Pitfalls

#### Pitfall 1: Looping Through the Whole Range

This works for small examples but fails for large ranges:

```python
for value in range(left, right + 1):
    answer &= value
```

The interval length can be enormous, while the bit-width is small.

#### Pitfall 2: Thinking Only Equal `1` Bits in the Endpoints Matter

It is not enough for a bit to be `1` in both `left` and `right`.

It must be `1` in every number between them.

For example:

```text
left  = 10 = 1010
right = 14 = 1110
```

Both endpoints have the `2` bit set:

```text
1010
1110
  ^
```

But the range includes:

```text
12 = 1100
```

where that bit is `0`, so it cannot survive.

The correct answer is based on the common prefix, not just endpoint bit agreement.

#### Pitfall 3: Forgetting to Shift Back

After finding the common prefix, the algorithm must restore it to the original bit position:

```python
return left << shift
```

Returning `left` directly would return only the compressed prefix.

For `left = 5`, `right = 7`, the prefix is `1`, but the answer is `100`, which is `4`.

#### Pitfall 4: Off-by-One Thinking About the Range

The range is inclusive:

```text
[left, right]
```

So both endpoints participate in the AND.

The common-prefix method naturally handles inclusivity because it compares the actual boundary values.

#### Pitfall 5: Using Signed-Integer Assumptions Unnecessarily

The LeetCode problem uses non-negative integers. In Python, right shift on non-negative integers behaves exactly like dropping low binary bits.

If adapting this idea to signed fixed-width integers in another setting, define the width and sign behavior explicitly.

---

### 14. First-Principles Summary

Bitwise AND asks which bit positions are `1` in every input number.

For a whole integer range, low bits change frequently as the numbers count upward. Any bit that changes within the range is guaranteed not to survive the AND.

The only bits that can survive are the high bits shared by both ends of the range before their first binary difference.

So the problem reduces from:

```text
AND every number in the interval
```

to:

```text
find the common binary prefix of left and right, then fill the rest with zeros
```

That is why repeatedly shifting `left` and `right` right until they match gives the answer. The shifts remove changing low bits; shifting the shared prefix back appends the required zeros.

## Implementation
See `solutions/bit_manipulation/p201_bitwise_and_of_numbers_range.py`.

## Tests
See `tests/bit_manipulation/test_p201_bitwise_and_of_numbers_range.py`.

## Examples

### Example 1
- Input: `{'left': 5, 'right': 7}`
- Output: `4`

### Example 2
- Input: `{'left': 0, 'right': 0}`
- Output: `0`

### Example 3
- Input: `{'left': 1, 'right': 2147483647}`
- Output: `0`

## Follow-up Practice

- Write `left` and `right` in binary, then mark their shared prefix.
- Trace how many right shifts are needed before the two values become equal.
- Check ranges that cross a power of two, such as `[7, 8]` or `[15, 16]`.
- Compare the common-prefix method with the alternative `right &= right - 1` method.
