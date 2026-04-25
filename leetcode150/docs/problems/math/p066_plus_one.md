# 66. Plus One

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/plus-one/
- Official Group: Math
- Pattern Group: Math
- Patterns: math

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an array `digits` that represents a non-negative integer.

Each entry is one decimal digit:

```text
digits[i] is between 0 and 9
```

The digits are ordered from most significant to least significant:

```text
[1, 2, 3] represents 123
[4, 3, 2, 1] represents 4321
[9] represents 9
```

The task is to add one to that represented number and return the resulting digits in the same array form.

So the mathematical operation is simple:

```text
number represented by digits + 1
```

But the implementation constraint matters:

> We should solve the problem directly on the digit representation, not by relying on converting the whole array into an integer.

That distinction is important because this problem is really about understanding how grade-school addition works at the digit level.

### 2. Start From the Brute Force Baseline

The most direct idea is:

1. Convert the digit array into an integer.
2. Add `1`.
3. Convert the result back into digits.

Conceptually:

```python
number = 0

for digit in digits:
    number = number * 10 + digit

number += 1

return digits_of(number)
```

For example:

```text
digits = [1, 2, 3]

number = 123
number + 1 = 124
answer = [1, 2, 4]
```

This is easy to understand, but it misses the point of the problem.

In languages with fixed-width integers, the represented number may be too large to fit in `int` or `long`. Even in Python, where integers can grow arbitrarily large, converting the whole number is unnecessary work. The input is already split into exactly the units we need: decimal digits.

The better question is:

> When adding one, which digits can actually change?

### 3. The Key Observation

Adding one starts at the ones place, which is the last element of the array.

For most digits, there is no carry:

```text
0 + 1 = 1
1 + 1 = 2
...
8 + 1 = 9
```

If the last digit is anything from `0` through `8`, only that last digit changes.

For example:

```text
[1, 2, 3] -> [1, 2, 4]
[4, 3, 2, 1] -> [4, 3, 2, 2]
```

The only special digit is `9`:

```text
9 + 1 = 10
```

A single decimal digit cannot store `10`, so that position becomes `0` and sends a carry of `1` to the digit on its left.

For example:

```text
[1, 2, 9] + 1
         9 + 1 = 10
         write 0, carry 1

result: [1, 3, 0]
```

So the entire problem is controlled by a suffix of trailing `9`s.

Examples:

```text
[1, 2, 3] has no trailing 9s      -> increment 3
[1, 2, 9] has one trailing 9      -> turn 9 to 0, increment 2
[1, 9, 9] has two trailing 9s     -> turn both 9s to 0, increment 1
[9, 9, 9] is all trailing 9s      -> turn all to 0, add new leading 1
```

This is the first-principles insight:

> Adding one only propagates left while the current digit is `9`.

As soon as we find a digit less than `9`, we can increment it and stop.

### 4. The Carry Invariant

A clean way to reason about the algorithm is to track the carry.

Initially:

```text
carry = 1
```

because the operation is "plus one."

We process digits from right to left. At any moment, the invariant is:

```text
All positions to the right of the current index already contain
the correct final digits, assuming the current carry must still be
added to the current index.
```

This invariant captures grade-school addition exactly.

When the current digit is less than `9`:

```text
digit + carry <= 9
```

So we can add the carry into this digit, set `carry` to `0`, and stop because nothing further left changes.

When the current digit is `9` and `carry` is `1`:

```text
9 + 1 = 10
```

So the current digit becomes `0`, and the carry remains `1` for the next digit to the left.

The important part is not just that the algorithm works, but why it is safe to stop early:

> Once the carry becomes `0`, every digit to the left is unchanged.

No remaining arithmetic can affect those more significant digits.

### 5. Detailed Algorithm

Use the input array as the working digit array.

1. Start from the last index, because addition begins at the ones place.
2. If `digits[i]` is less than `9`, increment it and return immediately.
3. If `digits[i]` is `9`, set it to `0` and continue one position left.
4. If the loop finishes, every original digit was `9`.
5. In that all-`9` case, insert or create a leading `1` before the zeros.

In Python-like pseudocode:

```python
def plusOne(digits):
    for i in range(len(digits) - 1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits

        digits[i] = 0

    return [1] + digits
```

The early return is the central optimization. It is not a trick; it follows directly from the carry invariant. If a digit can absorb the carry without becoming `10`, the carry disappears, and there is no reason to inspect any earlier digit.

### 6. Walk Through the Examples

#### Example 1: `[1, 2, 3]`

Start at the last digit:

```text
[1, 2, 3]
       ^
```

The current digit is `3`, which is less than `9`.

So it can absorb the `+1` directly:

```text
3 + 1 = 4
```

The carry is gone, and all earlier digits stay the same:

```text
[1, 2, 4]
```

#### Example 2: `[4, 3, 2, 1]`

Start at the last digit:

```text
[4, 3, 2, 1]
          ^
```

The current digit is `1`, so:

```text
1 + 1 = 2
```

No carry remains. Return:

```text
[4, 3, 2, 2]
```

#### Example 3: `[9]`

Start at the only digit:

```text
[9]
 ^
```

The digit is `9`:

```text
9 + 1 = 10
```

So write `0` in this position and carry `1` left:

```text
[0]
```

There is no digit left to receive the carry. That means the number gained a new most significant digit.

So prepend `1`:

```text
[1, 0]
```

#### Additional Carry Example: `[1, 9, 9]`

This example shows carry propagation more clearly.

Start at the last digit:

```text
[1, 9, 9]
       ^
```

Last digit:

```text
9 + 1 = 10
write 0, carry 1
```

Array becomes:

```text
[1, 9, 0]
    ^
```

Next digit is also `9`:

```text
9 + 1 = 10
write 0, carry 1
```

Array becomes:

```text
[1, 0, 0]
 ^
```

Now the current digit is `1`, which is less than `9`:

```text
1 + 1 = 2
```

Carry disappears, so we return:

```text
[2, 0, 0]
```

This matches the arithmetic fact:

```text
199 + 1 = 200
```

### 7. Correctness Argument

We prove that the algorithm returns the digit representation of the original number plus one.

The algorithm processes digits from right to left, just like decimal addition.

At the start, the pending carry is exactly `1`, which represents the required `+1` operation.

For each processed digit, there are two cases.

If the digit is less than `9`, then adding the carry produces a valid single digit. The algorithm increments that digit and returns. All digits to its right have already been corrected by the previous carry steps, and all digits to its left are unchanged because the carry is now gone. Therefore the entire array represents the original number plus one.

If the digit is `9`, then adding the carry produces `10`. The correct result for this position is `0`, with carry `1` sent to the next position on the left. The algorithm does exactly that by setting the digit to `0` and continuing left. Therefore, after this step, the processed suffix is correct and the remaining work is exactly to add the carry to the next unprocessed digit.

If the loop finishes without returning, every digit was `9`. Each original digit correctly becomes `0`, and one carry remains beyond the most significant position. The only valid decimal representation is therefore a new leading `1` followed by all zeros. The algorithm returns exactly that.

Thus, in all cases, the returned array represents the input number plus one.

### 8. Complexity

Let `n` be the number of digits.

- Time: `O(n)` in the worst case, when all digits are `9` and the carry travels through the entire array.
- Space: `O(1)` extra space if mutating and returning the input array, except for the all-`9` case where the returned result needs one additional digit.

The best case is faster in practice: if the last digit is not `9`, the algorithm returns after one step.

### 9. Common Pitfalls

- Converting the entire array to an integer, which avoids the digit-level reasoning and may overflow in fixed-width languages.
- Forgetting the all-`9` case, such as `[9] -> [1, 0]` or `[9, 9, 9] -> [1, 0, 0, 0]`.
- Continuing to scan after incrementing a digit less than `9`; once the carry disappears, earlier digits must not change.
- Appending `1` at the end for the all-`9` case instead of adding it at the front.
- Mishandling the order of digits: the array is most-significant first, but addition starts from the least-significant end.
- Assuming only the last digit can change; trailing `9`s force changes farther left.

### 10. First-Principles Summary

The input is already a decimal representation split into digits. Adding one is therefore the same operation taught in grade-school addition: start at the ones place, write the resulting digit, and carry if the result reaches `10`.

The only digit that can create a carry when adding one is `9`. Every trailing `9` becomes `0`, and the carry moves one step left. The first digit less than `9` absorbs the carry and increases by one. If no such digit exists, the original number was all `9`s, so the result needs a new leading `1`.

The whole algorithm is just this invariant made explicit:

```text
processed suffix is already correct, and at most one carry remains
to apply to the next digit on the left
```

## Implementation
See `solutions/math/p066_plus_one.py`.

## Tests
See `tests/math/test_p066_plus_one.py`.

## Examples

### Example 1
- Input: `{'digits': [1, 2, 3]}`
- Output: `[1, 2, 4]`

### Example 2
- Input: `{'digits': [4, 3, 2, 1]}`
- Output: `[4, 3, 2, 2]`

### Example 3
- Input: `{'digits': [9]}`
- Output: `[1, 0]`

## Follow-up Practice

- Trace the carry on `[1, 2, 9]`, `[1, 9, 9]`, and `[9, 9, 9]`.
- Explain why the algorithm can return immediately after incrementing a digit less than `9`.
- Rewrite the same idea using an explicit `carry` variable, then compare it with the shorter trailing-`9` version.
