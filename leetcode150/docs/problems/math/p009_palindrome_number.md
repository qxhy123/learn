# 9. Palindrome Number

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/palindrome-number/
- Official Group: Math
- Pattern Group: Math
- Patterns: math

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given an integer `x`, decide whether its decimal representation reads the same from left to right and from right to left.

For example:

```text
x = 121
```

The digits are:

```text
1 2 1
```

Reading from the left gives `121`.
Reading from the right also gives `121`.
So the answer is `True`.

But for:

```text
x = -121
```

the written integer is:

```text
- 1 2 1
```

The minus sign is part of the representation. Reversing the characters would put the minus sign at the end:

```text
1 2 1 -
```

That is not the same integer representation, so the answer is `False`.

For:

```text
x = 10
```

the reverse of the digits is `01`, which represents `1`, not `10`. The trailing zero in the original number becomes a leading zero after reversal, so the answer is `False`.

The real question is:

> Can we compare the front half and back half of the decimal digits of `x` without converting the whole number into a string?

The follow-up for this classic problem usually asks for a numeric solution. That means we should reason from decimal place value rather than from characters.

---

### 2. Start From the Brute Force Baseline

The simplest correct idea is to convert the integer to a string and compare it with its reverse:

```python
def is_palindrome_string(x: int) -> bool:
    text = str(x)
    return text == text[::-1]
```

This works because a palindrome is fundamentally a symmetry property of a sequence.

For `121`:

```text
text        = "121"
text[::-1] = "121"
```

For `10`:

```text
text        = "10"
text[::-1] = "01"
```

For `-121`:

```text
text        = "-121"
text[::-1] = "121-"
```

This baseline is easy to understand and often acceptable in normal programming. However, it uses `O(d)` extra space for the string, where `d` is the number of decimal digits.

We can do better by using the same idea directly on digits.

---

### 3. Decimal Digits From First Principles

Every non-negative integer is built from powers of ten.

For example:

```text
12321 = 1 * 10000
      + 2 * 1000
      + 3 * 100
      + 2 * 10
      + 1
```

The last digit is easy to extract:

```text
12321 % 10 = 1
```

Removing the last digit is also easy:

```text
12321 // 10 = 1232
```

So repeated `% 10` and `// 10` let us read digits from right to left.

A palindrome, however, requires matching left-to-right and right-to-left structure. One direct numeric approach is:

1. Reverse all digits of `x` into a new integer.
2. Compare the reversed integer with the original.

For example:

```text
original = 121

reverse = 0
take 1: reverse = 0 * 10 + 1 = 1
take 2: reverse = 1 * 10 + 2 = 12
take 1: reverse = 12 * 10 + 1 = 121
```

At the end, `reverse == original`, so `121` is a palindrome.

This full-reversal approach is correct, but in fixed-width integer languages it can overflow for large inputs. Python integers do not overflow, but the first-principles improvement is still useful: we do not need to reverse the entire number.

---

### 4. Key Observation: Only Half the Digits Are Needed

To decide whether a number is a palindrome, we only need to compare mirrored halves.

For an even number of digits:

```text
x = 1221

left half  = 12
right half = 21
reversed right half = 12
```

The number is a palindrome because the left half equals the reversed right half.

For an odd number of digits:

```text
x = 12321

left half   = 12
middle      = 3
right half  = 21
reversed right half = 12
```

The middle digit does not need to match anything. It can be ignored.

This suggests the algorithm:

```text
Move digits from the end of x into a reversed number.
Stop when the reversed part has at least as many digits as the remaining part.
Then compare the two halves.
```

Instead of reversing all digits, we reverse only the right half.

---

### 5. Handle Impossible Cases First

Before building the reversal, two cases can be decided immediately.

#### Negative numbers

Any negative number is not a palindrome under the standard decimal representation:

```text
-121 != 121-
```

So:

```python
if x < 0:
    return False
```

#### Positive numbers ending in zero

A positive number ending in `0` cannot be a palindrome.

For example:

```text
10, 100, 120, 1230
```

If such a number were a palindrome, its first digit would also have to be `0`. But normal decimal notation never writes leading zeros for positive integers.

The only exception is the number `0` itself, which is a palindrome.

So:

```python
if x != 0 and x % 10 == 0:
    return False
```

This condition is important because otherwise `10` can be mishandled by half-reversal logic.

---

### 6. The Digit/Reversal Invariant

We maintain two integers:

```text
remaining = the part of x whose right-side digits have not yet been moved
reversed_half = the reverse of the digits moved out of remaining
```

Initially:

```text
remaining = x
reversed_half = 0
```

Each step removes the last digit from `remaining` and appends it to `reversed_half`:

```text
digit = remaining % 10
reversed_half = reversed_half * 10 + digit
remaining = remaining // 10
```

The invariant is:

```text
After k steps, reversed_half contains the last k digits of the original number,
but in reverse order, and remaining contains the original number without those k digits.
```

Example with `12321`:

```text
start:
remaining     = 12321
reversed_half = 0

after moving last digit 1:
remaining     = 1232
reversed_half = 1

after moving last digit 2:
remaining     = 123
reversed_half = 12

after moving last digit 3:
remaining     = 12
reversed_half = 123
```

At that point `reversed_half` has more digits than `remaining`, which tells us we crossed the middle of an odd-length number.

---

### 7. When Should the Loop Stop?

We continue while:

```text
remaining > reversed_half
```

Why does this work?

`reversed_half` starts with fewer digits than `remaining`. Each iteration moves one digit from `remaining` to `reversed_half`, so `remaining` loses one digit and `reversed_half` gains one digit.

Eventually one of two things happens.

#### Even digit count

For `1221`:

```text
start: remaining = 1221, reversed_half = 0
step1: remaining = 122,  reversed_half = 1
step2: remaining = 12,   reversed_half = 12
```

Now the two halves have the same length and value.

The loop stops because:

```text
remaining > reversed_half
12 > 12 is false
```

For an even-length palindrome, we need:

```text
remaining == reversed_half
```

#### Odd digit count

For `12321`:

```text
start: remaining = 12321, reversed_half = 0
step1: remaining = 1232,  reversed_half = 1
step2: remaining = 123,   reversed_half = 12
step3: remaining = 12,    reversed_half = 123
```

Now `reversed_half` has one extra digit: the middle digit `3`.

The middle digit does not matter, so remove it with integer division:

```text
reversed_half // 10 = 12
```

For an odd-length palindrome, we need:

```text
remaining == reversed_half // 10
```

Therefore the final test is:

```python
return remaining == reversed_half or remaining == reversed_half // 10
```

---

### 8. Detailed Algorithm

1. If `x` is negative, return `False`.
2. If `x` is not zero and ends in zero, return `False`.
3. Set `remaining = x`.
4. Set `reversed_half = 0`.
5. While `remaining > reversed_half`:
   1. Extract the last digit of `remaining`.
   2. Append that digit to `reversed_half`.
   3. Drop the last digit from `remaining`.
6. Return `True` if either:
   - the two halves are exactly equal, or
   - the remaining left half equals `reversed_half` after dropping its middle digit.

---

### 9. Pseudocode

```text
function isPalindrome(x):
    if x < 0:
        return false

    if x != 0 and x % 10 == 0:
        return false

    remaining = x
    reversed_half = 0

    while remaining > reversed_half:
        digit = remaining % 10
        reversed_half = reversed_half * 10 + digit
        remaining = remaining // 10

    return remaining == reversed_half
        or remaining == reversed_half // 10
```

Equivalent Python implementation:

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False

        if x != 0 and x % 10 == 0:
            return False

        remaining = x
        reversed_half = 0

        while remaining > reversed_half:
            digit = remaining % 10
            reversed_half = reversed_half * 10 + digit
            remaining //= 10

        return remaining == reversed_half or remaining == reversed_half // 10
```

---

### 10. Walkthrough: `x = 121`

Start:

```text
remaining = 121
reversed_half = 0
```

First iteration:

```text
digit = 121 % 10 = 1
reversed_half = 0 * 10 + 1 = 1
remaining = 121 // 10 = 12
```

State:

```text
remaining = 12
reversed_half = 1
```

The loop continues because:

```text
12 > 1
```

Second iteration:

```text
digit = 12 % 10 = 2
reversed_half = 1 * 10 + 2 = 12
remaining = 12 // 10 = 1
```

State:

```text
remaining = 1
reversed_half = 12
```

The loop stops because:

```text
1 > 12 is false
```

This is an odd-length number, so `reversed_half` contains the middle digit `2` plus the reversed right side. Drop the middle digit:

```text
reversed_half // 10 = 12 // 10 = 1
```

Compare:

```text
remaining == reversed_half // 10
1 == 1
```

Return `True`.

---

### 11. Walkthrough: `x = 1221`

Start:

```text
remaining = 1221
reversed_half = 0
```

Move the last digit:

```text
digit = 1
reversed_half = 1
remaining = 122
```

Move the next digit:

```text
digit = 2
reversed_half = 12
remaining = 12
```

The loop stops because `remaining > reversed_half` is now false:

```text
12 > 12 is false
```

This is an even-length number, so the halves should be exactly equal:

```text
remaining == reversed_half
12 == 12
```

Return `True`.

---

### 12. Walkthrough: `x = 10`

Before the loop, check the trailing-zero rule:

```text
x != 0 is true
x % 10 == 0 is true
```

So return `False` immediately.

This is correct because `10` reversed is `01`, and normal integer notation does not preserve that leading zero.

---

### 13. Correctness

We prove that the algorithm returns `True` exactly when `x` is a palindrome.

#### Lemma 1: The loop invariant is maintained

At the start, `remaining = x` and `reversed_half = 0`, so no digits have been moved. The invariant is true.

During one loop iteration, `remaining % 10` extracts the last unmoved digit. Multiplying `reversed_half` by `10` shifts its digits left by one decimal place, and adding the extracted digit appends that digit to the right. Then `remaining // 10` removes the extracted digit from `remaining`.

Therefore, after each iteration, `reversed_half` is exactly the reverse of the suffix removed from the original number, and `remaining` is exactly the prefix that has not been removed. The invariant is maintained.

#### Lemma 2: When the loop stops, at least half the digits have been reversed

Each iteration removes one digit from `remaining` and adds one digit to `reversed_half`. The loop continues only while `remaining > reversed_half`. Once it stops, `reversed_half` has reached the same digit-length as `remaining` for even-length inputs, or one extra digit for odd-length inputs.

Thus the algorithm has reversed exactly the right-side half, possibly including the middle digit.

#### Lemma 3: The final comparison is exactly the palindrome condition

If the number has an even number of digits, there is no middle digit. By Lemma 1 and Lemma 2, `remaining` is the left half and `reversed_half` is the reversed right half. The number is a palindrome exactly when:

```text
remaining == reversed_half
```

If the number has an odd number of digits, `reversed_half` contains the middle digit as its last-added extra digit. The middle digit does not affect whether the outer digits mirror each other. Removing it with `reversed_half // 10` leaves the reversed right half. The number is a palindrome exactly when:

```text
remaining == reversed_half // 10
```

#### Theorem: The algorithm is correct

The early checks correctly reject all negative numbers and all positive numbers ending in zero, both of which cannot be palindromes. For every remaining non-negative input, the loop invariant shows that the algorithm separates the number into the unreversed left part and reversed right part. The final comparison covers both even- and odd-length numbers. Therefore the algorithm returns `True` if and only if `x` is a palindrome.

---

### 14. Complexity

Let `d` be the number of decimal digits in `x`.

The loop processes only half of the digits, so it runs `d / 2` iterations, which is still:

```text
O(d)
```

Since `d = floor(log10(x)) + 1` for positive `x`, this can also be written as:

```text
O(log x)
```

The algorithm uses only a constant number of integer variables:

```text
O(1)
```

So:

- Time: `O(log x)` for positive `x`, equivalently `O(d)` digits.
- Space: `O(1)` auxiliary space.

---

### 15. Common Pitfalls

- Treating negative numbers as palindromes after ignoring the minus sign. In this problem, `-121` is not a palindrome.
- Forgetting the trailing-zero case. `10` should return `False`, while `0` should return `True`.
- Reversing the entire number in a fixed-width language without considering overflow.
- Using `/` instead of integer division. Digit removal must use `//` in Python.
- Stopping the loop too late. Reversing only half avoids overflow and makes the odd/even comparison precise.
- Forgetting to drop the middle digit for odd-length numbers. That is why the final condition includes `reversed_half // 10`.

---

### 16. First-Principles Summary

A palindrome is mirror symmetry.

For a decimal integer, mirror symmetry means:

```text
left half == reverse(right half)
```

The operation `% 10` reveals the next digit from the right, and `// 10` removes it. By repeatedly moving right-side digits into `reversed_half`, we build the reverse of the right half while shrinking the remaining left side.

The central invariant is:

```text
remaining contains the still-unprocessed left prefix,
reversed_half contains the reversed processed right suffix.
```

When `reversed_half` catches up to `remaining`, the number has been split around its middle. Equal halves mean an even-length palindrome; equal after dropping the middle digit means an odd-length palindrome.

That is the entire idea: do not compare every digit explicitly, and do not reverse the whole number. Reverse only enough digits to expose the mirror relationship.

## Implementation
See `solutions/math/p009_palindrome_number.py`.

## Tests
See `tests/math/test_p009_palindrome_number.py`.

## Examples

### Example 1
- Input: `{'x': 121}`
- Output: `True`

### Example 2
- Input: `{'x': -121}`
- Output: `False`

### Example 3
- Input: `{'x': 10}`
- Output: `False`

## Follow-up Practice
- Trace `1221`, `12321`, and `1234321` by hand until the loop stops.
- Explain why `x != 0 and x % 10 == 0` is an immediate rejection.
- State the invariant before coding: what does `remaining` mean, and what does `reversed_half` mean?
- Compare the half-reversal method with the full-reversal method and identify where overflow could matter in fixed-width languages.
