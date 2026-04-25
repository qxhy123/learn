# 69. Sqrt(x)

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/sqrtx/
- Official Group: Math
- Pattern Group: Math
- Patterns: math

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given a non-negative integer `x`, return the integer part of its square root.

In mathematical notation, we want:

```text
floor(sqrt(x))
```

That means the answer is the largest integer `a` such that:

```text
a * a <= x
```

For example:

```text
x = 8
sqrt(8) is about 2.828...
floor(sqrt(8)) = 2
```

So the problem is not asking for a decimal square root. It is asking for the biggest whole number whose square does not exceed `x`.

Another way to state the target is:

```text
Find the largest integer a where a^2 <= x.
```

This reformulation is the entire problem. Once we avoid floating point and focus on the inequality `a * a <= x`, the solution becomes an integer search problem.

### 2. Start From the Brute Force Baseline

The most direct method is to try every possible answer from small to large:

```python
answer = 0
candidate = 0

while candidate * candidate <= x:
    answer = candidate
    candidate += 1

return answer
```

This works because candidates are checked in increasing order. The last candidate that satisfies:

```text
candidate * candidate <= x
```

is exactly the integer square root.

For `x = 8`, the brute-force trace is:

```text
0 * 0 = 0 <= 8   valid
1 * 1 = 1 <= 8   valid
2 * 2 = 4 <= 8   valid
3 * 3 = 9 > 8    too large
```

The last valid candidate is `2`.

The problem is speed. If `x` is very large, the loop can take about `sqrt(x)` iterations. For example, if `x` is near `2^31 - 1`, the answer is around `46340`, so brute force still does tens of thousands of checks. That may pass in Python for this input size, but it does not use the most important structure of the problem.

The repeated work is obvious:

```text
If 100^2 is too large, then 101^2, 102^2, 103^2, ... are all too large too.
```

The candidates are ordered. We should search that order instead of scanning every value.

### 3. The Key Observation

The predicate:

```text
candidate * candidate <= x
```

changes from true to false only once.

For a fixed `x`, imagine writing the result for every candidate:

```text
candidate:  0   1   2   3   4   5   6   ...
valid?:     T   T   T   F   F   F   F   ...
```

Using `x = 8`:

```text
0^2 <= 8  true
1^2 <= 8  true
2^2 <= 8  true
3^2 <= 8  false
4^2 <= 8  false
```

Why does the truth value never become true again?

Because squaring non-negative integers is monotonic:

```text
if a < b, then a * a < b * b
```

So once a candidate square is larger than `x`, every larger candidate square is also larger than `x`.

That gives the exact shape binary search needs:

```text
true true true true false false false
```

The answer is the last `true` position.

### 4. Search Space Boundaries

The answer is always between:

```text
0 and x
```

because:

```text
0^2 <= x
```

and for `x >= 1`:

```text
sqrt(x) <= x
```

So a simple binary search can start with:

```text
left = 0
right = x
```

There is also a tighter upper bound for larger inputs:

```text
right = x // 2
```

for `x >= 2`, because the square root of any integer at least `2` is at most half of that integer. But using `right = x` is simpler and still `O(log x)`. The extra range only adds a small constant number of binary-search steps.

The boundary cases are important:

```text
x = 0  -> 0
x = 1  -> 1
```

They also work naturally with `left = 0` and `right = x`.

### 5. Binary-Search Invariant

We want the largest valid candidate.

During binary search, keep these meanings:

```text
answer = best valid candidate found so far
left..right = candidates not yet ruled out
```

The invariant is:

```text
answer * answer <= x
```

whenever `answer` has been updated.

At each step, choose the middle candidate:

```text
mid = (left + right) // 2
```

Then compare `mid * mid` with `x`.

There are three cases.

#### Case 1: Exact Square

```text
mid * mid == x
```

Then `mid` is the square root exactly, so return `mid` immediately.

Example:

```text
x = 4
mid = 2
2 * 2 = 4
return 2
```

#### Case 2: Candidate Is Too Small

```text
mid * mid < x
```

Then `mid` is valid, but it may not be the largest valid number. Larger candidates might still work.

So we record it:

```text
answer = mid
```

and search to the right:

```text
left = mid + 1
```

This preserves the meaning of `answer`: it is still the best valid candidate seen so far.

#### Case 3: Candidate Is Too Large

```text
mid * mid > x
```

Then `mid` is invalid. Because squares increase as candidates increase, every candidate larger than `mid` is also invalid.

So we discard `mid` and everything above it:

```text
right = mid - 1
```

We do not change `answer`, because `mid` is not valid.

### 6. Detailed Algorithm

The algorithm is:

1. Initialize the candidate range from `0` to `x`.
2. Initialize `answer = 0`.
3. While the range is not empty:
   - Pick the middle candidate.
   - Compute its square.
   - If the square equals `x`, return the middle candidate.
   - If the square is less than `x`, store the middle candidate as the best valid answer so far and search larger values.
   - If the square is greater than `x`, search smaller values.
4. When the range is empty, return the best valid candidate recorded in `answer`.

The reason the final `answer` is safe is that it is only updated when a candidate is known to satisfy:

```text
candidate * candidate <= x
```

So it never becomes too large.

The reason it is maximal is that binary search only discards values after proving they cannot improve the answer.

### 7. Walkthrough: `x = 8`

Start:

```text
left = 0
right = 8
answer = 0
```

#### Step 1

```text
mid = (0 + 8) // 2 = 4
mid * mid = 16
```

`16 > 8`, so `4` is too large.

Every value above `4` is also too large, so move left:

```text
right = 3
answer = 0
```

#### Step 2

```text
left = 0
right = 3
mid = (0 + 3) // 2 = 1
mid * mid = 1
```

`1 <= 8`, so `1` is valid.

But it might not be the largest valid candidate, so record it and search right:

```text
answer = 1
left = 2
```

#### Step 3

```text
left = 2
right = 3
mid = (2 + 3) // 2 = 2
mid * mid = 4
```

`4 <= 8`, so `2` is valid.

Record it and search right:

```text
answer = 2
left = 3
```

#### Step 4

```text
left = 3
right = 3
mid = 3
mid * mid = 9
```

`9 > 8`, so `3` is too large.

Move left:

```text
right = 2
```

Now:

```text
left = 3
right = 2
```

The search range is empty. Return:

```text
answer = 2
```

That matches `floor(sqrt(8))`.

### 8. Walkthrough: `x = 4`

Start:

```text
left = 0
right = 4
answer = 0
```

First middle:

```text
mid = 2
mid * mid = 4
```

This is an exact square, so return:

```text
2
```

The exact-square case is not required for correctness, because the normal binary-search logic would also eventually return `2`. But returning immediately is simple and avoids unnecessary work.

### 9. Python Code

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        left = 0
        right = x
        answer = 0

        while left <= right:
            mid = (left + right) // 2
            square = mid * mid

            if square == x:
                return mid

            if square < x:
                answer = mid
                left = mid + 1
            else:
                right = mid - 1

        return answer
```

### 10. Pseudocode

```text
function mySqrt(x):
    left = 0
    right = x
    answer = 0

    while left <= right:
        mid = floor((left + right) / 2)
        square = mid * mid

        if square == x:
            return mid

        if square < x:
            answer = mid
            left = mid + 1
        else:
            right = mid - 1

    return answer
```

### 11. Correctness

We prove that the algorithm returns `floor(sqrt(x))`.

Let the desired answer be:

```text
A = largest integer such that A * A <= x
```

#### The Algorithm Never Records an Invalid Answer

The variable `answer` is updated only in this case:

```text
mid * mid < x
```

So every recorded value satisfies:

```text
answer * answer <= x
```

Therefore, the algorithm never returns a value whose square is greater than `x`.

#### Discarding the Right Half Is Safe When `mid` Is Too Large

If:

```text
mid * mid > x
```

then `mid` is not a valid answer.

For every candidate `c > mid`, because `c` is larger and non-negative:

```text
c * c > mid * mid > x
```

So no candidate from `mid` through the old `right` can be valid. Setting:

```text
right = mid - 1
```

cannot discard the true answer.

#### Discarding the Left Half Is Safe When `mid` Is Valid

If:

```text
mid * mid < x
```

then `mid` is a valid answer candidate, so storing it is safe.

Every candidate smaller than `mid` cannot be better than `mid`, because the problem asks for the largest valid integer. Therefore, after recording `mid`, it is safe to search only larger candidates by setting:

```text
left = mid + 1
```

This does not lose the best answer: either the true answer is `mid`, already stored in `answer`, or it is larger than `mid` and remains in the search range.

#### Exact Squares Are Returned Correctly

If:

```text
mid * mid == x
```

then `mid` is exactly `sqrt(x)`. Since `mid` is an integer, it is also `floor(sqrt(x))`, so returning `mid` is correct.

#### Termination Gives the Largest Valid Candidate

The loop ends when:

```text
left > right
```

At that point, every candidate has either been examined directly or ruled out by one of the safe discard arguments above.

`answer` is valid, and any discarded larger value was discarded only after being proven too large or after the search had moved beyond a valid value while keeping it recorded.

So `answer` is the largest integer whose square is at most `x`.

Therefore, the algorithm returns `floor(sqrt(x))`.

### 12. Complexity

Each iteration cuts the remaining search interval roughly in half.

The initial interval has at most `x + 1` candidates:

```text
0, 1, 2, ..., x
```

So the number of iterations is logarithmic in `x`.

Complexity:

```text
Time:  O(log x)
Space: O(1)
```

The algorithm stores only a few integers: `left`, `right`, `mid`, `square`, and `answer`.

### 13. Common Pitfalls

#### Using Floating Point

A tempting solution is:

```python
return int(x ** 0.5)
```

For this LeetCode problem it may often appear to work, but it avoids the core integer reasoning and can be risky in fixed-precision environments. Floating-point square roots can introduce rounding issues for very large integers.

The integer binary-search solution avoids that entirely.

#### Returning the First Valid Candidate

If `mid * mid < x`, `mid` is valid, but it may not be the answer.

For `x = 15`:

```text
2^2 = 4 <= 15
3^2 = 9 <= 15
4^2 = 16 > 15
```

The answer is `3`, not the first valid number found. That is why the algorithm stores `answer = mid` and continues searching to the right.

#### Forgetting to Save the Last Valid Candidate

If the loop only moves pointers and never stores the last valid `mid`, it may lose the floor value when the final checked candidate is too large.

For `x = 8`, the final valid candidate is `2`, but the algorithm also checks `3` and finds it too large. Returning `mid` at the end would be wrong unless `mid` still happens to be `2`.

Use an explicit `answer` variable or a carefully designed boundary convention.

#### Infinite Loops From Pointer Updates

After checking `mid`, always exclude it from the next interval unless returning immediately:

```text
left = mid + 1
right = mid - 1
```

Using `left = mid` or `right = mid` can get stuck when the interval has one or two values.

#### Overflow in Fixed-Width Languages

In Python, `mid * mid` is safe because integers can grow as needed.

In languages with fixed-width integers, `mid * mid` can overflow. A common alternative is to compare using division:

```text
mid <= x / mid
```

instead of computing `mid * mid` directly.

Be careful with `mid = 0` if using division.

#### Mishandling `0` and `1`

The binary-search version with:

```text
left = 0
right = x
answer = 0
```

handles both naturally:

```text
mySqrt(0) = 0
mySqrt(1) = 1
```

If using a tighter range like `left = 1`, handle `x = 0` separately.

### 14. First-Principles Summary

This problem follows from a small set of facts:

```text
1. The integer square root is the largest integer a such that a * a <= x.
2. For non-negative integers, larger candidates have larger squares.
3. Therefore, the predicate a * a <= x has the shape true...true false...false.
4. Binary search can find the boundary between true and false.
5. Since the answer is the last true value, store each valid midpoint before searching larger candidates.
```

So the whole algorithm is:

> Search the ordered candidate values, discard impossible halves using the monotonic square function, and keep the largest candidate seen whose square does not exceed `x`.

## Implementation

See `solutions/math/p069_sqrtx.py`.

## Tests

See `tests/math/test_p069_sqrtx.py`.

## Examples

### Example 1
- Input: `{'x': 4}`
- Output: `2`

### Example 2
- Input: `{'x': 8}`
- Output: `2`

## Follow-up Practice
- Trace the binary-search invariant on `x = 0`, `x = 1`, `x = 8`, and `x = 15`.
- Implement the same idea with `right = x // 2` for `x >= 2`.
- In a fixed-width language, rewrite the comparison to avoid `mid * mid` overflow.
