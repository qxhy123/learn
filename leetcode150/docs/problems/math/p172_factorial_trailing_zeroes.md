# 172. Factorial Trailing Zeroes

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/factorial-trailing-zeroes/
- Official Group: Math
- Pattern Group: Math
- Patterns: math

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given a non-negative integer `n`, compute:

```text
n! = n * (n - 1) * (n - 2) * ... * 2 * 1
```

Return how many zero digits appear at the very end of the decimal representation of `n!`.

For example:

```text
5! = 5 * 4 * 3 * 2 * 1 = 120
```

The number `120` ends with exactly one zero, so the answer is:

```text
1
```

For `3`:

```text
3! = 3 * 2 * 1 = 6
```

The number `6` has no trailing zeroes, so the answer is:

```text
0
```

The important word is **trailing**. We do not care about zeroes in the middle of the number. We only care how many times the final decimal number is divisible by `10`.

So the problem can be restated as:

> How many factors of `10` are contained in `n!`?

---

### 2. Start From the Brute Force Idea

The most direct approach is to compute `n!`, then count how many times the result ends in digit `0`.

Conceptually:

```python
factorial = 1

for value in range(1, n + 1):
    factorial *= value

zeroes = 0
while factorial > 0 and factorial % 10 == 0:
    zeroes += 1
    factorial //= 10

return zeroes
```

This is easy to understand and gives the right answer for small `n`.

For example, for `n = 10`:

```text
10! = 3628800
```

The final two digits are zeroes, so the answer is `2`.

But this approach has a serious problem: factorials grow extremely fast.

```text
10!  = 3,628,800
20!  = 2,432,902,008,176,640,000
100! = a 158-digit number
```

Even if a language has arbitrary-size integers, multiplying a huge factorial only to inspect its final zeroes is wasteful. The final zeroes are not determined by the whole number. They are determined by a much smaller piece of information: how many factors of `10` appear in the product.

---

### 3. The Key Observation: A Trailing Zero Comes From `10`

In base 10, adding one trailing zero means the number is divisible by `10`.

```text
120 has 1 trailing zero  because 120 = 12 * 10
7000 has 3 trailing zeroes because 7000 = 7 * 10 * 10 * 10
```

So every trailing zero in `n!` corresponds to one factor of:

```text
10 = 2 * 5
```

That means we need to count how many pairs of prime factors `(2, 5)` appear inside the full product:

```text
n! = 1 * 2 * 3 * 4 * ... * n
```

Each pair creates one `10`, and each `10` creates one trailing zero.

At first, it may seem like we should count both `2`s and `5`s. But factorials contain far more factors of `2` than factors of `5`.

For example, in `10!`:

```text
2 contributes a factor 2
4 contributes two factors 2 * 2
6 contributes a factor 2
8 contributes three factors 2 * 2 * 2
10 contributes a factor 2 and a factor 5
```

Factors of `2` appear in every even number. Factors of `5` appear only in multiples of `5`.

Therefore, the limiting factor is always the number of `5`s.

So the problem becomes:

> Count how many times prime factor `5` appears in the factorization of all numbers from `1` to `n`.

---

### 4. The Factor-Count Invariant

The algorithm maintains this invariant:

```text
answer = number of factors of 5 counted so far from n!
```

We do not build `n!`.

Instead, we count contributions from groups of numbers:

```text
multiples of 5   contribute at least one factor of 5
multiples of 25  contribute one extra factor of 5
multiples of 125 contribute one more extra factor of 5
...
```

Why do we need multiple groups?

Because some numbers contain more than one factor of `5`.

For example:

```text
5  = 5             contributes 1 factor of 5
10 = 2 * 5         contributes 1 factor of 5
15 = 3 * 5         contributes 1 factor of 5
20 = 4 * 5         contributes 1 factor of 5
25 = 5 * 5         contributes 2 factors of 5
50 = 2 * 5 * 5     contributes 2 factors of 5
125 = 5 * 5 * 5    contributes 3 factors of 5
```

If we only count multiples of `5`, then `25` would be counted once, but it should contribute two `5`s. So we count:

```text
floor(n / 5)   numbers that contribute at least one 5
floor(n / 25)  numbers that contribute a second 5
floor(n / 125) numbers that contribute a third 5
...
```

The total is:

```text
floor(n / 5) + floor(n / 25) + floor(n / 125) + ...
```

We stop when the divisor exceeds `n`, because then there are no multiples left.

---

### 5. Why This Counts Exactly the Right Thing

Consider any number `x` in `1..n`.

Suppose `x` contains exactly `k` factors of `5`.

That means:

```text
x is divisible by 5^1
x is divisible by 5^2
...
x is divisible by 5^k
```

but not by `5^(k + 1)`.

In the summation:

```text
floor(n / 5) + floor(n / 25) + floor(n / 125) + ...
```

that number `x` is counted once in each group it belongs to:

```text
one count for divisibility by 5
one count for divisibility by 25
one count for divisibility by 125
...
```

So a number with exactly `k` factors of `5` is counted exactly `k` times.

That is precisely what we need, because every individual factor `5` can pair with a factor `2` to form one trailing zero.

This is the core invariant:

```text
After processing divisor = 5^i,
answer equals the number of factors of 5 contributed by all powers 5^1 through 5^i.
```

When all powers of `5` up to `n` have been processed, `answer` equals the total number of factors of `5` in `n!`, and therefore equals the number of trailing zeroes.

---

### 6. Detailed Algorithm

Use a divisor that starts at `5`.

At each step:

1. Count how many numbers from `1` to `n` are divisible by the current divisor.
2. Add that count to the answer.
3. Multiply the divisor by `5` to count the next extra layer of `5`s.
4. Stop when the divisor is greater than `n`.

In pseudocode:

```text
answer = 0
divisor = 5

while divisor <= n:
    answer += n // divisor
    divisor *= 5

return answer
```

The expression:

```text
n // divisor
```

counts how many multiples of `divisor` are at most `n`.

For example, when `divisor = 25` and `n = 100`:

```text
100 // 25 = 4
```

Those four numbers are:

```text
25, 50, 75, 100
```

Each of them contributes one extra factor of `5` beyond the factor already counted in the `divisor = 5` pass.

---

### 7. Example Walkthrough: `n = 3`

Compute:

```text
3! = 6
```

There are no multiples of `5` from `1` to `3`.

Algorithm state:

```text
answer = 0
divisor = 5
```

Check the loop condition:

```text
5 <= 3 is false
```

So the algorithm returns:

```text
0
```

This matches the fact that `6` has no trailing zeroes.

---

### 8. Example Walkthrough: `n = 5`

Compute:

```text
5! = 120
```

Trailing zeroes come from factors of `10 = 2 * 5`.

Algorithm state:

```text
answer = 0
divisor = 5
```

First pass:

```text
n // divisor = 5 // 5 = 1
answer = 0 + 1 = 1
divisor = 5 * 5 = 25
```

Now:

```text
25 <= 5 is false
```

Return:

```text
1
```

The single contributing factor of `5` comes from the number `5` itself. There are enough `2`s from the rest of the factorial, so that one `5` creates one trailing zero.

---

### 9. Larger Walkthrough: `n = 100`

Instead of computing `100!`, count factors of `5`.

Start:

```text
answer = 0
divisor = 5
```

Count multiples of `5`:

```text
100 // 5 = 20
```

These are:

```text
5, 10, 15, ..., 100
```

Each contributes at least one factor of `5`.

Now:

```text
answer = 20
divisor = 25
```

Count multiples of `25`:

```text
100 // 25 = 4
```

These are:

```text
25, 50, 75, 100
```

Each contributes a second factor of `5`, so add `4`.

Now:

```text
answer = 24
divisor = 125
```

Since:

```text
125 > 100
```

stop and return:

```text
24
```

So `100!` has `24` trailing zeroes.

Notice how the number `25` is counted twice overall:

```text
once as a multiple of 5
once as a multiple of 25
```

That is correct because:

```text
25 = 5 * 5
```

---

### 10. Code

Python implementation:

```python
class Solution:
    def trailingZeroes(self, n: int) -> int:
        zeroes = 0
        divisor = 5

        while divisor <= n:
            zeroes += n // divisor
            divisor *= 5

        return zeroes
```

A slightly different but equivalent version repeatedly divides `n` by `5`:

```python
class Solution:
    def trailingZeroes(self, n: int) -> int:
        zeroes = 0

        while n > 0:
            n //= 5
            zeroes += n

        return zeroes
```

This works because:

```text
first division  gives floor(original_n / 5)
second division gives floor(original_n / 25)
third division  gives floor(original_n / 125)
...
```

Both versions implement the same mathematical formula. The explicit `divisor` version mirrors the reasoning more directly.

---

### 11. Correctness

We prove that the algorithm returns the number of trailing zeroes in `n!`.

A trailing zero in a decimal integer is created by a factor of `10`.

Since:

```text
10 = 2 * 5
```

counting trailing zeroes is equivalent to counting how many pairs of factors `(2, 5)` appear in the prime factorization of `n!`.

In `n!`, there are at least as many factors of `2` as factors of `5`, because every multiple of `5` can be paired with an even number, and even numbers occur more frequently than multiples of `5`. Therefore, the number of `(2, 5)` pairs is exactly the number of factors of `5` in `n!`.

The algorithm adds:

```text
floor(n / 5^i)
```

for every positive power `5^i <= n`.

For a fixed number `x` between `1` and `n`, if `x` contains exactly `k` factors of `5`, then `x` is divisible by:

```text
5^1, 5^2, ..., 5^k
```

and is not divisible by `5^(k + 1)`.

Thus, across the algorithm's summation, `x` contributes exactly `k` total counts. This equals the exact number of factors of `5` in `x`.

Summing over every `x` from `1` to `n`, the algorithm counts exactly the total number of factors of `5` in `n!`.

Since that total equals the number of trailing zeroes, the algorithm returns the correct answer.

---

### 12. Complexity

The divisor sequence is:

```text
5, 25, 125, 625, ...
```

Each step multiplies the divisor by `5`, so the number of loop iterations is logarithmic in `n`.

- Time: `O(log_5 n)`, usually written as `O(log n)`.
- Space: `O(1)`.

The algorithm never computes `n!`, so it avoids huge intermediate numbers.

---

### 13. Common Pitfalls

- **Computing the factorial directly.** This is unnecessary and becomes huge very quickly.
- **Counting only multiples of `5`.** This misses extra factors from numbers like `25`, `50`, `75`, `100`, and `125`.
- **Forgetting that `25` contributes two `5`s.** One is counted by `n // 5`; the extra one is counted by `n // 25`.
- **Trying to count factors of `10` directly.** It is simpler to count factors of `5`, because factors of `2` are never the limiting resource in a factorial.
- **Using floating-point logarithms or powers.** Integer division and integer multiplication are exact and safer.
- **Mishandling `n = 0`.** By definition, `0! = 1`, which has no trailing zeroes, and the loop naturally returns `0`.

---

### 14. First-Principles Summary

The problem looks like it asks about the decimal digits of a huge factorial, but the digits are a distraction.

A trailing zero means a factor of `10`.

A factor of `10` means one `2` paired with one `5`.

In a factorial, factors of `2` are abundant, so each factor of `5` determines one trailing zero.

So instead of computing:

```text
1 * 2 * 3 * ... * n
```

we count:

```text
how many 5s appear in that product
```

The count is:

```text
floor(n / 5) + floor(n / 25) + floor(n / 125) + ...
```

That formula is the whole algorithm.

## Implementation
See `solutions/math/p172_factorial_trailing_zeroes.py`.

## Tests
See `tests/math/test_p172_factorial_trailing_zeroes.py`.

## Examples

### Example 1
- Input: `{'n': 3}`
- Output: `0`

### Example 2
- Input: `{'n': 5}`
- Output: `1`

### Example 3
- Input: `{'n': 0}`
- Output: `0`

## Follow-up Practice
- Explain why counting factors of `5` is enough.
- Manually compute the answer for `n = 25`, `n = 50`, and `n = 100`.
- Identify which numbers contribute extra factors of `5` beyond the first one.
