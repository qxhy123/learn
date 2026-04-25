# 50. Pow(x, n)

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/powx-n/
- Official Group: Math
- Pattern Group: Math
- Patterns: math

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given a floating-point number `x` and an integer `n`, compute:

```text
x^n
```

That means multiplying `x` by itself `n` times when `n` is positive:

```text
x^5 = x * x * x * x * x
```

The exponent can also be zero or negative:

```text
x^0  = 1
x^-n = 1 / x^n
```

So the problem is not asking for a new mathematical definition of power. It is asking us to implement the usual power function efficiently and carefully.

The important constraint is that `n` can be very large in magnitude. A solution that literally performs one multiplication per exponent unit may do far too much work.

The problem becomes:

> How can we compute `x^n` without multiplying by `x` one step at a time?

### 2. Start From the Brute-Force Baseline

The most direct method is repeated multiplication.

For a positive exponent:

```python
answer = 1.0
for _ in range(n):
    answer *= x
return answer
```

For example:

```text
x = 2, n = 10

answer = 1
answer = 2
answer = 4
answer = 8
answer = 16
answer = 32
answer = 64
answer = 128
answer = 256
answer = 512
answer = 1024
```

This is correct because it exactly follows the definition of exponentiation.

But it costs:

```text
O(n) multiplications
```

If `n = 1,000,000,000`, that is one billion loop iterations. The mathematical result has structure, but the brute-force algorithm ignores it.

For negative `n`, brute force usually computes the positive power and then takes a reciprocal:

```python
answer = 1.0
for _ in range(abs(n)):
    answer *= x

if n < 0:
    answer = 1.0 / answer
```

That still costs `O(abs(n))`, so the fundamental problem remains.

### 3. The Key Observation: Powers Can Be Reused

The brute-force method treats every multiplication as equally necessary:

```text
x^10 = x * x * x * x * x * x * x * x * x * x
```

But powers have a stronger structure.

If we know `x^5`, then:

```text
x^10 = x^5 * x^5
```

If we know `x^4`, then:

```text
x^8 = x^4 * x^4
```

Squaring lets one multiplication double the exponent represented by a value.

That is much more powerful than multiplying by one extra `x` each time.

The core identities are:

```text
x^n = (x^(n/2))^2        when n is even
x^n = x * x^(n - 1)      when n is odd
```

For an odd exponent, we peel off one `x`, making the remaining exponent even:

```text
x^9 = x * x^8
```

For an even exponent, we split it in half:

```text
x^8 = (x^4)^2
```

This is the entire idea behind exponentiation by squaring.

### 4. Why Binary Exponentiation Fits Naturally

Another way to see the same idea is through the binary representation of `n`.

For example:

```text
10 in binary is 1010
```

That means:

```text
10 = 8 + 2
```

So:

```text
x^10 = x^8 * x^2
```

Instead of building every power from `x^1` through `x^10`, we only need powers whose exponents are powers of two:

```text
x^1, x^2, x^4, x^8, x^16, ...
```

Each one is obtained by squaring the previous one:

```text
x^2 = x^1 * x^1
x^4 = x^2 * x^2
x^8 = x^4 * x^4
```

Then we multiply into the answer only when the corresponding binary bit of the exponent is `1`.

For `n = 10`:

```text
10 = 1010₂

use x^2
use x^8
skip x^1
skip x^4
```

So:

```text
x^10 = x^8 * x^2
```

This reduces the number of steps from proportional to `n` to proportional to the number of bits in `n`.

### 5. The Exponentiation-by-Squaring Invariant

We maintain three values:

```text
result
base
exponent
```

Initially, after handling the sign of `n`:

```text
result = 1
base = x
exponent = abs(n)
```

The invariant is:

```text
result * base^exponent == x^abs(original_n)
```

This statement is the reason the algorithm is correct.

At the start:

```text
result * base^exponent
= 1 * x^abs(n)
= x^abs(n)
```

So the invariant holds.

Now look at one loop step.

#### If `exponent` is odd

An odd exponent contains one unpaired copy of `base`:

```text
base^exponent = base * base^(exponent - 1)
```

So we move that one copy into `result`:

```text
result = result * base
exponent = exponent - 1
```

The product stays the same:

```text
new_result * base^new_exponent
= (old_result * base) * base^(old_exponent - 1)
= old_result * base^old_exponent
```

The invariant is preserved.

#### Then we square the base and halve the exponent

Once the exponent is even, this identity applies:

```text
base^exponent = (base * base)^(exponent / 2)
```

So we update:

```text
base = base * base
exponent = exponent // 2
```

Again, the product stays the same:

```text
result * new_base^new_exponent
= result * (old_base * old_base)^(old_exponent / 2)
= result * old_base^old_exponent
```

The invariant is preserved.

When `exponent` becomes `0`:

```text
result * base^0 = result * 1 = result
```

By the invariant, `result` equals the desired positive power.

If the original exponent was negative, the final answer is the reciprocal:

```text
1 / result
```

### 6. Detailed Algorithm

The algorithm has two phases: normalize the exponent, then process its binary bits.

1. If `n` is negative, remember that the final answer must be inverted.
2. Work with `exponent = abs(n)` so the loop only handles a nonnegative integer.
3. Set `result = 1.0` because multiplying by `1` does not change the answer.
4. Set `base = x` because the first power available is `x^1`.
5. While `exponent > 0`:
   - If `exponent` is odd, multiply `result` by `base`.
   - Square `base` so it represents the next power of two.
   - Divide `exponent` by `2` using integer division.
6. If the original `n` was negative, return `1.0 / result`; otherwise return `result`.

The loop reads the exponent from least significant binary bit to most significant binary bit.

For each bit:

```text
current bit is 1 -> include this base power in result
current bit is 0 -> skip this base power
```

After every iteration, `base` moves from:

```text
x^1 -> x^2 -> x^4 -> x^8 -> ...
```

and `exponent` shifts right by one binary bit.

### 7. Walkthrough: `x = 2.0`, `n = 10`

We want:

```text
2^10 = 1024
```

Initialize:

```text
result = 1
base = 2
exponent = 10
```

Invariant:

```text
result * base^exponent = 1 * 2^10
```

#### Iteration 1

```text
exponent = 10, even
```

The current binary bit is `0`, so we do not multiply `result`.

Square the base and halve the exponent:

```text
base = 2 * 2 = 4
exponent = 10 // 2 = 5
result = 1
```

Now the invariant says:

```text
1 * 4^5 = 2^10
```

#### Iteration 2

```text
exponent = 5, odd
```

The current binary bit is `1`, so include the current base:

```text
result = 1 * 4 = 4
```

Then square the base and halve the exponent:

```text
base = 4 * 4 = 16
exponent = 5 // 2 = 2
```

Now:

```text
result * base^exponent = 4 * 16^2 = 4 * 256 = 1024
```

#### Iteration 3

```text
exponent = 2, even
```

Skip multiplying `result`.

```text
base = 16 * 16 = 256
exponent = 2 // 2 = 1
result = 4
```

Now:

```text
4 * 256^1 = 1024
```

#### Iteration 4

```text
exponent = 1, odd
```

Include the current base:

```text
result = 4 * 256 = 1024
```

Then update:

```text
base = 256 * 256
exponent = 1 // 2 = 0
```

The loop stops.

Since the original exponent was positive, return:

```text
1024.0
```

### 8. Walkthrough: `x = 2.0`, `n = -2`

A negative exponent means reciprocal:

```text
2^-2 = 1 / 2^2
```

First compute the positive power using `exponent = 2`.

Initialize:

```text
result = 1
base = 2
exponent = 2
```

#### Iteration 1

```text
exponent = 2, even
```

Skip multiplying `result`.

```text
base = 2 * 2 = 4
exponent = 1
result = 1
```

#### Iteration 2

```text
exponent = 1, odd
```

Include the current base:

```text
result = 1 * 4 = 4
```

Then:

```text
base = 4 * 4 = 16
exponent = 0
```

The positive power is:

```text
2^2 = 4
```

Because the original exponent was negative, return:

```text
1 / 4 = 0.25
```

### 9. Code

Python implementation:

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1.0

        negative_exponent = n < 0
        exponent = -n if negative_exponent else n

        result = 1.0
        base = x

        while exponent > 0:
            if exponent % 2 == 1:
                result *= base

            base *= base
            exponent //= 2

        if negative_exponent:
            return 1.0 / result

        return result
```

Equivalent pseudocode:

```text
function pow(x, n):
    negative = n < 0
    exponent = abs(n)

    result = 1
    base = x

    while exponent > 0:
        if exponent is odd:
            result = result * base

        base = base * base
        exponent = floor(exponent / 2)

    if negative:
        return 1 / result
    return result
```

### 10. Correctness

We prove that the algorithm returns `x^n`.

Let `original_n` be the input exponent.

After sign handling, the loop computes `x^abs(original_n)`. If `original_n` is negative, the algorithm returns the reciprocal afterward.

The loop maintains this invariant:

```text
result * base^exponent = x^abs(original_n)
```

#### Initialization

Before the loop:

```text
result = 1
base = x
exponent = abs(original_n)
```

Therefore:

```text
result * base^exponent
= 1 * x^abs(original_n)
= x^abs(original_n)
```

So the invariant holds initially.

#### Maintenance

During each loop iteration, there are two cases.

If `exponent` is odd, the algorithm multiplies `result` by `base`. This moves one factor of `base` from `base^exponent` into `result`, preserving the total product represented by the invariant.

Then the algorithm squares `base` and halves `exponent`. This is valid because:

```text
base^exponent = (base^2)^(exponent / 2)
```

for the remaining even exponent contribution represented after the odd bit is accounted for.

Thus, after every iteration:

```text
result * base^exponent
```

still equals the original positive power.

#### Termination

The loop stops when:

```text
exponent = 0
```

By the invariant:

```text
result * base^0 = x^abs(original_n)
```

Since `base^0 = 1`:

```text
result = x^abs(original_n)
```

If `original_n >= 0`, this is exactly `x^original_n`.

If `original_n < 0`, the mathematical definition is:

```text
x^original_n = 1 / x^abs(original_n)
```

The algorithm returns `1 / result`, so it returns the correct value in that case as well.

Therefore, the algorithm returns the required power.

### 11. Complexity

Each loop iteration halves `exponent`.

So the number of iterations is the number of binary digits in `abs(n)`:

```text
O(log |n|)
```

Each iteration performs only a constant amount of work.

Time complexity:

```text
O(log |n|)
```

The algorithm uses only a few variables:

```text
result, base, exponent, negative_exponent
```

Space complexity:

```text
O(1)
```

### 12. Common Pitfalls

#### Pitfall 1: Multiplying `abs(n)` Times

A loop that runs once per exponent unit is too slow for large exponents.

The whole point of the problem is to reuse squared powers and reduce the exponent by half each step.

#### Pitfall 2: Forgetting Negative Exponents

For `n < 0`, the answer is not negative.

It is reciprocal:

```text
x^-2 = 1 / x^2
```

So `2^-2` is `0.25`, not `-4`.

#### Pitfall 3: Returning `0` for `n == 0`

Any nonzero base raised to the zero power is `1`:

```text
x^0 = 1
```

The implementation naturally handles this because the loop never runs and `result` starts at `1.0`.

#### Pitfall 4: Losing the Odd Exponent Bit

When `exponent` is odd, one copy of the current `base` must be multiplied into `result` before halving the exponent.

If you only square `base` and halve `exponent`, you silently discard that binary bit.

For example, with `n = 5`:

```text
5 = 101₂
```

The powers for bits `1` and `4` must both be included.

#### Pitfall 5: Integer Overflow in Other Languages

Python integers can grow as needed, but languages with fixed-width integers can have a special edge case:

```text
n = -2^31
```

In some languages, `abs(n)` overflows because `2^31` is outside the positive range of a signed 32-bit integer.

A common fix is to cast `n` to a wider integer type before negating it.

#### Pitfall 6: Floating-Point Precision Expectations

The answer is a floating-point value, so tiny rounding differences can occur.

For example, repeated multiplication with `2.1` may produce a value very close to `9.261`, not always a perfectly decimal representation internally.

That is normal for floating-point arithmetic.

### 13. First-Principles Summary

The brute-force definition of `x^n` says to multiply `x` by itself `n` times.

The first-principles improvement is to ask what information each multiplication gives us.

Multiplying by `x` increases the represented exponent by only `1`:

```text
x^k -> x^(k + 1)
```

But squaring doubles the represented exponent:

```text
x^k -> x^(2k)
```

That doubling is the source of the logarithmic runtime.

The algorithm keeps a precise invariant:

```text
result * base^exponent = original positive power
```

When the current exponent bit is `1`, it moves the current `base` into `result`. Then it squares `base` and shifts the exponent right by one bit.

So the algorithm is not a trick or a memorized pattern. It is just the binary representation of the exponent plus the algebraic identity:

```text
(x^a)^2 = x^(2a)
```

That is why exponentiation by squaring computes large powers quickly.

## Implementation
See `solutions/math/p050_powx_n.py`.

## Tests
See `tests/math/test_p050_powx_n.py`.

## Examples

### Example 1
- Input: `{'x': 2.0, 'n': 10}`
- Output: `1024.0`

### Example 2
- Input: `{'x': 2.1, 'n': 3}`
- Output: `9.261`

### Example 3
- Input: `{'x': 2.0, 'n': -2}`
- Output: `0.25`

## Follow-up Practice
- Trace the loop for `x = 3`, `n = 13`, and write the binary representation of `13`.
- Explain why an odd exponent forces one multiplication into `result` before halving.
- Check boundary cases such as `n = 0`, `n = 1`, `n = -1`, `x = 1`, and `x = -1`.
