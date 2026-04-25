# 190. Reverse Bits

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/reverse-bits/
- Official Group: Bit Manipulation
- Pattern Group: Bit Manipulation
- Patterns: bit-manipulation

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an integer `n` that represents a **32-bit unsigned binary number**.

The task is to return the integer whose 32 bits are the same bits, but in the opposite order.

If the input bits are:

```text
b31 b30 b29 ... b2 b1 b0
```

then the output bits must be:

```text
b0 b1 b2 ... b29 b30 b31
```

The leftmost bit becomes the rightmost bit. The rightmost bit becomes the leftmost bit.

This is not decimal reversal.

For example, the number `43261596` has the 32-bit binary form:

```text
00000010100101000001111010011100
```

After reversing all 32 bit positions, we get:

```text
00111001011110000010100101000000
```

That binary number is `964176192`.

So the problem is really asking:

> Move every bit from its original 32-bit position to its mirror position, then interpret the resulting 32-bit pattern as an unsigned integer.

The fixed width matters. We reverse **exactly 32 positions**, including leading zeroes.

### 2. Start From the Direct Bit-Mapping Definition

The mathematical definition is straightforward.

If bit `i` of `n` is `1`, then bit `31 - i` of the answer should be `1`.

For example:

```text
original position:  i
new position:       31 - i
```

So one baseline approach is:

1. Start with `answer = 0`.
2. For each bit position `i` from `0` to `31`:
   - Extract bit `i` from `n`.
   - Place it into position `31 - i` of `answer`.
3. Return `answer`.

Pseudocode:

```python
answer = 0

for i in range(32):
    bit = (n >> i) & 1
    answer |= bit << (31 - i)

return answer
```

This is already efficient because `32` is a constant. It is a useful first-principles baseline because it directly mirrors the definition of the problem.

But there is an even more natural way to build the reversed number: consume bits from right to left and append them to the answer from left to right.

### 3. The Key Observation

A binary number is a sequence of bits.

The operation:

```python
n & 1
```

extracts the current least significant bit, meaning the rightmost bit.

The operation:

```python
n >>= 1
```

removes that rightmost bit from consideration by shifting the remaining bits one position to the right.

So if we repeatedly do this:

```text
read rightmost bit of n
remove rightmost bit of n
```

we read the original bits in this order:

```text
b0, b1, b2, ..., b30, b31
```

That is exactly the order in which they should appear in the reversed answer from left to right.

Therefore the problem can be solved by treating `n` as a source of bits and `answer` as a growing destination:

```text
n gives us bits from right to left
answer receives bits from left to right
```

To append a new bit to the right side of `answer`, shift `answer` left by one and insert the bit:

```python
answer = (answer << 1) | bit
```

That single line means:

```text
make room for one more bit, then write the next reversed bit into the empty slot
```

### 4. The Bit-Position Invariant

The most important part of this problem is keeping the positions straight.

Let the original input bits be:

```text
b31 b30 ... b2 b1 b0
```

where `b0` is the least significant bit.

After `k` iterations, we have consumed the original bits:

```text
b0, b1, ..., b(k-1)
```

These are the first `k` bits that should appear in the reversed answer.

The invariant is:

```text
After k iterations, answer stores the k-bit sequence b0 b1 ... b(k-1).
```

In other words, after `k` iterations, `answer` is not yet the final 32-bit value, but its built prefix is correct.

The next iteration extracts `bk` from `n`. Shifting `answer` left creates one new empty low bit:

```text
answer:        b0 b1 ... b(k-1)
answer << 1:   b0 b1 ... b(k-1) 0
```

Then OR-ing with `bk` writes that extracted bit into the new final position:

```text
(answer << 1) | bk = b0 b1 ... b(k-1) bk
```

So the invariant remains true for `k + 1` iterations.

After exactly `32` iterations, the invariant says:

```text
answer stores b0 b1 ... b31
```

That is precisely the 32-bit reversal of the input.

### 5. Detailed Algorithm

Use two variables:

```text
answer: the reversed bits built so far
n:      the remaining input bits not yet consumed
```

Algorithm:

1. Initialize `answer = 0`.
2. Repeat exactly `32` times:
   - Shift `answer` left by one to make room for the next bit.
   - Extract the current rightmost bit of `n` using `n & 1`.
   - Append that bit to `answer` using OR.
   - Shift `n` right by one so the next original bit becomes the rightmost bit.
3. Return `answer`.

Python-style code:

```python
class Solution:
    def reverseBits(self, n: int) -> int:
        answer = 0

        for _ in range(32):
            answer = (answer << 1) | (n & 1)
            n >>= 1

        return answer
```

The order of operations inside the loop is important:

```python
answer = (answer << 1) | (n & 1)
```

means:

```text
1. move existing reversed prefix left
2. copy current input bit into the new rightmost position
```

Then:

```python
n >>= 1
```

means:

```text
discard the input bit we just used
```

### 6. Detailed Example Walkthrough

Use a small width first, because the same reasoning applies to 32 bits.

Suppose the input is an 8-bit value:

```text
n = 00010110
```

The bits from left to right are:

```text
0 0 0 1 0 1 1 0
```

The reversed result should be:

```text
0 1 1 0 1 0 0 0
```

Now trace the append-from-right algorithm.

Initial state:

```text
answer = 0
n      = 00010110
```

Iteration 1:

```text
n & 1 = 0
answer = (0 << 1) | 0 = 0
n becomes 00001011
```

Built reversed prefix:

```text
0
```

Iteration 2:

```text
n & 1 = 1
answer = (0 << 1) | 1 = 1
n becomes 00000101
```

Built reversed prefix:

```text
01
```

Iteration 3:

```text
n & 1 = 1
answer = (01 << 1) | 1 = 011
n becomes 00000010
```

Built reversed prefix:

```text
011
```

Iteration 4:

```text
n & 1 = 0
answer = (011 << 1) | 0 = 0110
n becomes 00000001
```

Built reversed prefix:

```text
0110
```

Iteration 5:

```text
n & 1 = 1
answer = (0110 << 1) | 1 = 01101
n becomes 00000000
```

Built reversed prefix:

```text
01101
```

The remaining original high bits are zeroes. We still must continue until the fixed width is complete.

Iteration 6:

```text
n & 1 = 0
answer becomes 011010
```

Iteration 7:

```text
n & 1 = 0
answer becomes 0110100
```

Iteration 8:

```text
n & 1 = 0
answer becomes 01101000
```

Final 8-bit reversed result:

```text
01101000
```

For the real problem, the same loop runs `32` times instead of `8` times.

Official example:

```text
input:  00000010100101000001111010011100
output: 00111001011110000010100101000000
```

Decimal form:

```text
input:  43261596
output: 964176192
```

The reason leading zeroes appear in the output calculation is that the input is interpreted as a full 32-bit value, not as a shortened binary string.

### 7. Why This Is Not Just Reversing the Visible Binary String

In many languages, converting `43261596` to binary gives:

```text
10100101000001111010011100
```

That omits leading zeroes.

If you reverse only this visible string, you are not solving the same problem. The actual input is treated as:

```text
00000010100101000001111010011100
```

Those leading zeroes are real positions in the 32-bit representation. After reversal, they become trailing zeroes.

That is why the loop must run exactly `32` times, even if `n` becomes `0` earlier.

Stopping early loses information about width.

For example, with an 8-bit toy input:

```text
n = 00000001
```

The reversed result is:

```text
10000000
```

If we stopped once `n` became `0`, we would produce only:

```text
1
```

That is missing seven required zero positions.

### 8. Correctness Argument

We prove that the algorithm returns the 32-bit reversal of the input.

Let the original input bits be:

```text
b31 b30 ... b2 b1 b0
```

where `b0` is the least significant bit.

The algorithm repeats this operation exactly `32` times:

```python
answer = (answer << 1) | (n & 1)
n >>= 1
```

Invariant:

```text
After k iterations, answer contains exactly the k-bit sequence b0 b1 ... b(k-1).
```

Base case:

Before the loop, `k = 0` and `answer = 0`. The answer contains no consumed bits yet, so the invariant is true.

Inductive step:

Assume that after `k` iterations, `answer` contains:

```text
b0 b1 ... b(k-1)
```

At the start of iteration `k + 1`, the current least significant bit of `n` is `bk`, because the previous `k` lower bits have already been shifted away.

The algorithm shifts `answer` left by one, changing:

```text
b0 b1 ... b(k-1)
```

into:

```text
b0 b1 ... b(k-1) 0
```

Then it ORs in `bk`, producing:

```text
b0 b1 ... b(k-1) bk
```

So after `k + 1` iterations, `answer` contains exactly the first `k + 1` bits of the reversed output. The invariant is preserved.

Termination:

The loop runs exactly `32` iterations. By the invariant, after the loop `answer` contains:

```text
b0 b1 ... b31
```

That is the original 32-bit sequence reversed. Therefore the algorithm returns the correct result.

### 9. Complexity

The loop always runs exactly `32` iterations.

So under the LeetCode fixed-width definition:

```text
Time:  O(1)
Space: O(1)
```

If generalized to a variable bit width `w`, the time would be `O(w)` and the auxiliary space would still be `O(1)`.

### 10. Common Pitfalls

#### Stopping when `n == 0`

Do not write:

```python
while n:
    ...
```

That reverses only the visible nonzero suffix of the binary representation. The problem requires all 32 positions.

Use:

```python
for _ in range(32):
    ...
```

#### Forgetting to shift `answer` before inserting the bit

This is correct:

```python
answer = (answer << 1) | (n & 1)
```

This is not enough:

```python
answer |= n & 1
```

Without shifting, every extracted bit would be written into the same low position.

#### Confusing source direction and destination direction

The input is read from right to left:

```text
least significant bit first
```

The output is built from left to right by repeatedly shifting the current answer left.

That is the reversal.

#### Ignoring fixed-width behavior

The value `0` reversed over 32 bits is still `0`, but the value `1` reversed over 32 bits is:

```text
10000000000000000000000000000000
```

which is `2147483648`, not `1`.

#### Worrying about signed integers in Python

LeetCode describes the input as unsigned. Python integers are unbounded, but the loop only reads 32 low bits, so the implementation naturally behaves as a 32-bit unsigned reversal for valid inputs.

In languages with signed 32-bit integers, use unsigned shifts or unsigned integer types when needed.

### 11. First-Principles Summary

This problem follows from a few simple bit facts:

```text
1. A 32-bit integer is a fixed sequence of 32 binary positions.
2. Reversing bits means position i moves to position 31 - i.
3. n & 1 extracts the next original bit from the right.
4. n >> 1 discards the bit that was just consumed.
5. answer << 1 makes room to append the next reversed bit.
6. Repeating this exactly 32 times preserves the fixed-width meaning.
```

So the whole algorithm is:

> Read the input bits from least significant to most significant, append each one to the growing answer, and do this exactly 32 times so leading zeroes are handled correctly.

## Implementation

See `solutions/bit_manipulation/p190_reverse_bits.py`.

## Tests

See `tests/bit_manipulation/test_p190_reverse_bits.py`.

## Examples

### Example 1
- Input: `{'raw': '43261596\n2147483644'}`
- Output: `'See official examples'`

## Follow-up Practice

- Write the 32-bit binary form before reversing; do not drop leading zeroes.
- Trace the loop on small 4-bit or 8-bit examples first.
- Check edge cases such as `0`, `1`, powers of two, and all bits set.
