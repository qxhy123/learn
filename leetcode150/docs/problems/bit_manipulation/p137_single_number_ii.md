# 137. Single Number II

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/single-number-ii/
- Official Group: Bit Manipulation
- Pattern Group: Bit Manipulation
- Patterns: bit-manipulation

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an integer array `nums` with a very specific promise:

```text
Every value appears exactly three times, except one value that appears exactly once.
```

Return the value that appears once.

For example:

```text
nums = [2, 2, 3, 2]
```

The value `2` appears three times, and `3` appears once, so the answer is:

```text
3
```

Another example:

```text
nums = [0, 1, 0, 1, 0, 1, 99]
```

The values `0` and `1` each appear three times, and `99` appears once, so the answer is:

```text
99
```

The problem is not asking for all frequencies. It asks for the one number that survives after every triple-occurring number cancels out.

That cancellation is easy to imagine with a hash map, but the intended bit-manipulation insight is stronger:

> If a number appears three times, then each of its `1` bits contributes `3` to the count at that bit position. Counts contributed by triple-occurring numbers are multiples of `3`, so only the single number changes the remainder modulo `3`.

---

### 2. Start From the Baseline Frequency Solution

The most direct solution is to count how many times each value appears:

```python
counts = {}

for num in nums:
    counts[num] = counts.get(num, 0) + 1

for num, count in counts.items():
    if count == 1:
        return num
```

This is correct because the input guarantee says exactly one number has count `1` and every other number has count `3`.

Its cost is:

```text
Time:  O(n)
Space: O(n)
```

The time is already optimal because every element may be needed. The space is the part we can improve.

The deeper question is:

> Can we remember only the residue of each bit count modulo `3`, instead of remembering a frequency for every distinct integer?

Yes. That leads to an `O(1)` auxiliary-space bit solution.

---

### 3. Why Bit Counts Work

Think of every integer as a column of bits.

For a single bit position, each array value contributes either:

```text
0 if that bit is off
1 if that bit is on
```

Now sum those contributions across the entire array.

If a non-answer value appears three times, then at any bit position it contributes one of these totals:

```text
0 + 0 + 0 = 0
1 + 1 + 1 = 3
```

Both are `0` modulo `3`.

So after taking the bit count modulo `3`, all triple-occurring numbers disappear. The remaining residue is exactly the bit of the single number:

```text
bit_sum_at_position % 3 = answer_bit_at_position
```

Example with `nums = [2, 2, 3, 2]`:

```text
2 = 10₂
2 = 10₂
3 = 11₂
2 = 10₂
```

Count each bit:

```text
ones bit: 0 + 0 + 1 + 0 = 1 -> 1 % 3 = 1
 twos bit: 1 + 1 + 1 + 1 = 4 -> 4 % 3 = 1
```

The remaining bits are:

```text
11₂ = 3
```

That gives the answer.

---

### 4. The Simple Per-Bit Algorithm

A straightforward bit-count implementation is:

1. Choose a fixed integer width, usually `32` bits for LeetCode-style signed integers.
2. For each bit position `bit` from `0` to `31`:
   - Count how many numbers have that bit set.
   - If the count is not divisible by `3`, set that bit in the answer.
3. Convert the result back to a signed integer if the sign bit is set.

Pseudocode:

```python
answer = 0

for bit in range(32):
    count = 0

    for num in nums:
        if (num >> bit) & 1:
            count += 1

    if count % 3 != 0:
        answer |= 1 << bit

# If using a 32-bit signed interpretation:
if answer >= 2**31:
    answer -= 2**32

return answer
```

Why the signed conversion matters:

- Many languages store integers in fixed-width two's-complement form.
- Python integers are unbounded, so a reconstructed 32-bit negative number initially looks like a large positive value.
- Subtracting `2**32` converts that 32-bit pattern back to the intended negative integer.

For example, the 32-bit pattern for `-1` is all `1`s:

```text
11111111111111111111111111111111₂
```

As an unsigned value, that is `4294967295`. As a signed 32-bit value, it is `-1`.

This per-bit solution is often the easiest one to explain and debug.

---

### 5. From Bit Counts to a State Machine

There is also a compact one-pass bitmask solution that stores the modulo-`3` count state for all bit positions at once.

For each bit position, the count modulo `3` can be only:

```text
0, 1, or 2
```

When we read a new `1` at that bit, the state advances:

```text
0 -> 1 -> 2 -> 0
```

When we read a new `0`, the state stays the same.

We can encode the states with two bitmasks:

```text
ones = bit positions whose count is 1 modulo 3
twos = bit positions whose count is 2 modulo 3
```

For any bit position, it should never be in both masks at the same time:

```text
ones & twos == 0
```

That is the invariant.

After processing some prefix of the array:

```text
if a bit has appeared 0 mod 3 times: it is in neither ones nor twos
if a bit has appeared 1 mod 3 times: it is in ones
if a bit has appeared 2 mod 3 times: it is in twos
```

After the entire array is processed, every triple-occurring number has contributed `0 mod 3`, and the single number has contributed `1 mod 3`. Therefore:

```text
ones == the single number
```

---

### 6. Deriving the State-Machine Update

For each incoming number `num`, every bit of `num` should advance one step in the cycle:

```text
not seen -> seen once -> seen twice -> not seen
```

A common update is:

```python
ones = (ones ^ num) & ~twos
twos = (twos ^ num) & ~ones
```

This works because of two ideas.

First, XOR toggles membership for bits present in `num`:

```text
ones ^ num
```

For a bit that is `1` in `num`:

- if it was not in `ones`, it is tentatively added to `ones`
- if it was in `ones`, it is tentatively removed from `ones`

Second, the masks clear illegal overlap:

```text
& ~twos
& ~ones
```

A bit cannot mean both "seen once" and "seen twice". When a bit moves into `twos`, it must be absent from `ones`; when it completes the third sighting, it must be absent from both.

There is another equivalent version that computes the two new masks from the old masks more explicitly:

```python
new_ones = (ones ^ num) & ~twos
new_twos = (twos ^ num) & ~new_ones
ones = new_ones
twos = new_twos
```

The exact ordering matters because `twos` is cleared using the updated `ones` in this form.

---

### 7. State-Machine Walkthrough

Use the official example:

```text
nums = [2, 2, 3, 2]
```

In binary, using only the lower two bits:

```text
2 = 10₂
3 = 11₂
```

Start:

```text
ones = 00₂
twos = 00₂
```

Read first `2`:

```text
num  = 10₂
ones = 10₂   # bit 1 has appeared once
twos = 00₂
```

Read second `2`:

```text
num  = 10₂
ones = 00₂
twos = 10₂   # bit 1 has appeared twice
```

Read `3`:

```text
num  = 11₂
ones = 01₂   # bit 0 has appeared once; bit 1 completes third sighting and clears
twos = 00₂
```

Read third `2`:

```text
num  = 10₂
ones = 11₂
twos = 00₂
```

The final `ones` mask is:

```text
11₂ = 3
```

That is the single number.

This walkthrough shows the key invariant in action: `ones` and `twos` are not storing particular numbers from the array. They are storing, for every bit position simultaneously, how many times that bit has appeared modulo `3`.

---

### 8. Detailed Algorithm

Using the state-machine formulation:

1. Initialize two masks:

   ```text
   ones = 0
   twos = 0
   ```

2. For each `num` in `nums`:
   - Update `ones` to include bits that have now appeared `1 mod 3` times.
   - Update `twos` to include bits that have now appeared `2 mod 3` times.
   - Clear any bit from a mask when it belongs to the other state.

3. Return `ones`.

Python-style pseudocode:

```python
def singleNumber(nums):
    ones = 0
    twos = 0

    for num in nums:
        ones = (ones ^ num) & ~twos
        twos = (twos ^ num) & ~ones

    return ones
```

In Python, this compact version usually handles negative numbers naturally because Python bitwise operations behave consistently with an infinite two's-complement model. If writing the per-bit counting version, explicitly choose and convert a fixed width such as `32` bits.

---

### 9. Correctness Argument

We prove the state-machine algorithm returns the number that appears once.

#### Invariant

After processing any prefix of `nums`, for every bit position:

- the bit is set in `ones` if and only if the number of processed values with that bit set is congruent to `1` modulo `3`
- the bit is set in `twos` if and only if the number of processed values with that bit set is congruent to `2` modulo `3`
- the bit is set in neither mask if and only if the count is congruent to `0` modulo `3`

#### Initialization

Before processing any numbers, every bit count is `0`, which is `0 modulo 3`.

Both masks are initialized to `0`, so every bit is in neither mask. The invariant holds.

#### Maintenance

Consider one bit position and one incoming number.

If the incoming bit is `0`, then the count for that bit does not change. XOR with `0` leaves the corresponding mask bits unchanged, and the clearing operations preserve the valid state. The invariant remains true.

If the incoming bit is `1`, then the count for that bit increases by one. The valid states must cycle as:

```text
0 mod 3 -> 1 mod 3 -> 2 mod 3 -> 0 mod 3
```

The update using XOR toggles the bit into or out of the current mask, and the `& ~other_mask` clearing step prevents the bit from occupying both states. Therefore each bit advances exactly one position in the modulo-`3` cycle. The invariant remains true.

Because bitwise operations apply the same Boolean rule to every bit position in parallel, the invariant is maintained for the entire integer masks.

#### Termination

After all numbers are processed, every number that appears three times contributes `3` occurrences to each of its set bits, which is `0 modulo 3`.

Only the single number contributes `1` occurrence to its set bits.

By the invariant, exactly those bits are set in `ones`. Therefore `ones` equals the single number, so returning `ones` is correct.

---

### 10. Complexity

For the state-machine solution:

```text
Time:  O(n)
Space: O(1)
```

Each number is processed once, and each update uses a constant number of bitwise operations.

For the explicit 32-bit counting solution:

```text
Time:  O(32n) = O(n)
Space: O(1)
```

The constant `32` comes from the fixed integer width.

---

### 11. Common Pitfalls

- **Using plain XOR only.** XOR cancels pairs, not triples. It solves the version where every other number appears twice, but not this problem.
- **Forgetting negative numbers in a per-bit solution.** In Python, reconstructing a 32-bit answer requires converting values with the sign bit set back into negative integers.
- **Updating `twos` from the wrong `ones`.** In the compact state-machine formula, `twos = (twos ^ num) & ~ones` uses the updated `ones`.
- **Letting a bit live in both masks.** The invariant requires `ones & twos == 0`; the clearing masks are not optional.
- **Thinking `ones` stores a seen number.** It stores bit positions with count `1 modulo 3`, which only becomes the answer after all triples have canceled.
- **Generalizing without changing the state space.** This two-mask machine is specific to modulo `3`; different repetition counts need different logic.

---

### 12. First-Principles Summary

The array promise gives us a cancellation rule:

```text
three copies of the same bit contribute 0 modulo 3
```

So the answer can be recovered independently at every bit position by counting set bits modulo `3`.

The per-bit solution does this explicitly by summing each position. The state-machine solution does the same work implicitly: `ones` and `twos` encode the modulo-`3` count for all bit positions at the same time.

At the end, all bits from triple-occurring numbers have returned to state `0`, and the single number's bits remain in state `1`. That remaining `ones` mask is the answer.

## Implementation
See `solutions/bit_manipulation/p137_single_number_ii.py`.

## Tests
See `tests/bit_manipulation/test_p137_single_number_ii.py`.

## Examples

### Example 1
- Input: `{'nums': [2, 2, 3, 2]}`
- Output: `3`

### Example 2
- Input: `{'nums': [0, 1, 0, 1, 0, 1, 99]}`
- Output: `99`

## Follow-up Practice
- Trace the bit counts modulo `3` for `[2, 2, 3, 2]` by hand.
- Trace `ones` and `twos` for one positive example and one negative-number example.
- Re-derive why XOR alone is insufficient when duplicates appear three times.
