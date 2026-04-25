# 202. Happy Number

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/happy-number/
- Official Group: Hashmap
- Pattern Group: Hash Table
- Patterns: hash-table

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given a positive integer `n`, repeatedly perform this operation:

```text
Replace the number with the sum of the squares of its decimal digits.
```

For example, if the current number is `19`, its digits are `1` and `9`, so the next number is:

```text
1^2 + 9^2 = 1 + 81 = 82
```

A number is called **happy** if this repeated process eventually reaches `1`.

If the process never reaches `1`, the number is not happy.

So the problem is not asking us to compute one digit-square sum. It is asking:

> If we keep applying the same deterministic transformation to `n`, do we eventually reach `1`, or do we get trapped forever somewhere else?

The output is a boolean:

```text
True  if the process reaches 1
False if the process never reaches 1
```

---

### 2. Understand the Transformation

Define a helper function:

```text
next(x) = sum of squares of the digits of x
```

Then the process is just a sequence:

```text
n, next(n), next(next(n)), next(next(next(n))), ...
```

For `n = 19`, the sequence begins:

```text
19 -> 82 -> 68 -> 100 -> 1
```

Since it reaches `1`, `19` is happy.

For `n = 2`, the sequence begins:

```text
2 -> 4 -> 16 -> 37 -> 58 -> 89 -> 145 -> 42 -> 20 -> 4 -> ...
```

At that point, `4` appears again. Because the rule is deterministic, once `4` repeats, everything after it must repeat too:

```text
4 -> 16 -> 37 -> 58 -> 89 -> 145 -> 42 -> 20 -> 4 -> ...
```

That cycle does not contain `1`, so `2` is not happy.

---

### 3. Start From the Brute Force Idea

The most direct simulation is:

1. Compute the next number.
2. If it is `1`, return `True`.
3. Otherwise, keep going.

Conceptually:

```python
while n != 1:
    n = sum_of_squared_digits(n)

return True
```

This works for happy numbers because they eventually reach `1`.

But it has a serious flaw for unhappy numbers: it may loop forever.

For example, starting from `2`, the process never reaches `1`. A plain `while n != 1` loop has no reason to stop.

So the brute-force simulation is missing one essential question:

> How do we know that continuing cannot change the answer anymore?

---

### 4. The Key Observation: Repetition Means a Cycle

The transformation from one number to the next is deterministic.

That means:

```text
If the current number is x, the next number is always the same value next(x).
```

There is no randomness and no hidden state.

So if we ever see the same number twice, the future is forced to repeat exactly as before.

Suppose the sequence contains:

```text
... -> 37 -> 58 -> 89 -> 145 -> 42 -> 20 -> 4 -> 16 -> 37 -> ...
```

Once `37` appears a second time, the sequence after that second `37` must be:

```text
58 -> 89 -> 145 -> 42 -> 20 -> 4 -> 16 -> 37 -> ...
```

which is the same loop again.

Therefore:

```text
If the process reaches 1, the number is happy.
If the process repeats a non-1 value, the number is not happy.
```

This is the whole reason a hash set is enough.

We do not need to remember the order of all numbers. We only need fast membership:

```text
Have I seen this number before?
```

---

### 5. Why the Process Must Eventually Reach 1 or Repeat

A natural concern is:

> Could the numbers grow forever without reaching 1 or repeating?

No.

For a number with `d` digits, the largest possible sum of squared digits is:

```text
d * 9^2 = 81d
```

For large numbers, `81d` is much smaller than the number itself.

For example, even a 10-digit number maps to at most:

```text
10 * 81 = 810
```

So after enough applications, the sequence falls into a small bounded range. Once it is inside a finite set of possible values, one of two things must happen:

1. It reaches `1`.
2. It revisits some previous value and enters a cycle.

This is the same basic idea as walking through rooms with one outgoing door from each room. If you never reach the exit and there are only finitely many rooms, eventually you must enter a room you have already visited.

---

### 6. The Seen-Set Invariant

Maintain a set called `seen`.

The invariant is:

```text
Before each iteration, seen contains exactly the non-1 numbers already produced by the process.
```

At each current number `n`:

1. If `n == 1`, the process reached the happy terminal value, so return `True`.
2. If `n in seen`, this exact state has occurred before, so the process is in a cycle that does not reach `1`, so return `False`.
3. Otherwise, add `n` to `seen` and replace `n` with `sum_of_squared_digits(n)`.

The order matters.

We check for `1` before declaring a repeated value unhappy. In practice `1` would lead to itself forever:

```text
1 -> 1 -> 1 -> ...
```

but that is the successful cycle, not a failure cycle.

---

### 7. Detailed Algorithm

Use two pieces of logic:

1. A helper that computes the digit-square sum of one number.
2. A loop that simulates the process while detecting cycles.

Digit-square helper:

```text
Start total at 0.
While x has digits left:
    Take the last digit using x % 10.
    Add digit * digit to total.
    Remove the last digit using x //= 10.
Return total.
```

Main algorithm:

```text
seen = empty set

while n is not 1 and n has not been seen before:
    add n to seen
    n = sum_of_squared_digits(n)

return whether n is 1
```

This form is compact because there are only two possible reasons the loop stops:

```text
n == 1       -> happy
n in seen    -> cycle, not happy
```

---

### 8. Python-Style Pseudocode

```python
def isHappy(n: int) -> bool:
    def digit_square_sum(x: int) -> int:
        total = 0

        while x > 0:
            digit = x % 10
            total += digit * digit
            x //= 10

        return total

    seen = set()

    while n != 1 and n not in seen:
        seen.add(n)
        n = digit_square_sum(n)

    return n == 1
```

The helper can also be written using string conversion:

```python
def digit_square_sum(x: int) -> int:
    return sum(int(ch) ** 2 for ch in str(x))
```

Both versions express the same idea. The arithmetic version makes the digit operation explicit and avoids building a string.

---

### 9. Walk Through Example 1

Input:

```text
n = 19
```

Start:

```text
seen = {}
current = 19
```

`19` is not `1` and has not been seen, so add it:

```text
seen = {19}
```

Compute the next number:

```text
1^2 + 9^2 = 82
```

Now:

```text
current = 82
```

`82` is not `1` and has not been seen:

```text
seen = {19, 82}
8^2 + 2^2 = 64 + 4 = 68
```

Now:

```text
current = 68
```

Continue:

```text
seen = {19, 82, 68}
6^2 + 8^2 = 36 + 64 = 100
```

Now:

```text
current = 100
```

Continue:

```text
seen = {19, 82, 68, 100}
1^2 + 0^2 + 0^2 = 1
```

Now:

```text
current = 1
```

The loop stops because `current == 1`, so return:

```text
True
```

---

### 10. Walk Through Example 2

Input:

```text
n = 2
```

The sequence is:

```text
2 -> 4 -> 16 -> 37 -> 58 -> 89 -> 145 -> 42 -> 20 -> 4
```

Track the set:

```text
current = 2,   seen = {}
current = 4,   seen = {2}
current = 16,  seen = {2, 4}
current = 37,  seen = {2, 4, 16}
current = 58,  seen = {2, 4, 16, 37}
current = 89,  seen = {2, 4, 16, 37, 58}
current = 145, seen = {2, 4, 16, 37, 58, 89}
current = 42,  seen = {2, 4, 16, 37, 58, 89, 145}
current = 20,  seen = {2, 4, 16, 37, 58, 89, 145, 42}
current = 4,   seen = {2, 4, 16, 37, 58, 89, 145, 42, 20}
```

At the end, `current = 4`, and `4` is already in `seen`.

That proves the sequence has entered a cycle. Since the loop checks for `1` on every step and never found it, this cycle does not lead to `1`.

Return:

```text
False
```

---

### 11. Correctness

We prove that the algorithm returns `True` exactly when `n` is a happy number.

#### Lemma 1: The algorithm follows the required process exactly.

Each loop iteration replaces the current number with the sum of the squares of its digits. That is exactly the operation defined by the problem. Therefore, the algorithm examines the same sequence of numbers produced by repeatedly applying the happy-number rule.

#### Lemma 2: If the algorithm returns `True`, the original number is happy.

The algorithm returns `True` only when the current number is `1`. By Lemma 1, this current number appears in the sequence generated from the original input. Therefore, the repeated process reaches `1`, so the original number is happy.

#### Lemma 3: If the algorithm returns `False`, the original number is not happy.

The algorithm returns `False` only when the current number has already appeared in `seen`. By the invariant, every number in `seen` appeared earlier in the same generated sequence.

Because the transformation is deterministic, reaching the same number again forces all future numbers to repeat the same cycle. The algorithm also would have stopped earlier if the sequence had reached `1`. Therefore, this repeated cycle does not include a first reach of `1`, and the original number is not happy.

#### Lemma 4: The algorithm terminates.

Repeated digit-square sums eventually enter a bounded finite range. Inside a finite range, the sequence must either reach `1` or repeat a previous value. The algorithm stops in either case.

#### Theorem: The algorithm is correct.

By Lemma 2, every `True` result is correct. By Lemma 3, every `False` result is correct. By Lemma 4, the algorithm always reaches one of these results. Therefore, the algorithm correctly decides whether the input number is happy.

---

### 12. Complexity

Let `k` be the number of digits in the starting number.

Computing one digit-square sum costs:

```text
O(k)
```

for the number currently being processed.

The sequence quickly falls into a small bounded range, so for normal LeetCode integer constraints, the number of distinct states visited is effectively constant. More generally, the first few transitions shrink very large inputs, and after that only bounded values are processed.

So the practical complexity is:

```text
Time:  O(k)
Space: O(1)
```

If we describe the simulation purely in terms of the number of distinct states visited before termination, call that number `m`, then:

```text
Time:  O(m * k)
Space: O(m)
```

where `m` is the number of unique intermediate values stored in `seen`.

For fixed-width integers, `m` is bounded by a constant.

---

### 13. Common Pitfalls

- **Looping until `n == 1` without cycle detection.** This hangs forever for unhappy numbers such as `2`.
- **Adding a number to `seen` after computing the next value but checking the wrong state.** The invariant should be clear: check whether the current state has already occurred before advancing.
- **Treating any cycle as failure before handling `1`.** The value `1` maps to itself, but that is the success condition.
- **Forgetting zeros in digit processing.** Zeros contribute `0`, so they do not change the sum, but they are still valid digits. For example, `100 -> 1`.
- **Thinking the set stores digits.** The set stores whole intermediate numbers, not individual digits. The cycle is a cycle of numbers.
- **Assuming the sequence might grow forever.** Digit-square sums force large numbers downward into a bounded range.

---

### 14. First-Principles Summary

A happy-number sequence is a deterministic walk through integer states.

From any current number, there is exactly one next number. Therefore, the only possible long-term outcomes are:

```text
reach 1
or
repeat a previous state forever
```

The hash set exists for one precise reason: to detect the second outcome.

The algorithm is just:

```text
simulate the required process
remember every state already seen
stop successfully at 1
stop unsuccessfully at the first repeated non-1 state
```

That is the entire problem reduced to its first principles.

## Implementation
See `solutions/hash_table/p202_happy_number.py`.

## Tests
See `tests/hash_table/test_p202_happy_number.py`.

## Examples

### Example 1
- Input: `{'n': 19}`
- Output: `True`

### Example 2
- Input: `{'n': 2}`
- Output: `False`

## Follow-up Practice
- Trace the exact sequence for `7`, `10`, and `116`.
- Explain why seeing the same intermediate number twice proves the future will repeat.
- Rewrite the solution using Floyd's cycle detection and compare the space usage with the seen-set version.
