# 189. Rotate Array

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/rotate-array/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: in-place, reversal

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an array `nums` and an integer `k`.

You must rotate the array to the right by `k` steps.

Rotating right by one step means:

```text
The last element moves to the front.
Every other element shifts one position to the right.
```

For example:

```text
[1, 2, 3, 4, 5, 6, 7]
```

after one right rotation becomes:

```text
[7, 1, 2, 3, 4, 5, 6]
```

After three right rotations, it becomes:

```text
[5, 6, 7, 1, 2, 3, 4]
```

The important requirement is that the operation is performed on the same array object. In LeetCode's usual contract for this problem, the method mutates `nums` in-place and does not need to return a new array.

So the real problem is:

> Rearrange the existing array so that the last `k` logical elements become the first `k` physical elements, while preserving the relative order inside both parts.

If the array is:

```text
[a0, a1, a2, ..., a(n-k-1), a(n-k), ..., a(n-1)]
```

then after rotating right by `k`, it should become:

```text
[a(n-k), ..., a(n-1), a0, a1, ..., a(n-k-1)]
```

That is, rotation cuts the array into two pieces:

```text
left part:  nums[0:n-k]
right part: nums[n-k:n]
```

and swaps their order:

```text
right part + left part
```

---

### 2. Start From the Brute Force Idea

The most literal simulation is to rotate the array one step at a time.

One right rotation can be done by saving the last element, shifting everything else right, and writing the saved element at index `0`:

```python
last = nums[-1]

for index in range(len(nums) - 1, 0, -1):
    nums[index] = nums[index - 1]

nums[0] = last
```

If we repeat that operation `k` times, the result is correct.

Conceptually:

```python
for _ in range(k):
    last = nums[-1]
    for index in range(len(nums) - 1, 0, -1):
        nums[index] = nums[index - 1]
    nums[0] = last
```

This directly matches the definition of rotation, but it is too expensive.

If `n = len(nums)`, each single-step rotation shifts `n - 1` values. Repeating that `k` times costs:

```text
O(n * k)
```

That can be far too slow when `k` is large.

There is also a simple extra-array baseline:

```python
rotated = [0] * n

for index, value in enumerate(nums):
    rotated[(index + k) % n] = value

nums[:] = rotated
```

This is `O(n)` time, but it uses `O(n)` extra space. It is a good way to understand the target index mapping, but the standard optimized solution avoids the extra array.

---

### 3. Normalize `k` First

Rotating by the array length changes nothing.

For an array of length `n`:

```text
rotate right by n     -> original array
rotate right by n + 1 -> same as rotate right by 1
rotate right by 2n+3  -> same as rotate right by 3
```

So only the remainder matters:

```python
k %= n
```

For example:

```text
nums = [1, 2, 3, 4, 5]
k = 12
```

Since:

```text
12 % 5 = 2
```

rotating right by `12` is the same as rotating right by `2`.

This normalization is not just an optimization. It also identifies the true cut point:

```text
cut = n - k
```

Everything before `cut` stays in order but moves behind the suffix. Everything from `cut` to the end stays in order but moves to the front.

---

### 4. The Key Observation: Rotation Is a Block Move

After reducing `k`, the desired result is:

```text
nums[n-k:n] + nums[0:n-k]
```

For example:

```text
nums = [1, 2, 3, 4, 5, 6, 7]
k = 3
```

The cut point is:

```text
n - k = 7 - 3 = 4
```

So the two blocks are:

```text
left block  = [1, 2, 3, 4]
right block = [5, 6, 7]
```

The answer is:

```text
right block + left block
= [5, 6, 7, 1, 2, 3, 4]
```

The challenge is doing this block reorder in-place.

A direct in-place block move is awkward because writing the suffix into the front overwrites values from the prefix before those prefix values have been moved elsewhere.

The reversal trick solves that overwrite problem by using a simpler primitive:

```text
reverse a contiguous section in-place
```

Reversal is safe because it only swaps pairs of elements. No value is destroyed; every overwritten slot has already had its old value saved by the swap.

---

### 5. Why Three Reversals Work

Let the original array be split into two blocks:

```text
A = nums[0:n-k]
B = nums[n-k:n]
```

The desired rotated array is:

```text
B A
```

Now consider what happens if we reverse the whole array.

Original:

```text
A B
```

Reverse all elements:

```text
reverse(B) reverse(A)
```

This puts the two blocks in the correct block order, but each block is internally backwards.

To fix that, reverse each block individually:

```text
reverse(reverse(B)) reverse(reverse(A))
```

which becomes:

```text
B A
```

That is exactly the rotated array.

So the algorithm is:

```text
1. Reverse the whole array.
2. Reverse the first k elements.
3. Reverse the remaining n - k elements.
```

The first reversal moves the suffix block to the front and the prefix block to the back. The second and third reversals restore the internal order inside those two blocks.

---

### 6. State and Invariant

The core helper operation is:

```text
reverse nums[left:right] in-place, where both endpoints are inclusive
```

During this helper, maintain two pointers:

```text
left  = first not-yet-fixed index in the section
right = last not-yet-fixed index in the section
```

The invariant is:

```text
All positions outside nums[left:right + 1] within the chosen section
already contain their final reversed values.
```

At each step:

1. Swap `nums[left]` and `nums[right]`.
2. Move `left` one step right.
3. Move `right` one step left.

Why is this safe?

In a reversed section, the element at the left edge belongs at the right edge, and the element at the right edge belongs at the left edge. Swapping those two values fixes both positions at once.

After the swap, those two positions never need to change again. That shrinks the unresolved middle section while preserving the invariant.

When `left >= right`, there are no more pairs to swap. The whole section has been reversed.

The higher-level rotation invariant after the three reversals is:

```text
After reversing the whole array:
    the two blocks are in rotated block order, but each block is internally reversed.

After reversing the first k elements:
    the front block is restored to the correct suffix order.

After reversing the remaining elements:
    the back block is restored to the correct prefix order.
```

---

### 7. Detailed Algorithm

Given `nums` and `k`:

1. Let `n = len(nums)`.
2. If `n` is `0`, there is nothing to rotate.
3. Reduce `k` using:

```python
k %= n
```

4. If `k` is `0`, the array already has the correct order.
5. Reverse the entire array:

```text
reverse(0, n - 1)
```

6. Reverse the first `k` elements:

```text
reverse(0, k - 1)
```

7. Reverse the remaining elements:

```text
reverse(k, n - 1)
```

The helper `reverse(left, right)` swaps inward until the section is reversed.

---

### 8. Pseudocode

```python
def rotate(nums, k):
    n = len(nums)
    if n == 0:
        return

    k %= n
    if k == 0:
        return

    def reverse(left, right):
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

    reverse(0, n - 1)
    reverse(0, k - 1)
    reverse(k, n - 1)
```

On LeetCode, this function usually returns `None` because the array is modified in-place.

If a local test harness expects the modified array as the observable result, it may return `nums` after mutation, but that return value is not necessary for the core algorithm.

---

### 9. Example Walkthrough: `[1, 2, 3, 4, 5, 6, 7]`, `k = 3`

Start:

```text
nums = [1, 2, 3, 4, 5, 6, 7]
k = 3
n = 7
k %= n -> 3
```

The desired cut is:

```text
left block  = [1, 2, 3, 4]
right block = [5, 6, 7]
```

The target is:

```text
[5, 6, 7, 1, 2, 3, 4]
```

#### Step 1: Reverse the whole array

Reverse indices `0` through `6`:

```text
[1, 2, 3, 4, 5, 6, 7]
 ^                 ^
 swap 1 and 7
```

After first swap:

```text
[7, 2, 3, 4, 5, 6, 1]
    ^           ^
    swap 2 and 6
```

After second swap:

```text
[7, 6, 3, 4, 5, 2, 1]
       ^     ^
       swap 3 and 5
```

After third swap:

```text
[7, 6, 5, 4, 3, 2, 1]
```

The block order is now right-block-then-left-block, but each block is backwards:

```text
reverse([5, 6, 7]) reverse([1, 2, 3, 4])
= [7, 6, 5] [4, 3, 2, 1]
```

#### Step 2: Reverse the first `k` elements

Reverse indices `0` through `2`:

```text
[7, 6, 5, 4, 3, 2, 1]
 ^     ^
 swap 7 and 5
```

After the swap:

```text
[5, 6, 7, 4, 3, 2, 1]
```

Now the suffix block is in the correct order:

```text
[5, 6, 7]
```

#### Step 3: Reverse the remaining elements

Reverse indices `3` through `6`:

```text
[5, 6, 7, 4, 3, 2, 1]
          ^        ^
          swap 4 and 1
```

After first swap:

```text
[5, 6, 7, 1, 3, 2, 4]
             ^  ^
             swap 3 and 2
```

After second swap:

```text
[5, 6, 7, 1, 2, 3, 4]
```

Now both blocks are in the correct order, and the array is fully rotated.

---

### 10. Example Walkthrough: `[-1, -100, 3, 99]`, `k = 2`

Start:

```text
nums = [-1, -100, 3, 99]
k = 2
n = 4
k %= n -> 2
```

The two blocks are:

```text
left block  = [-1, -100]
right block = [3, 99]
```

Target:

```text
[3, 99, -1, -100]
```

Reverse the whole array:

```text
[-1, -100, 3, 99]
-> [99, 3, -100, -1]
```

Reverse the first `2` elements:

```text
[99, 3, -100, -1]
-> [3, 99, -100, -1]
```

Reverse the remaining `2` elements:

```text
[3, 99, -100, -1]
-> [3, 99, -1, -100]
```

The result matches the expected output.

---

### 11. Correctness

We prove that the three-reversal algorithm rotates the array to the right by `k` positions.

After normalization, `0 <= k < n`. If `k = 0`, rotating by `0` positions leaves the array unchanged, so returning immediately is correct.

For `k > 0`, split the original array into two consecutive blocks:

```text
A = nums[0:n-k]
B = nums[n-k:n]
```

A right rotation by `k` positions should produce:

```text
B A
```

The first reversal reverses the entire array. Reversing a concatenation reverses the order of the blocks and reverses each block internally, so:

```text
A B
```

becomes:

```text
reverse(B) reverse(A)
```

The second reversal is applied exactly to the first `k` elements. Those elements are `reverse(B)`, so reversing them produces `B`.

The third reversal is applied exactly to the remaining `n - k` elements. Those elements are `reverse(A)`, so reversing them produces `A`.

Therefore the final array is:

```text
B A
```

which is exactly the definition of rotating the original array to the right by `k` positions.

The helper reversal is correct because each swap places the two outermost unresolved elements into their final reversed positions, then shrinks the unresolved section. By induction on the section length, the helper reverses every requested section correctly. Since the rotation algorithm uses only correct reversals on the exact three required sections, the final array is correct.

---

### 12. Complexity

Let `n = len(nums)`.

The algorithm reverses:

```text
the whole array      -> n elements
the first k elements -> k elements
the remaining part   -> n - k elements
```

Each reversal touches each element in its section at most once, so the total work is linear:

```text
O(n)
```

The algorithm uses only a few variables and swaps values in the original array:

```text
O(1) extra space
```

The input array itself is modified in-place.

---

### 13. Common Pitfalls

- Forgetting `k %= n`. Without this, `k` larger than `n` can produce wrong section boundaries.
- Not handling an empty array before computing `k % n`, because modulo by zero is invalid.
- Reversing the wrong ranges. The three ranges are `0..n-1`, then `0..k-1`, then `k..n-1`.
- Confusing right rotation with left rotation. Right rotation moves the suffix to the front; left rotation moves the prefix to the back.
- Returning a new array when the caller expects in-place mutation of `nums`.
- Shifting elements one step at a time, which is correct but can degrade to `O(n * k)`.
- Overwriting prefix values when trying to copy the suffix to the front manually without temporary storage.
- Treating `k = 0` as if the second reversal should use an invalid meaningful range. The helper may tolerate it, but an early return makes the intent clearer.

---

### 14. First-Principles Summary

A right rotation is not mysterious movement of individual elements. It is a block transformation:

```text
A B -> B A
```

where `B` is the suffix of length `k`.

The hard part is doing that block transformation in-place without losing values. Reversal is the safe primitive because it only swaps pairs.

The whole idea is:

```text
A B
reverse everything -> reverse(B) reverse(A)
reverse each block -> B A
```

So the first-principles path is:

1. Understand rotation as moving the suffix block to the front.
2. Reduce `k` because full rotations cancel out.
3. Use reversal to change block order safely in-place.
4. Restore the internal order of each block with two more reversals.

That gives a linear-time, constant-space in-place algorithm.

## Implementation

See `solutions/array_string/p189_rotate_array.py`.

## Tests

See `tests/array_string/test_p189_rotate_array.py`.

## Examples

### Example 1
- Input: `{'nums': [1, 2, 3, 4, 5, 6, 7], 'k': 3}`
- Output: `[5, 6, 7, 1, 2, 3, 4]`

### Example 2
- Input: `{'nums': [-1, -100, 3, 99], 'k': 2}`
- Output: `[3, 99, -1, -100]`

## Follow-up Practice

- Trace the three reversals on an array of odd length and an array of even length.
- Try `k = 0`, `k = n`, and `k > n` to confirm why modulo normalization matters.
- Compare the one-step-at-a-time simulation, the extra-array mapping solution, and the three-reversal in-place solution.
- Explain why reversing the whole array alone is close but not enough.
