# 238. Product of Array Except Self

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/product-of-array-except-self/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: prefix-suffix, array-product

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given an integer array:

```text
nums = [nums[0], nums[1], ..., nums[n - 1]]
```

return a new array `answer` where each position `i` contains the product of every input value except `nums[i]` itself.

In mathematical form:

```text
answer[i] = nums[0] * nums[1] * ... * nums[i - 1] * nums[i + 1] * ... * nums[n - 1]
```

For example:

```text
nums = [1, 2, 3, 4]
```

The answer is:

```text
answer[0] =     2 * 3 * 4 = 24
answer[1] = 1 *     3 * 4 = 12
answer[2] = 1 * 2 *     4 = 8
answer[3] = 1 * 2 * 3     = 6
```

So we return:

```text
[24, 12, 8, 6]
```

The important constraint is that the usual LeetCode version asks us not to use division and to run in linear time. That matters because the tempting shortcut:

```text
total_product / nums[i]
```

is not the intended solution and also becomes awkward or invalid when zeros appear.

So the real problem is:

> For every index, compute the product of all numbers strictly to its left and all numbers strictly to its right, without dividing.

---

### 2. Start From the Brute Force Idea

The most direct solution is to compute each answer independently.

For every index `i`:

1. Start `product = 1`.
2. Scan every index `j`.
3. If `j != i`, multiply by `nums[j]`.
4. Store the result in `answer[i]`.

Conceptually:

```python
answer = []

for i in range(len(nums)):
    product = 1
    for j in range(len(nums)):
        if j != i:
            product *= nums[j]
    answer.append(product)
```

This is correct because it follows the definition exactly.

But it repeats almost all of the same work for neighboring indices. For `nums = [1, 2, 3, 4]`, the product `1 * 2` is useful for multiple positions, and the product `3 * 4` is useful for multiple positions. The brute-force method forgets those partial products and recomputes them again and again.

Its cost is:

```text
n choices for i
n scanned values for each i
```

So the time complexity is `O(n^2)`.

We need to reuse information across indices.

---

### 3. The Key Observation

For a fixed index `i`, the product except self naturally splits into two independent pieces:

```text
product of everything before i
*
product of everything after i
```

That is:

```text
answer[i] = left_product[i] * right_product[i]
```

where:

```text
left_product[i]  = nums[0] * nums[1] * ... * nums[i - 1]
right_product[i] = nums[i + 1] * nums[i + 2] * ... * nums[n - 1]
```

Notice what is deliberately missing from both pieces:

```text
nums[i]
```

The value at index `i` is excluded because the left side stops before `i`, and the right side starts after `i`.

This gives us a division-free plan:

1. Compute the product of all values to the left of each index.
2. Compute the product of all values to the right of each index.
3. Multiply those two pieces together.

The only question is how to do that without wasting extra space.

---

### 4. Prefix Products: What Is Known From the Left?

Suppose we scan from left to right and maintain:

```text
prefix = product of all numbers already seen
```

When we are standing at index `i`, the numbers already seen are exactly:

```text
nums[0], nums[1], ..., nums[i - 1]
```

So before multiplying in `nums[i]`, `prefix` is exactly the product of everything to the left of `i`.

That means we can write:

```text
answer[i] = prefix
```

Then we update:

```text
prefix *= nums[i]
```

so that the next index sees a prefix that includes the current value.

For `nums = [1, 2, 3, 4]`:

```text
start prefix = 1

i = 0: answer[0] = 1          then prefix *= 1 -> 1
i = 1: answer[1] = 1          then prefix *= 2 -> 2
i = 2: answer[2] = 2          then prefix *= 3 -> 6
i = 3: answer[3] = 6          then prefix *= 4 -> 24
```

After the left-to-right pass:

```text
answer = [1, 1, 2, 6]
```

At this point, `answer[i]` means:

```text
product of all values to the left of i
```

It is not the final answer yet, because each position still needs the product of values to its right.

---

### 5. Suffix Products: What Is Known From the Right?

Now scan from right to left and maintain:

```text
suffix = product of all numbers already seen from the right
```

When we are standing at index `i`, the numbers already seen from the right are exactly:

```text
nums[i + 1], nums[i + 2], ..., nums[n - 1]
```

So before multiplying in `nums[i]`, `suffix` is exactly the product of everything to the right of `i`.

The current `answer[i]` already contains the left product. Therefore we finish the index by multiplying in the right product:

```text
answer[i] *= suffix
```

Then update:

```text
suffix *= nums[i]
```

so that the next index to the left sees a suffix that includes the current value.

This order is crucial. We multiply by `suffix` before updating `suffix` with `nums[i]`; otherwise `nums[i]` would accidentally be included in its own answer.

---

### 6. State and Invariants

The algorithm uses three pieces of state:

```text
answer = output array being built
prefix = product of values strictly left of the current index during the first pass
suffix = product of values strictly right of the current index during the second pass
```

First pass invariant:

```text
Before processing index i from left to right,
prefix == nums[0] * nums[1] * ... * nums[i - 1].
```

So assigning:

```text
answer[i] = prefix
```

stores the correct left-side product for index `i`.

Second pass invariant:

```text
Before processing index i from right to left,
suffix == nums[i + 1] * nums[i + 2] * ... * nums[n - 1].
```

So assigning:

```text
answer[i] *= suffix
```

turns the stored left-side product into:

```text
left product * right product
```

which is exactly the product of every number except `nums[i]`.

The empty product is `1`. That is why `prefix` starts as `1` for index `0`, which has no values to its left, and `suffix` starts as `1` for index `n - 1`, which has no values to its right.

---

### 7. Detailed Algorithm

1. Let `n = len(nums)`.
2. Create `answer` with length `n`.
3. Set `prefix = 1`.
4. Scan `i` from `0` to `n - 1`:
   - Store the product of everything before `i`:

```text
answer[i] = prefix
```

   - Include `nums[i]` for the next index:

```text
prefix *= nums[i]
```

5. Set `suffix = 1`.
6. Scan `i` from `n - 1` down to `0`:
   - Multiply the already-stored left product by the product of everything after `i`:

```text
answer[i] *= suffix
```

   - Include `nums[i]` for the next index to the left:

```text
suffix *= nums[i]
```

7. Return `answer`.

---

### 8. Pseudocode

```python
def productExceptSelf(nums):
    n = len(nums)
    answer = [1] * n

    prefix = 1
    for i in range(n):
        answer[i] = prefix
        prefix *= nums[i]

    suffix = 1
    for i in range(n - 1, -1, -1):
        answer[i] *= suffix
        suffix *= nums[i]

    return answer
```

The repository scaffold for this problem currently keeps the implementation placeholder in the solution file, but the intended algorithm is the two-pass prefix/suffix method shown here.

---

### 9. Walkthrough: `nums = [1, 2, 3, 4]`

First pass: store left products.

```text
prefix starts at 1
```

| Index | `nums[i]` | `prefix` before index | Write to `answer[i]` | `prefix` after update |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 1 | 1 |
| 1 | 2 | 1 | 1 | 2 |
| 2 | 3 | 2 | 2 | 6 |
| 3 | 4 | 6 | 6 | 24 |

After the first pass:

```text
answer = [1, 1, 2, 6]
```

These are not final values. They mean:

```text
answer[0] = product left of index 0 = 1
answer[1] = product left of index 1 = 1
answer[2] = product left of index 2 = 1 * 2 = 2
answer[3] = product left of index 3 = 1 * 2 * 3 = 6
```

Second pass: multiply by right products.

```text
suffix starts at 1
```

| Index | `nums[i]` | Current `answer[i]` | `suffix` before index | Final `answer[i]` | `suffix` after update |
|---:|---:|---:|---:|---:|---:|
| 3 | 4 | 6 | 1 | 6 | 4 |
| 2 | 3 | 2 | 4 | 8 | 12 |
| 1 | 2 | 1 | 12 | 12 | 24 |
| 0 | 1 | 1 | 24 | 24 | 24 |

Final result:

```text
[24, 12, 8, 6]
```

---

### 10. Walkthrough With Zero: `nums = [-1, 1, 0, -3, 3]`

This example shows why division is fragile and why prefix/suffix products are safer.

The expected output is:

```text
[0, 0, 9, 0, 0]
```

At index `2`, the excluded value is `0`, so the product of everything else is:

```text
-1 * 1 * -3 * 3 = 9
```

At every other index, the excluded value is not the zero, so the remaining product still includes `0`, making the answer `0`.

The two-pass method handles this naturally.

First pass stores left products:

| Index | `nums[i]` | `prefix` before index | `answer[i]` | `prefix` after update |
|---:|---:|---:|---:|---:|
| 0 | -1 | 1 | 1 | -1 |
| 1 | 1 | -1 | -1 | -1 |
| 2 | 0 | -1 | -1 | 0 |
| 3 | -3 | 0 | 0 | 0 |
| 4 | 3 | 0 | 0 | 0 |

After the first pass:

```text
answer = [1, -1, -1, 0, 0]
```

Second pass multiplies by right products:

| Index | `nums[i]` | Current `answer[i]` | `suffix` before index | Final `answer[i]` | `suffix` after update |
|---:|---:|---:|---:|---:|---:|
| 4 | 3 | 0 | 1 | 0 | 3 |
| 3 | -3 | 0 | 3 | 0 | -9 |
| 2 | 0 | -1 | -9 | 9 | 0 |
| 1 | 1 | -1 | 0 | 0 | 0 |
| 0 | -1 | 1 | 0 | 0 | 0 |

Final result:

```text
[0, 0, 9, 0, 0]
```

No special zero case was needed.

---

### 11. Correctness

We prove that the algorithm returns the product of all elements except self for every index.

#### Lemma 1: After the first pass, `answer[i]` equals the product of all elements strictly to the left of `i`.

Before index `i` is processed in the first pass, the invariant says:

```text
prefix = nums[0] * nums[1] * ... * nums[i - 1]
```

The algorithm writes:

```text
answer[i] = prefix
```

Therefore `answer[i]` receives exactly the product of all elements strictly to the left of `i`.

Then the algorithm updates:

```text
prefix *= nums[i]
```

so the invariant is true for the next index. Since `prefix` starts as `1`, the invariant is also true at index `0`, where there are no elements to the left.

Thus Lemma 1 holds for all indices.

#### Lemma 2: During the second pass, before index `i` is processed, `suffix` equals the product of all elements strictly to the right of `i`.

The second pass scans from right to left. Before processing index `i`, all elements to the right of `i` have already been multiplied into `suffix`, and `nums[i]` has not yet been multiplied into it.

So:

```text
suffix = nums[i + 1] * nums[i + 2] * ... * nums[n - 1]
```

The initial value `suffix = 1` is correct at the last index because there are no elements to its right.

After processing index `i`, the algorithm updates:

```text
suffix *= nums[i]
```

which makes `suffix` correct for the next index to the left.

Thus Lemma 2 holds for all indices.

#### Theorem: The final `answer[i]` is the product of every element except `nums[i]`.

By Lemma 1, before the second pass modifies `answer[i]`, it contains:

```text
product of all elements left of i
```

By Lemma 2, when the second pass reaches `i`, `suffix` contains:

```text
product of all elements right of i
```

The algorithm computes:

```text
answer[i] *= suffix
```

So the final value is:

```text
(product of all elements left of i) * (product of all elements right of i)
```

That is exactly the product of all elements in the array except `nums[i]`.

Therefore the algorithm is correct.

---

### 12. Complexity

Let `n` be the length of `nums`.

The algorithm performs:

```text
one left-to-right pass over n elements
one right-to-left pass over n elements
```

So the time complexity is:

```text
O(n)
```

The output array has length `n`. Besides that output, the algorithm uses only two scalar variables:

```text
prefix
suffix
```

So the extra space complexity is:

```text
O(1)
```

If the output array is counted as allocated space, total space is `O(n)`. In this problem, it is customary to exclude the required output array from extra-space analysis.

---

### 13. Common Pitfalls

#### Updating `prefix` too early

Wrong order:

```python
prefix *= nums[i]
answer[i] = prefix
```

This includes `nums[i]` in its own answer, which violates the problem statement.

Correct order:

```python
answer[i] = prefix
prefix *= nums[i]
```

#### Updating `suffix` too early

Wrong order:

```python
suffix *= nums[i]
answer[i] *= suffix
```

This also includes `nums[i]` in its own answer.

Correct order:

```python
answer[i] *= suffix
suffix *= nums[i]
```

#### Trying to divide by each element

Division appears simple:

```python
answer[i] = total_product // nums[i]
```

But the problem asks for a solution without division, and zeros make division invalid or require many special cases. The prefix/suffix method avoids all of that.

#### Forgetting the empty product is `1`

For the first element, there is no left side. For the last element, there is no right side. The correct neutral value for multiplication is `1`, not `0`.

That is why both running products start at `1`.

#### Allocating unnecessary prefix and suffix arrays

A valid beginner solution is:

```text
left[i]  = product left of i
right[i] = product right of i
answer[i] = left[i] * right[i]
```

But the right-side products do not need their own array. The output array can store left products first, and a single running `suffix` can finish the answers.

---

### 14. First-Principles Summary

The product except self is not mysterious. For each index, the answer is made of exactly two parts:

```text
things before the index * things after the index
```

A left-to-right scan can know only the first part, because it has seen the prefix. A right-to-left scan can know only the second part, because it has seen the suffix. Combining those two facts gives a linear-time algorithm.

The key discipline is to maintain products that are strictly outside the current index:

```text
prefix excludes nums[i]
suffix excludes nums[i]
```

Once that invariant is clear, zeros, negative numbers, and boundary positions all work without special handling.

## Implementation

See `solutions/array_string/p238_product_of_array_except_self.py`.

## Tests

See `tests/array_string/test_p238_product_of_array_except_self.py`.

## Examples

### Example 1
- Input: `{'nums': [1, 2, 3, 4]}`
- Output: `[24, 12, 8, 6]`

### Example 2
- Input: `{'nums': [-1, 1, 0, -3, 3]}`
- Output: `[0, 0, 9, 0, 0]`

## Follow-up Practice
- Trace both passes and write down `prefix`, `suffix`, and `answer` after each index.
- Compare the brute-force `O(n^2)` version with the two-pass `O(n)` version on the same input.
- Explain why the algorithm still works when the array contains one zero, multiple zeros, or negative numbers.
