# 80. Remove Duplicates from Sorted Array II

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: in-place, slow-fast-pointers

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given a **sorted** array `nums`.

You must modify it **in place** so that:

```text
each distinct value appears at most twice
```

Then return the new logical length.

The first important clarification is:

```text
You do not need to physically delete the extra elements from the Python list object.
You only need the prefix nums[0:new_length] to be correct.
```

So if the input is:

```text
[1, 1, 1, 2, 2, 3]
```

the correct transformed prefix is:

```text
[1, 1, 2, 2, 3]
```

and the return value is:

```text
5
```

The real problem is:

> While scanning a sorted array once, decide which elements belong in the final kept prefix, without using extra array space.

---

### 2. Start From the Brute-Force Baseline

The most direct idea is:

1. Scan the array.
2. Count how many times each value has appeared.
3. Copy the value into a new list only if its count is at most `2`.
4. Copy that new list back if needed.

Conceptually:

```python
result = []
count = {}

for x in nums:
    count[x] = count.get(x, 0) + 1
    if count[x] <= 2:
        result.append(x)
```

This works logically, but it does not satisfy the spirit of the problem well:

```text
it uses extra output storage
```

and it also ignores the strongest piece of structure in the input:

```text
the array is already sorted
```

That sorted order is what allows the in-place solution.

---

### 3. The Key Observation: In a Sorted Array, Duplicates Form One Block

Because `nums` is sorted:

```text
equal values are consecutive
```

So when you are reading left to right, all copies of a value arrive together.

For example:

```text
[0, 0, 1, 1, 1, 1, 2, 3, 3]
```

is naturally grouped as:

```text
[0, 0] [1, 1, 1, 1] [2] [3, 3]
```

That means you do not need a hash map to know whether the current element is the third, fourth, or fifth copy of some value from far in the past.

You only need to know:

```text
If I keep this element, would the kept prefix now contain three copies of the same value?
```

Since the array is sorted, the only way that can happen is if the current value matches the last two kept values.

That leads to the decisive first-principles rule:

> Keep the current number if it is different from the element two positions before the current write position.

Why "two positions before"?

Because:

- If the current value is different from `nums[write - 2]`, then adding it cannot create three equal copies in the kept prefix.
- If the current value is equal to `nums[write - 2]`, then the last two kept positions already hold that same value, so keeping this one would create an illegal third copy.

---

### 4. What State Do We Need?

We maintain one integer:

```text
write = length of the valid kept prefix so far
```

While reading the array, we interpret:

```text
nums[0:write]
```

as the deduplicated answer for everything we have processed so far.

We also have a read pointer:

```text
read = index of the next original element being examined
```

So the state is:

- `read` tells us what input element we are considering.
- `write` tells us where the next kept element should be written.

---

### 5. The Invariant

After processing `nums[0:read]`, maintain this invariant:

```text
nums[0:write] is exactly the correct answer for the processed prefix,
and every value appears there at most twice.
```

This invariant is the whole algorithm.

If it stays true after every step, then once we finish scanning the array:

```text
nums[0:write]
```

is the final valid array prefix, and `write` is the required answer.

Now ask:

```text
When is it safe to append nums[read] to that prefix?
```

There are two easy cases.

#### Case 1: Fewer than two elements have been kept

If `write < 2`, then the kept prefix is too short to violate the "at most twice" rule.

So the first two elements are always safe to keep:

```text
write = 0 or 1  -> always accept the current element
```

#### Case 2: At least two elements have been kept

Now the only dangerous situation is:

```text
current value == last kept value == second-last kept value
```

But "second-last kept value" is exactly:

```text
nums[write - 2]
```

So:

- If `nums[read] != nums[write - 2]`, keep it.
- If `nums[read] == nums[write - 2]`, skip it.

That is the entire decision rule.

---

### 6. The Algorithm Step by Step

1. Initialize:

```text
write = 0
```

2. Scan the array from left to right using `read`.

3. For each `nums[read]`:

- If `write < 2`, copy it into `nums[write]` and increment `write`.
- Otherwise compare it with `nums[write - 2]`.
- If they are different, copy it into `nums[write]` and increment `write`.
- If they are equal, skip it.

4. Return `write`.

In compact pseudocode:

```text
write = 0

for read from 0 to n - 1:
    if write < 2 or nums[read] != nums[write - 2]:
        nums[write] = nums[read]
        write += 1

return write
```

Notice something subtle and important:

```text
we compare against nums[write - 2], not nums[read - 2]
```

That is because `write` tracks the already accepted prefix. The decision must be based on the kept output, not merely on the original input positions.

---

### 7. Detailed Walkthrough

Take:

```text
nums = [1, 1, 1, 2, 2, 3]
```

We will track:

- `read`
- `nums[read]`
- `write`
- kept prefix `nums[0:write]`

Start:

```text
write = 0
kept prefix = []
```

#### Step 1: `read = 0`, value = `1`

Since `write < 2`, keep it.

Write `1` into `nums[0]`.

```text
write = 1
kept prefix = [1]
```

#### Step 2: `read = 1`, value = `1`

Still `write < 2`, so keep it.

Write `1` into `nums[1]`.

```text
write = 2
kept prefix = [1, 1]
```

#### Step 3: `read = 2`, value = `1`

Now `write >= 2`, so compare with `nums[write - 2] = nums[0] = 1`.

They are equal:

```text
nums[read] == nums[write - 2]
```

That means the kept prefix already ends with two `1`s, so keeping this one would create three copies.

Skip it.

```text
write = 2
kept prefix = [1, 1]
```

#### Step 4: `read = 3`, value = `2`

Compare with `nums[write - 2] = nums[0] = 1`.

They are different, so keep `2`.

Write `2` into `nums[2]`.

```text
write = 3
kept prefix = [1, 1, 2]
```

#### Step 5: `read = 4`, value = `2`

Compare with `nums[write - 2] = nums[1] = 1`.

They are different, so keep `2`.

Write `2` into `nums[3]`.

```text
write = 4
kept prefix = [1, 1, 2, 2]
```

This is correct: the second `2` is allowed.

#### Step 6: `read = 5`, value = `3`

Compare with `nums[write - 2] = nums[2] = 2`.

They are different, so keep `3`.

Write `3` into `nums[4]`.

```text
write = 5
kept prefix = [1, 1, 2, 2, 3]
```

Finished.

Return:

```text
5
```

and the meaningful prefix is:

```text
[1, 1, 2, 2, 3]
```

---

### 8. Why the Second Example Also Works

For:

```text
[0, 0, 1, 1, 1, 1, 2, 3, 3]
```

the kept prefix evolves like this:

```text
[]
[0]
[0, 0]
[0, 0, 1]
[0, 0, 1, 1]
[0, 0, 1, 1]      <- third 1 skipped
[0, 0, 1, 1]      <- fourth 1 skipped
[0, 0, 1, 1, 2]
[0, 0, 1, 1, 2, 3]
[0, 0, 1, 1, 2, 3, 3]
```

So the returned length is:

```text
7
```

and the valid prefix is:

```text
[0, 0, 1, 1, 2, 3, 3]
```

---

### 9. Python Code

```python
class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        write = 0

        for read in range(len(nums)):
            if write < 2 or nums[read] != nums[write - 2]:
                nums[write] = nums[read]
                write += 1

        return write
```

The code is short because the invariant is strong.

---

### 10. Why This Is Correct

We prove correctness by maintaining the invariant.

#### Base case

Before processing any elements:

```text
write = 0
nums[0:write] = []
```

This empty prefix is trivially correct for the empty processed prefix.

#### Inductive step

Assume after processing up to `read - 1`, the prefix `nums[0:write]` is exactly the correct answer for the processed elements, with each value appearing at most twice.

Now process `nums[read]`.

There are two possibilities.

##### Option A: We keep `nums[read]`

This happens when:

```text
write < 2
```

or:

```text
nums[read] != nums[write - 2]
```

If `write < 2`, adding one more element cannot create three equal copies.

If `write >= 2` and `nums[read] != nums[write - 2]`, then the current value is not equal to the value that would complete a triple in the sorted kept prefix. Since duplicates are consecutive in sorted order, appending it preserves the rule that each value appears at most twice.

So after writing it into `nums[write]`, the new prefix is still valid and still contains exactly the correct kept elements.

##### Option B: We skip `nums[read]`

This happens when:

```text
write >= 2 and nums[read] == nums[write - 2]
```

Because the array is sorted, the last two kept elements of that value are already present in the kept prefix, and the current element is another copy of the same value. Keeping it would create a third occurrence, violating the problem constraint.

So skipping it is exactly the correct action.

Thus the invariant remains true after every step.

At the end of the scan, `nums[0:write]` is exactly the desired modified prefix, so returning `write` is correct.

---

### 11. Complexity

- Time: `O(n)`, because each element is examined once.
- Extra space: `O(1)`, because we only use a few integer variables.

This is optimal for an in-place one-pass solution.

---

### 12. Common Pitfalls

#### Pitfall 1: Comparing with `nums[write - 1]`

If you write:

```python
if nums[read] != nums[write - 1]:
```

you are solving the wrong problem:

```text
keep each value at most once
```

That would incorrectly discard the second allowed copy.

#### Pitfall 2: Comparing with the original read-side neighborhood

The decision must be based on the **kept prefix**, not on where the element originally came from.

So the correct comparison is:

```text
nums[read] vs nums[write - 2]
```

not something like:

```text
nums[read] vs nums[read - 2]
```

because earlier extra duplicates may already have been skipped.

#### Pitfall 3: Forgetting that arrays of length `0`, `1`, or `2` are already valid

Those cases should naturally work with the `write < 2` condition.

#### Pitfall 4: Thinking the whole tail of the array must look clean

Only the prefix up to the returned length matters.

Anything beyond that is irrelevant to the problem.

---

### 13. First-Principles Summary

The problem looks like duplicate removal, but the real structure is:

```text
sorted input + in-place output prefix + at most two copies
```

Because the array is sorted, all equal values arrive together.

So to decide whether the current value can be kept, you do not need a full frequency table. You only need to know whether the already kept prefix ends with two copies of the same value.

That is exactly what this comparison asks:

```text
nums[read] != nums[write - 2]
```

So the algorithm is:

1. Treat `nums[0:write]` as the valid built answer.
2. Read each number once.
3. Keep it only if it would not become a third copy.
4. Return `write`.

That is why a one-pass, `O(1)`-space in-place solution is possible.

## Implementation

See `solutions/array_string/p080_remove_duplicates_from_sorted_array_ii.py`.

## Tests

See `tests/array_string/test_p080_remove_duplicates_from_sorted_array_ii.py`.

## Examples

### Example 1
- Input: `{'nums': [1, 1, 1, 2, 2, 3]}`
- Output: `'5, nums = [1,1,2,2,3,_]'`

### Example 2
- Input: `{'nums': [0, 0, 1, 1, 1, 1, 2, 3, 3]}`
- Output: `'7, nums = [0,0,1,1,2,3,3,_,_]'`

## Follow-up Practice
- Trace the invariant after each index.
- Test empty/singleton/boundary inputs.
- Compare a brute-force version with the optimized invariant.
