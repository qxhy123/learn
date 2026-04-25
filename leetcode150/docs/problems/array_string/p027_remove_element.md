# 27. Remove Element

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/remove-element/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: in-place, slow-fast-pointers

## First-Principles Explanation

### 1. What the Problem Is Actually Asking

You are given:

- an array `nums`
- a value `val`

You must remove every occurrence of `val` **in place** and return how many values remain.

That sounds simple, but the wording matters:

- You are **not** asked to create and return a new array.
- You are **not** asked to physically shrink Python's list object.
- You are asked to rearrange the existing array so that the first `k` positions contain the values that should be kept, where `k` is the returned length.

Everything after index `k - 1` is irrelevant.

So the real task is:

> Scan the array once, keep every value that is not `val`, and compact those kept values into the front of the same array.

The problem also allows the order of kept elements to change, but the simplest linear solution naturally preserves their original order anyway.

### 2. Start With the Most Direct Baseline

If we ignore the in-place requirement, the easiest idea is:

1. Create a new list `kept`.
2. Append every `x` in `nums` such that `x != val`.
3. Copy `kept` back into the front of `nums`.
4. Return `len(kept)`.

Conceptually:

```python
kept = []

for x in nums:
    if x != val:
        kept.append(x)

for i in range(len(kept)):
    nums[i] = kept[i]

return len(kept)
```

This is easy to reason about, but it uses extra memory proportional to the number of kept elements.

That violates the spirit of the problem. The whole point is to realize:

> We do not need a second array. We only need to know where the next kept element should go.

### 3. A Worse In-Place Idea to Avoid

Another instinct is:

1. Walk through the array.
2. Whenever you see `val`, shift everything after it one step left.

That works logically, but it can become quadratic.

For example, if `nums` is mostly `val`, then every removal triggers a long shift. Repeating that many times costs `O(n^2)`.

So the key question becomes:

> Can we avoid repeated shifting by writing each kept element exactly once?

Yes.

### 4. Key Observation

When you read `nums` from left to right, every element belongs to exactly one of two groups:

- discard it because it equals `val`
- keep it because it does not equal `val`

If an element should be kept, the only thing we need to know is:

> At which front position should this kept element be written?

That suggests two pointers:

- `read`: which element are we currently inspecting?
- `write`: where should the next kept element be placed?

As `read` moves through the array:

- if `nums[read] == val`, do nothing except continue
- if `nums[read] != val`, copy it to `nums[write]` and advance `write`

At the end:

- `write` equals the number of kept elements
- the prefix `nums[0:write]` contains the answer

This avoids repeated shifting because every kept element is written at most once.

### 5. The Invariant

The whole algorithm is powered by one precise invariant:

> After processing indices `0` through `read - 1`, the subarray `nums[0:write]` contains exactly the elements from that processed prefix that are not equal to `val`, in their original left-to-right order.

This invariant tells us everything we need:

- The prefix before `write` is already correct.
- Everything from `write` onward is still unprocessed or irrelevant.
- When we see a keepable element, placing it at `nums[write]` is safe because that is exactly the next slot in the compacted prefix.

This is the first-principles heart of the solution:

> `write` is not "the current index." It is "the size of the valid compacted prefix built so far."

### 6. Detailed Algorithm

Initialize:

```text
write = 0
```

Then scan the array with `read` from left to right:

1. Look at `nums[read]`.
2. If it equals `val`, skip it.
3. Otherwise:
   - assign `nums[write] = nums[read]`
   - increment `write`
4. After the scan ends, return `write`.

Pseudocode:

```python
write = 0

for read in range(len(nums)):
    if nums[read] != val:
        nums[write] = nums[read]
        write += 1

return write
```

A small but important detail:

Sometimes `read == write`, so the assignment writes a value onto itself. That is completely fine. The algorithm stays simple by not treating that case specially.

### 7. Walk Through Example 1

Input:

```text
nums = [3, 2, 2, 3]
val = 3
```

Start:

```text
write = 0
```

Process `read = 0`:

```text
nums[0] = 3
```

This equals `val`, so discard it.

```text
write = 0
nums is still [3, 2, 2, 3]
```

Process `read = 1`:

```text
nums[1] = 2
```

Keep it. Write it to `nums[write] = nums[0]`.

```text
nums becomes [2, 2, 2, 3]
write = 1
```

Process `read = 2`:

```text
nums[2] = 2
```

Keep it. Write it to `nums[1]`.

```text
nums becomes [2, 2, 2, 3]
write = 2
```

Process `read = 3`:

```text
nums[3] = 3
```

Discard it.

Final result:

```text
write = 2
nums[:2] = [2, 2]
```

So we return `2`.

### 8. Walk Through Example 2

Input:

```text
nums = [0, 1, 2, 2, 3, 0, 4, 2]
val = 2
```

We will track the meaningful prefix `nums[0:write]`.

Start:

```text
write = 0
valid prefix = []
```

`read = 0`, value `0`:

- keep it
- write to `nums[0]`

```text
write = 1
valid prefix = [0]
```

`read = 1`, value `1`:

- keep it
- write to `nums[1]`

```text
write = 2
valid prefix = [0, 1]
```

`read = 2`, value `2`:

- discard it

```text
write = 2
valid prefix = [0, 1]
```

`read = 3`, value `2`:

- discard it

```text
write = 2
valid prefix = [0, 1]
```

`read = 4`, value `3`:

- keep it
- write to `nums[2]`

```text
nums becomes [0, 1, 3, 2, 3, 0, 4, 2]
write = 3
valid prefix = [0, 1, 3]
```

`read = 5`, value `0`:

- keep it
- write to `nums[3]`

```text
nums becomes [0, 1, 3, 0, 3, 0, 4, 2]
write = 4
valid prefix = [0, 1, 3, 0]
```

`read = 6`, value `4`:

- keep it
- write to `nums[4]`

```text
nums becomes [0, 1, 3, 0, 4, 0, 4, 2]
write = 5
valid prefix = [0, 1, 3, 0, 4]
```

`read = 7`, value `2`:

- discard it

Final result:

```text
write = 5
nums[:5] = [0, 1, 3, 0, 4]
```

That is a valid answer because those are exactly the elements different from `2`.

LeetCode's sample output may show the kept prefix in a different order, such as `[0,1,4,0,3]`, because the problem allows any arrangement of the kept elements in the first `k` positions. The forward-copy solution above simply preserves the original relative order.

### 9. Reference Implementation

```python
def removeElement(nums, val):
    write = 0

    for read in range(len(nums)):
        if nums[read] != val:
            nums[write] = nums[read]
            write += 1

    return write
```

### 10. Why This Is Correct

We prove correctness from the invariant.

Before the loop starts:

- no elements have been processed
- `write = 0`
- `nums[0:write]` is an empty prefix

So the invariant holds: the kept elements from the empty processed prefix are exactly the empty list.

Now assume the invariant holds before processing index `read`.

There are two cases:

1. `nums[read] == val`

   This element should not appear in the output. We leave `write` unchanged and do not modify the compacted prefix. Therefore `nums[0:write]` still contains exactly the kept elements from the processed prefix.

2. `nums[read] != val`

   This element belongs in the output. By the invariant, `nums[0:write]` already contains exactly the kept elements from earlier indices. So the correct place for the new kept element is exactly `nums[write]`. After writing it there and incrementing `write`, the prefix `nums[0:write]` contains exactly the kept elements from the extended processed prefix.

Thus the invariant is preserved in all cases.

After the loop ends, every index has been processed. By the invariant, `nums[0:write]` contains exactly all elements of the original array that are not equal to `val`. Therefore returning `write` gives the correct count, and the required in-place prefix is correct.

### 11. Complexity

- Time: `O(n)` because each element is inspected once.
- Extra space: `O(1)` because only a few variables are used.

### 12. Common Pitfalls

- Returning the modified array instead of the new length.
- Forgetting that only the first `k` positions matter after the function returns.
- Trying to delete from the Python list while iterating, which changes indices and makes the logic messy.
- Repeatedly shifting elements left after every match, which degrades to `O(n^2)`.
- Confusing this problem with "stable removal is required." The problem does not require stability, even though this solution happens to preserve order.

### 13. First-Principles Summary

This problem is not really about deletion. It is about **compaction**.

The winning idea is:

- treat the array as a stream of values to inspect
- maintain a front region that already contains all kept values
- use `write` as the boundary of that valid region

Once you see that `write` is "the length of the cleaned prefix so far," the algorithm becomes unavoidable:

- skip unwanted values
- copy wanted values forward
- return the final boundary

That is why a single pass with two pointers solves the problem cleanly.

## Implementation

See `solutions/array_string/p027_remove_element.py`.

## Tests

See `tests/array_string/test_p027_remove_element.py`.

## Examples

### Example 1
- Input: `{'nums': [3, 2, 2, 3], 'val': 3}`
- Output: `'2, nums = [2,2,_,_]'`

### Example 2
- Input: `{'nums': [0, 1, 2, 2, 3, 0, 4, 2], 'val': 2}`
- Output: `'5, nums = [0,1,4,0,3,_,_,_]'`

## Follow-up Practice
- Trace the invariant after each index.
- Test empty/singleton/all-removed/no-removed inputs.
- Compare the compacting approach with a repeated-shift approach and explain why one is linear and the other can be quadratic.
