# 228. Summary Ranges

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/summary-ranges/
- Official Group: Intervals
- Pattern Group: Intervals
- Patterns: intervals, consecutive-run scanning

## First-Principles Explanation

### What The Problem Is Asking
You are given a sorted array of unique integers, `nums`, and you must compress it into the shortest list of human-readable consecutive ranges.

A consecutive range is a maximal block of numbers where every next number is exactly one larger than the previous number. The word *maximal* matters: if `[0, 1, 2]` appears together, the answer should contain one range, `"0->2"`, not `"0->1"` and `"2"`, because the run can still be extended through `2`.

The output formatting rule is:

- If a range contains one number, write just that number, for example `"7"`.
- If a range contains at least two numbers, write its first and last value as `"start->end"`, for example `"4->5"` or `"0->2"`.

So the problem is not asking us to find arbitrary intervals or merge unsorted intervals. The input is already sorted and has no duplicates. The real task is to identify every maximal consecutive run and then format each run correctly.

### Brute-Force Baseline
A very direct way to think about the problem is: for every number, ask whether it belongs to the same range as the previous number or starts a new range.

A clumsy brute-force version might build each range by repeatedly checking whether `current + 1` exists somewhere in the remaining array. If the array were not sorted, that could lead to repeated searches:

```text
for each unused number x:
    start = x
    while x + 1 exists in the unused numbers:
        extend the range
    output start or start->x
```

With linear searches, this can degrade toward `O(n^2)`: each attempt to extend a run may scan the array again.

But the input gives us two powerful facts for free:

1. `nums` is sorted in increasing order.
2. Every number is unique.

Because of those facts, we never need to search for `x + 1`. If `x + 1` is present and belongs to the same run, it must be exactly the next array element. That turns the problem from repeated searching into one left-to-right pass.

### Key Observation
For any adjacent pair `nums[i - 1]` and `nums[i]`:

- If `nums[i] == nums[i - 1] + 1`, both numbers are in the same consecutive run.
- Otherwise, `nums[i - 1]` is the final number of the current run, and `nums[i]` starts a new run.

There is no future element that can repair a gap. For example, once we see `2` followed by `4`, the missing `3` cannot appear later, because the array is sorted. Therefore the run ending at `2` is complete at that exact moment.

This is the whole reason the one-pass algorithm is correct: sorted order makes every gap final.

### Run / Range Invariant
During the scan, maintain the start of the current run.

After processing elements up to index `i - 1`, the invariant is:

```text
All ranges that ended before the current run have already been emitted,
and current_start is the first value of the only run that is still open.
```

The open run always ends at the most recently processed value. We do not need to store every value inside the run, because consecutiveness tells us the interior automatically. If the run starts at `0` and the latest value is `2`, then the run is exactly `[0, 1, 2]`.

When we inspect the next value, there are only two possible cases:

1. It continues the open run, so the invariant remains true and no output is produced yet.
2. It breaks the open run, so we emit the completed range and start a new open run at the current value.

At the end of the array, one run may still be open, so it must be emitted after the loop.

### Detailed Algorithm
Handle the array as a stream of sorted values.

1. If `nums` is empty, return an empty list. There are no ranges to summarize.
2. Set `start = nums[0]`. This begins the first open run.
3. Scan from the second element to the end.
4. For each index `i`, compare `nums[i]` with `nums[i - 1]`.
5. If `nums[i] == nums[i - 1] + 1`, the current run continues; do nothing.
6. Otherwise, the previous value `nums[i - 1]` is the end of the current run. Format and append the range from `start` to `nums[i - 1]`.
7. Start a new run by setting `start = nums[i]`.
8. After the scan, append the final open range from `start` to `nums[-1]`.
9. Return the accumulated list.

The only helper idea is range formatting:

```text
if start == end:
    output "start"
else:
    output "start->end"
```

### Detailed Example Walkthrough
Consider:

```text
nums = [0, 1, 2, 4, 5, 7]
```

Start with:

```text
start = 0
result = []
```

Now scan adjacent pairs.

1. Compare `0` and `1`.
   - `1 == 0 + 1`, so the run continues.
   - The open run is now `[0, 1]`.
   - Do not output yet, because the run might continue.

2. Compare `1` and `2`.
   - `2 == 1 + 1`, so the run continues.
   - The open run is now `[0, 1, 2]`.

3. Compare `2` and `4`.
   - `4 != 2 + 1`, so there is a gap.
   - Because the array is sorted, no later `3` can appear between them.
   - The current run is complete: start `0`, end `2`.
   - Append `"0->2"`.
   - Start a new run at `4`.

```text
result = ["0->2"]
start = 4
```

4. Compare `4` and `5`.
   - `5 == 4 + 1`, so the run continues.
   - The open run is `[4, 5]`.

5. Compare `5` and `7`.
   - `7 != 5 + 1`, so the run `[4, 5]` is complete.
   - Append `"4->5"`.
   - Start a new run at `7`.

```text
result = ["0->2", "4->5"]
start = 7
```

The loop ends, but `7` is still an open run. Its start and end are both `7`, so append `"7"`.

Final answer:

```text
["0->2", "4->5", "7"]
```

### Code / Pseudocode
Python-style implementation:

```python
def summaryRanges(nums):
    if not nums:
        return []

    def format_range(start, end):
        if start == end:
            return str(start)
        return f"{start}->{end}"

    result = []
    start = nums[0]

    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            continue

        result.append(format_range(start, nums[i - 1]))
        start = nums[i]

    result.append(format_range(start, nums[-1]))
    return result
```

The helper `format_range` is not required, but it keeps the core scan focused on the run invariant instead of mixing formatting details into the control flow.

### Correctness
We prove that the algorithm returns exactly the required summary ranges.

First, every emitted range is consecutive. A range is emitted only from the current `start` to the previous array value. The algorithm keeps a run open only while each adjacent pair differs by exactly `1`. Therefore every value between `start` and the emitted end has appeared as part of one uninterrupted adjacent chain, so the emitted range is consecutive.

Second, every emitted range is maximal. The algorithm emits a range only when it finds a gap, meaning `nums[i] != nums[i - 1] + 1`, or when the array ends. In the gap case, the next value cannot belong to the current run. Since the array is sorted, no missing value can appear later to bridge the gap. In the end-of-array case, there is no later value that could extend the run. Thus every emitted range is as long as possible.

Third, every input number appears in exactly one emitted range. The algorithm starts the first run at `nums[0]`, extends the open run across consecutive adjacent values, and starts a new run exactly when the previous one is emitted. It never skips an element: each element either continues the current run or becomes the start of the next run. Since each completed run is appended once, each number is covered once.

Because the algorithm emits only consecutive maximal ranges and covers every input number exactly once, the returned list is exactly the summary ranges required by the problem.

### Complexity
- Time: `O(n)`, where `n` is the length of `nums`. Each element is inspected a constant number of times.
- Space: `O(n)` for the output list in the worst case, when no two numbers are consecutive. Excluding the required output, the extra working space is `O(1)`.

No sorting is needed because the problem already guarantees that `nums` is sorted.

### Common Pitfalls
- Forgetting the empty-array case. Accessing `nums[0]` before checking emptiness will fail for `[]`.
- Emitting a range too early. Do not append `"0->1"` when the next value might be `2`; wait until a gap or the end.
- Forgetting the final open run. The loop only emits when it sees a gap, so the last run must be appended after the loop.
- Formatting singleton ranges incorrectly. A one-number range is `"7"`, not `"7->7"`.
- Using a generic interval-merge approach. There are no interval objects to sort or merge here; the input is already a sorted list of points.
- Assuming positive numbers only. Negative values work the same way: `-2, -1, 0` is a valid consecutive run.

### First-Principles Summary
The problem reduces to detecting maximal consecutive runs in an already sorted list of unique numbers. A run can remain open as long as each new value is exactly one greater than the previous value. The moment that condition fails, sorted order proves the current run cannot be extended, so it is safe and necessary to emit it. The only state needed is the start of the open run and the previous value provided by the scan.

## Implementation
See `solutions/intervals/p228_summary_ranges.py`.

## Tests
See `tests/intervals/test_p228_summary_ranges.py`.

## Examples

### Example 1
- Input: `{'nums': [0, 1, 2, 4, 5, 7]}`
- Output: `['0->2', '4->5', '7']`

### Example 2
- Input: `{'nums': [0, 2, 3, 4, 6, 8, 9]}`
- Output: `['0', '2->4', '6', '8->9']`

## Follow-up Practice
- Trace an input with a single value, such as `[5]`.
- Trace an input where every value is isolated, such as `[1, 3, 5]`.
- Trace an input with negative values crossing zero, such as `[-2, -1, 0, 2]`.
- Explain why a gap between adjacent sorted values permanently closes the current range.
