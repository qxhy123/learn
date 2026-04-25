# 274. H-Index

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/h-index/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: sorting, counting

## First-Principles Explanation

### What The Problem Is Asking

You are given an array `citations`, where `citations[i]` is the number of citations received by the `i`-th paper of one researcher.

The **h-index** is the largest integer `h` such that:

```text
at least h papers have at least h citations each
```

The word **largest** matters. Many values may satisfy the condition, but the answer is the maximum satisfying value.

For example, if the citations are `[3, 0, 6, 1, 5]`, then:

- `h = 1` works because at least 1 paper has at least 1 citation.
- `h = 2` works because at least 2 papers have at least 2 citations.
- `h = 3` works because 3 papers have at least 3 citations: `3`, `6`, and `5`.
- `h = 4` does **not** work because only 2 papers have at least 4 citations: `6` and `5`.

So the answer is `3`.

The problem is not asking for the average citation count, the median citation count, or the number of highly cited papers by some fixed standard. It asks for the point where the required citation threshold and the required number of papers are the same number.

### The Baseline: Try Every Possible `h`

A direct first attempt is to test every candidate `h`.

If there are `n` papers, the h-index can never be greater than `n`, because having `h > n` would require more than `n` papers. Therefore the only candidates are:

```text
0, 1, 2, ..., n
```

For each candidate `h`, count how many papers have at least `h` citations. If that count is at least `h`, then `h` is valid.

Pseudocode:

```text
answer = 0
for h from 0 to n:
    count = 0
    for citations_of_one_paper in citations:
        if citations_of_one_paper >= h:
            count += 1

    if count >= h:
        answer = h

return answer
```

This is simple and correct, but it repeats a lot of counting work.

- There are `n + 1` possible candidate values of `h`.
- For each candidate, it scans all `n` papers.

So the baseline takes `O(n^2)` time and `O(1)` extra space.

That is enough to understand the definition, but we can do better by preserving the right information about citation counts.

### Key Observation

The exact citation count above `n` does not matter.

Suppose there are `n = 5` papers. For h-index purposes:

```text
5 citations, 6 citations, 100 citations, and 1,000,000 citations
```

all behave the same once we are testing candidates up to `5`: each paper counts as a paper with at least `5` citations.

This gives two important facts:

1. The answer is always between `0` and `n`.
2. Citation counts larger than `n` can be grouped into the same bucket as `n`.

Another useful way to view the condition is:

```text
candidate h is valid if count_of_papers_with_citations_at_least_h >= h
```

So the whole problem reduces to knowing, for each possible threshold `h`, how many papers meet or exceed that threshold.

### Two Natural Approaches

There are two common ways to compute the answer.

#### Approach 1: Sort The Citations

Sort citations in ascending order. At index `i`, there are `n - i` papers from `i` to the end, and all of them have at least `citations[i]` citations because the array is sorted.

For a candidate based at position `i`, the possible h-index contributed there is:

```text
min(citations[i], n - i)
```

Why `min`?

- `citations[i]` is the citation threshold available at this position.
- `n - i` is the number of papers with at least that many citations.
- The h-index cannot exceed either the citation threshold or the number of qualifying papers.

Scanning all sorted positions and taking the maximum gives the h-index.

This approach is easy to reason about and takes `O(n log n)` time due to sorting.

#### Approach 2: Count Buckets

Because the answer cannot exceed `n`, create `n + 1` buckets:

```text
bucket[k] = number of papers with exactly k citations, for 0 <= k < n
bucket[n] = number of papers with n or more citations
```

Then scan possible `h` values from `n` down to `0`, maintaining how many papers have at least the current threshold.

This gives an `O(n)` time solution.

The bucket approach is the most direct optimization of the brute force idea:

- Brute force asks, “For this `h`, how many papers have at least `h` citations?”
- Buckets answer that question incrementally from high thresholds down to low thresholds.

### Invariant And State

For the bucket solution, the state is:

```text
buckets[0..n]
qualified
```

where:

- `buckets[c]` stores how many papers have citation count `c`, except `buckets[n]` stores all papers with `n` or more citations.
- `qualified` stores how many papers have at least the current candidate `h` citations during the descending scan.

The invariant during the descending scan is:

```text
After adding buckets[h], qualified equals the number of papers with at least h citations.
```

That invariant is exactly the quantity needed by the h-index definition.

When scanning from high to low:

- Before processing `h`, `qualified` counts papers with citations greater than `h`.
- After adding `buckets[h]`, it also includes papers with exactly `h` citations.
- Therefore it now counts papers with at least `h` citations.

At that moment, `h` is valid if:

```text
qualified >= h
```

Because we scan from `n` down to `0`, the first valid `h` is automatically the largest valid `h`, so it is safe to return immediately.

### Detailed Algorithm

1. Let `n` be the number of papers.
2. Create an integer array `buckets` of length `n + 1`, initialized with zeros.
3. For each citation count `citation` in `citations`:
   - If `citation >= n`, increment `buckets[n]`.
   - Otherwise, increment `buckets[citation]`.
4. Set `qualified = 0`.
5. For `h` from `n` down to `0`:
   - Add `buckets[h]` to `qualified`.
   - Now `qualified` is the number of papers with at least `h` citations.
   - If `qualified >= h`, return `h`.
6. The loop will always return at least `0`, because every input has at least zero papers with at least zero citations.

### Pseudocode

```text
function hIndex(citations):
    n = length(citations)
    buckets = array of n + 1 zeroes

    for citation in citations:
        if citation >= n:
            buckets[n] += 1
        else:
            buckets[citation] += 1

    qualified = 0

    for h from n down to 0:
        qualified += buckets[h]

        if qualified >= h:
            return h
```

Python-style code:

```python
def hIndex(citations: list[int]) -> int:
    n = len(citations)
    buckets = [0] * (n + 1)

    for citation in citations:
        if citation >= n:
            buckets[n] += 1
        else:
            buckets[citation] += 1

    qualified = 0
    for h in range(n, -1, -1):
        qualified += buckets[h]
        if qualified >= h:
            return h

    return 0
```

The final `return 0` is mostly defensive in languages or styles that require an explicit return outside the loop. Mathematically, the loop must return by the time it reaches `h = 0`.

### Detailed Example Walkthrough

Use the first official example:

```text
citations = [3, 0, 6, 1, 5]
n = 5
```

The possible h-index values are only `0` through `5`.

Build buckets of length `6`:

```text
index:   0  1  2  3  4  5
bucket:  0  0  0  0  0  0
```

Process each citation:

- Citation `3` increments `buckets[3]`.
- Citation `0` increments `buckets[0]`.
- Citation `6` is at least `n`, so it increments `buckets[5]`.
- Citation `1` increments `buckets[1]`.
- Citation `5` is at least `n`, so it increments `buckets[5]`.

Now buckets are:

```text
index h:              0  1  2  3  4  5
papers in bucket h:   1  1  0  1  0  2
```

Interpretation:

- 1 paper has exactly 0 citations.
- 1 paper has exactly 1 citation.
- 0 papers have exactly 2 citations.
- 1 paper has exactly 3 citations.
- 0 papers have exactly 4 citations.
- 2 papers have at least 5 citations.

Now scan downward.

Start:

```text
qualified = 0
```

Candidate `h = 5`:

```text
qualified += buckets[5] = 2
qualified = 2
```

There are 2 papers with at least 5 citations. Need at least 5 such papers, so `5` is not valid.

Candidate `h = 4`:

```text
qualified += buckets[4] = 0
qualified = 2
```

There are 2 papers with at least 4 citations. Need at least 4 such papers, so `4` is not valid.

Candidate `h = 3`:

```text
qualified += buckets[3] = 1
qualified = 3
```

There are 3 papers with at least 3 citations. Need at least 3 such papers, so `3` is valid.

Because the scan is descending, `3` is the largest valid value. Return `3`.

### Second Example Walkthrough

Use the second official example:

```text
citations = [1, 3, 1]
n = 3
```

Build buckets of length `4`:

- Citation `1` increments `buckets[1]`.
- Citation `3` is at least `n`, so it increments `buckets[3]`.
- Citation `1` increments `buckets[1]` again.

Buckets:

```text
index h:              0  1  2  3
papers in bucket h:   0  2  0  1
```

Scan downward:

- `h = 3`: `qualified = 1`. Only 1 paper has at least 3 citations, so `3` is invalid.
- `h = 2`: `qualified = 1`. Only 1 paper has at least 2 citations, so `2` is invalid.
- `h = 1`: `qualified = 3`. Three papers have at least 1 citation, so `1` is valid.

Return `1`.

### Correctness

We prove the bucket algorithm returns the h-index.

#### Lemma 1: Buckets preserve all information relevant to the h-index.

The h-index is never greater than `n`, where `n` is the number of papers. Therefore, for any candidate `h`, only whether a paper has at least `h` citations matters, and `h <= n`.

Any citation count greater than or equal to `n` qualifies for every possible positive candidate up to `n`, so storing all such papers in `buckets[n]` does not change whether any candidate `h` is valid.

Thus the bucket array preserves exactly the citation-threshold information needed to evaluate every possible h-index candidate.

#### Lemma 2: During the descending scan, after processing bucket `h`, `qualified` equals the number of papers with at least `h` citations.

The scan starts above all candidates with `qualified = 0`, meaning no buckets have been included yet.

When the scan reaches a value `h`, all buckets for citation counts greater than `h` have already been added to `qualified`. Adding `buckets[h]` includes the papers with exactly `h` citations. Therefore, after this addition, `qualified` counts all papers with citation count at least `h`.

This proves the invariant.

#### Lemma 3: When the algorithm returns `h`, that `h` is valid.

The algorithm returns only when `qualified >= h`. By Lemma 2, `qualified` is the number of papers with at least `h` citations. Therefore at least `h` papers have at least `h` citations, so `h` satisfies the definition of h-index.

#### Lemma 4: The returned `h` is the largest valid value.

The algorithm checks candidates in descending order from `n` to `0`. If it returns `h`, then every candidate greater than `h` was checked earlier and failed the condition. Therefore no larger candidate is valid.

#### Theorem: The algorithm returns the h-index.

By Lemma 3, the returned value is valid. By Lemma 4, no larger value is valid. Therefore the returned value is exactly the largest valid h-index.

### Complexity

Let `n` be the number of papers.

- Time: `O(n)`
  - One pass builds the buckets.
  - One descending pass scans at most `n + 1` buckets.
- Space: `O(n)`
  - The bucket array has length `n + 1`.

The sorting approach is also valid, but its time complexity is `O(n log n)` and its extra space depends on the language and sort implementation.

### Common Pitfalls

- **Forgetting that h-index is capped by `n`**: A researcher with one paper cited 1000 times has h-index `1`, not `1000`.
- **Checking exact citation counts instead of at-least counts**: The condition is not “exactly `h` papers have exactly `h` citations.” It is “at least `h` papers have at least `h` citations.”
- **Scanning buckets upward and returning too early**: If you scan from low to high, the first valid value may not be the largest. Descending order lets you return immediately.
- **Not grouping large citation counts**: If `citation >= n`, it should go into `buckets[n]`; otherwise very large values would require an unnecessarily large array.
- **Off-by-one bucket size**: You need `n + 1` buckets so that index `n` exists.
- **Using `>` instead of `>=`**: A paper with exactly `h` citations qualifies for candidate `h`.
- **Misreading zero citations**: Papers with zero citations matter only for `h = 0`; they do not help any positive h-index.

### First-Principles Summary

The h-index asks for a balance point: the same number `h` is both the minimum citation threshold and the minimum number of papers that must reach that threshold.

The brute-force solution checks every possible balance point independently. The optimized bucket solution notices that the answer can only be between `0` and `n`, so citation counts above `n` can be collapsed into one bucket. Then it scans possible thresholds from high to low while maintaining one exact state variable: how many papers have at least the current threshold.

Once that count reaches or exceeds the threshold, the current threshold is valid. Since the scan moves from largest candidate to smallest, the first valid threshold is the h-index.

## Implementation

See `solutions/array_string/p274_h_index.py`.

## Tests

See `tests/array_string/test_p274_h_index.py`.

## Examples

### Example 1
- Input: `{'citations': [3, 0, 6, 1, 5]}`
- Output: `3`

### Example 2
- Input: `{'citations': [1, 3, 1]}`
- Output: `1`

## Follow-up Practice

- Implement both the sorting and bucket versions, then compare their complexity.
- Trace `buckets` and `qualified` on `[0]`, `[100]`, `[1, 1, 1]`, and `[0, 1, 4, 5, 6]`.
- Explain why values greater than `n` can be grouped into `buckets[n]` without changing the answer.
- Write a brute-force checker and compare it against the bucket solution on small random arrays.
