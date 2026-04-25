# 169. Majority Element

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/majority-element/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: Boyer-Moore, voting

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given an array `nums`, return the element that appears more than half the time.

More precisely, if:

```text
n = len(nums)
```

then the majority element is the value whose frequency is greater than:

```text
n / 2
```

The problem guarantees that such an element exists.

For example:

```text
nums = [2, 2, 1, 1, 1, 2, 2]
```

The value `2` appears `4` times.

The array length is `7`, and:

```text
7 / 2 = 3.5
```

Since `4 > 3.5`, the answer is:

```text
2
```

The important promise is that we do not need to decide whether a majority exists. We only need to find it.

So the real problem is:

> Find the one value that occurs so often that all other values combined occur fewer times than it does.

That last phrasing is the key to the optimal solution.

---

### 2. Start From the Brute Force Idea

The most direct approach is to count every value.

Use a hash map:

```text
count[value] = how many times value has appeared
```

Then scan the array:

```python
counts = {}

for value in nums:
    counts[value] = counts.get(value, 0) + 1
    if counts[value] > len(nums) // 2:
        return value
```

This is correct because it directly implements the definition of a majority element.

If a value appears more than `n / 2` times, the count map will eventually show that.

The cost is:

```text
Time:  O(n)
Space: O(n)
```

The time is already optimal, because any algorithm must inspect the input in the worst case. But the extra space is not obviously necessary.

The deeper question is:

> Can we identify the majority without remembering the exact count of every distinct value?

Because the problem guarantees that one value dominates all the others combined, we can.

---

### 3. The Key Observation: Pairwise Cancellation

Suppose the majority element is `M`.

By definition:

```text
count(M) > count(not M)
```

where `count(not M)` means the total number of elements that are not `M`.

Now imagine repeatedly doing this operation:

```text
Remove one M and one non-M.
```

This removes two different elements.

Does it change which value is the majority among the remaining elements?

No.

Why?

Because every cancellation removes:

```text
1 copy of M
1 copy of something else
```

The majority's lead over all non-majority elements stays positive.

For example, in:

```text
[2, 2, 1, 1, 1, 2, 2]
```

`2` appears `4` times, and non-`2` values appear `3` times.

If we cancel one `2` with one non-`2`, the remaining counts become:

```text
2:      3
not 2:  2
```

Cancel again:

```text
2:      2
not 2:  1
```

Cancel again:

```text
2:      1
not 2:  0
```

The majority cannot be completely canceled because it starts with more copies than everything else combined.

This is the first-principles reason Boyer-Moore voting works:

> If two different values disagree, we can pair them off and discard both. The true majority, if guaranteed to exist, survives all such cancellations.

---

### 4. Turning Cancellation Into State

We do not want to physically remove elements from the array.

Instead, we maintain two pieces of state:

```text
candidate = the value currently representing the uncanceled group
votes     = how many more times candidate has appeared than opposing values
```

Think of `votes` as the size of a simplified stack after cancellations.

When the next value equals `candidate`:

```text
It supports the current candidate.
Increase votes.
```

When the next value differs from `candidate`:

```text
It cancels one vote for the current candidate.
Decrease votes.
```

When `votes` becomes `0`, the processed prefix has been fully canceled:

```text
No value from that prefix has a remaining advantage inside our compressed state.
```

So the next value is allowed to become a new candidate.

This does not mean the old prefix was irrelevant in a careless way. It means the old prefix can be partitioned into pairs of different values, so it contributes no net vote to the final majority race.

---

### 5. The Invariant

After processing some prefix of the array, the algorithm's state represents the result of canceling pairs of different values inside that prefix.

The invariant is:

```text
The processed prefix can be reduced to votes copies of candidate,
after removing zero or more pairs of different values.
```

If `votes == 0`, the processed prefix can be completely reduced into pairs of different values.

If `votes > 0`, the only uncanceled value in the compressed representation is `candidate`, repeated `votes` times.

This invariant is exactly the information we need because the majority element cannot disappear under valid pairwise cancellation.

The state is small:

```text
candidate: one array value
votes:     one integer
```

No hash map is needed.

---

### 6. Detailed Algorithm

Initialize:

```text
candidate = None
votes = 0
```

Then scan each `value` in `nums`.

For each `value`:

1. If `votes == 0`, choose this value as the current candidate:

```text
candidate = value
```

2. If `value == candidate`, it supports the candidate:

```text
votes += 1
```

3. Otherwise, it opposes the candidate and cancels one candidate vote:

```text
votes -= 1
```

After the scan, return `candidate`.

Because the problem guarantees a majority element exists, the final surviving candidate must be that majority element.

In Python-like pseudocode:

```python
def majorityElement(nums):
    candidate = None
    votes = 0

    for value in nums:
        if votes == 0:
            candidate = value

        if value == candidate:
            votes += 1
        else:
            votes -= 1

    return candidate
```

A compact equivalent form is:

```python
def majorityElement(nums):
    candidate = None
    votes = 0

    for value in nums:
        if votes == 0:
            candidate = value
            votes = 1
        elif value == candidate:
            votes += 1
        else:
            votes -= 1

    return candidate
```

Both versions express the same cancellation logic.

---

### 7. Walkthrough: `nums = [3, 2, 3]`

Start:

```text
candidate = None
votes = 0
```

#### Read `3`

`votes` is `0`, so choose `3` as the candidate.

The current value equals the candidate, so add one vote:

```text
candidate = 3
votes = 1
```

Compressed meaning:

```text
[3] remains uncanceled
```

#### Read `2`

`2` is different from candidate `3`.

So `2` cancels one `3` vote:

```text
candidate = 3
votes = 0
```

Compressed meaning:

```text
[3, 2] can be canceled as one pair of different values
```

There is no current survivor from this prefix.

#### Read `3`

`votes` is `0`, so choose this `3` as the candidate.

Then add one vote:

```text
candidate = 3
votes = 1
```

End of scan.

Return:

```text
3
```

This matches the majority element.

---

### 8. Walkthrough: `nums = [2, 2, 1, 1, 1, 2, 2]`

Start:

```text
candidate = None
votes = 0
```

#### Read first `2`

Choose `2` because `votes == 0`:

```text
candidate = 2
votes = 1
```

#### Read second `2`

It matches the candidate:

```text
candidate = 2
votes = 2
```

#### Read first `1`

It differs from the candidate, so cancel one vote:

```text
candidate = 2
votes = 1
```

One `1` has canceled one `2`.

#### Read second `1`

It differs again:

```text
candidate = 2
votes = 0
```

At this point, the processed prefix:

```text
[2, 2, 1, 1]
```

can be fully paired away:

```text
(2, 1), (2, 1)
```

No value has a net advantage in the compressed state.

#### Read third `1`

Since `votes == 0`, choose `1` as the new candidate:

```text
candidate = 1
votes = 1
```

This can feel surprising because the final answer is not `1`.

That is okay. The candidate is only the current survivor after cancellations in the prefix, not a permanent claim.

#### Read third `2`

It differs from candidate `1`, so cancel:

```text
candidate = 1
votes = 0
```

The recent `1` and `2` pair off.

#### Read fourth `2`

Since `votes == 0`, choose `2`:

```text
candidate = 2
votes = 1
```

End of scan.

Return:

```text
2
```

Even though the candidate temporarily became `1`, the true majority `2` survived the complete cancellation process.

---

### 9. Why Returning the Final Candidate Is Correct

Let `M` be the majority element.

The algorithm simulates repeatedly canceling pairs of different values.

Each decrement of `votes` corresponds to pairing one occurrence of the current candidate with one different value. Each time `votes` reaches `0`, the processed block represented by that candidate has been fully canceled into different-value pairs.

Now consider what cancellation can do to `M`.

Since `M` appears more than half the time:

```text
count(M) > count(all other values combined)
```

Every time an `M` is canceled, it must be canceled with a non-`M`.

But there are fewer non-`M` elements than `M` elements. Therefore, after all possible different-value cancellations, at least one `M` must remain.

The algorithm's final candidate is the value that remains after this cancellation process.

Because some `M` must remain, and the compressed final state can contain only one uncanceled value, that value must be `M`.

Therefore, the algorithm returns the majority element.

---

### 10. A More Formal Correctness Argument

We prove that the algorithm returns the majority element.

#### Lemma 1: The state represents pairwise cancellation

After processing each prefix of `nums`, the processed prefix can be transformed into `votes` copies of `candidate` by deleting pairs of different values.

Proof by induction over the scan:

- Before reading any values, `votes = 0`, so the empty prefix is represented correctly.
- If `votes == 0`, starting a new candidate with the current value leaves one uncanceled copy of that value.
- If the current value equals `candidate`, it adds one more uncanceled copy, so `votes` increases.
- If the current value differs from `candidate`, it can be paired with one uncanceled copy of `candidate`, so `votes` decreases.

Thus the invariant holds after every step.

#### Lemma 2: Pairwise cancellation cannot remove the true majority completely

Let `M` be the majority element.

There are more copies of `M` than copies of all other values combined.

Each canceled pair can remove at most one copy of `M`, and removing one copy of `M` requires also removing one non-`M` value.

Since there are not enough non-`M` values to pair with all copies of `M`, at least one copy of `M` remains after all possible different-value cancellations.

#### Theorem: The returned candidate is the majority element

By Lemma 1, the final algorithm state is a valid result of deleting different-value pairs from the full array.

By Lemma 2, the true majority element `M` must still remain after such cancellations.

The algorithm's final compressed state has only one possible remaining value: `candidate`.

Therefore:

```text
candidate == M
```

So the algorithm returns the majority element.

---

### 11. Complexity

The algorithm scans the array once.

Each element causes only constant work:

```text
compare
possibly assign candidate
increment or decrement votes
```

So the time complexity is:

```text
O(n)
```

The algorithm stores only two variables:

```text
candidate
votes
```

So the extra space complexity is:

```text
O(1)
```

This is better than the hash-map baseline, which also uses `O(n)` time but may use `O(n)` extra space.

---

### 12. Common Pitfalls

#### Pitfall 1: Forgetting that the majority is guaranteed

Boyer-Moore returns a candidate.

If the problem did not guarantee a majority element, the final candidate would need a second pass to verify its count:

```python
if nums.count(candidate) > len(nums) // 2:
    return candidate
```

For this LeetCode problem, the guarantee makes that verification unnecessary.

#### Pitfall 2: Treating `votes` as the actual frequency

`votes` is not the total number of times `candidate` appears.

It is the candidate's net advantage after cancellations in the processed prefix.

For example, after reading:

```text
[2, 2, 1]
```

we may have:

```text
candidate = 2
votes = 1
```

But `2` has appeared twice. The vote count is lower because one `2` was canceled by one `1`.

#### Pitfall 3: Thinking the candidate can never change

The candidate can change many times.

That is safe because when `votes` reaches `0`, the represented prefix has no uncanceled survivor. The next value starts a new unresolved block.

A temporary candidate is not a final answer until the scan is complete.

#### Pitfall 4: Updating in the wrong order

This pattern is safest when written as:

```python
if votes == 0:
    candidate = value

if value == candidate:
    votes += 1
else:
    votes -= 1
```

or as the explicit `if / elif / else` version.

Be careful not to decrement immediately after resetting the candidate.

#### Pitfall 5: Using sorting when constant space is expected

Sorting works because the majority element must occupy the middle index after sorting:

```python
return sorted(nums)[len(nums) // 2]
```

But sorting costs:

```text
O(n log n) time
```

and may use extra memory depending on the language/runtime.

Boyer-Moore gives the optimal one-pass constant-space solution.

---

### 13. First-Principles Summary

The majority element is stronger than every other value combined.

That means different values can be safely paired off and removed:

```text
one majority value + one non-majority value
```

Such cancellations cannot eliminate the true majority because it has more copies than all opponents together.

Boyer-Moore voting is just an efficient way to simulate this cancellation without storing the pairs:

```text
candidate = current uncanceled value
votes     = its net uncanceled count
```

When values match the candidate, the candidate gains support.

When values differ, one unit of support is canceled.

When support falls to zero, the processed block has canceled itself out, and the next value can start a new block.

Because a majority element is guaranteed to exist, the final survivor must be that majority element.

## Implementation

See `solutions/array_string/p169_majority_element.py`.

## Tests

See `tests/array_string/test_p169_majority_element.py`.

## Examples

### Example 1
- Input: `{'nums': [3, 2, 3]}`
- Output: `3`

### Example 2
- Input: `{'nums': [2, 2, 1, 1, 1, 2, 2]}`
- Output: `2`

## Follow-up Practice

- Trace `candidate` and `votes` after every element.
- Compare the hash-map baseline with Boyer-Moore voting.
- Add a second verification pass and see how the algorithm changes if the majority guarantee is removed.
