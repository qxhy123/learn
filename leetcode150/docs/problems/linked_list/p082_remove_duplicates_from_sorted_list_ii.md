# 82. Remove Duplicates from Sorted List II

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/
- Official Group: Linked List
- Pattern Group: Linked List
- Patterns: linked-list

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given the head of a **sorted** singly linked list.

Your job is not to keep one copy of each value. That would be LeetCode 83.

This problem asks for something stricter:

> Remove every value that appears more than once, and return only the values that appeared exactly once in the original list.

For example:

```text
1 -> 2 -> 3 -> 3 -> 4 -> 4 -> 5
```

The values `3` and `4` are duplicates, so all nodes with those values disappear:

```text
1 -> 2 -> 5
```

For another example:

```text
1 -> 1 -> 1 -> 2 -> 3
```

The value `1` appears more than once, so every `1` is removed:

```text
2 -> 3
```

The input is a linked list, so we cannot jump by index or easily delete a node by shifting later elements left. The only operation that changes the list structure is changing `next` pointers.

So the real problem is:

> While scanning a sorted linked list, connect together exactly the nodes whose values occur once, and skip every entire run of repeated values.

### 2. Why Sorted Order Matters

The sorted property is the reason this problem can be solved in one pass with constant extra space.

Because the list is sorted, equal values must appear next to each other:

```text
1 -> 2 -> 3 -> 3 -> 4 -> 4 -> 5
          -----     -----
```

That means duplicates form contiguous **runs**.

A run is a maximal group of nodes with the same value:

```text
1 -> 1 -> 1 -> 2 -> 3 -> 3 -> 4
---------    ---    -----    ---
 run 1       run 2  run 3    run 4
```

The answer keeps a run only if its length is exactly `1`.

So instead of thinking about individual nodes first, think about runs:

```text
If a value's run length is 1, keep that node.
If a value's run length is greater than 1, skip the whole run.
```

This is the core simplification created by sorting.

### 3. Start From the Brute Force Baseline

A very direct way to solve the problem is to count every value first.

For example:

```python
counts = {}
current = head

while current:
    counts[current.val] = counts.get(current.val, 0) + 1
    current = current.next
```

Then make a second pass and keep only nodes whose value has count `1`.

Conceptually:

```python
dummy = ListNode(0)
tail = dummy
current = head

while current:
    next_node = current.next
    if counts[current.val] == 1:
        tail.next = current
        tail = current
    current = next_node

tail.next = None
return dummy.next
```

This works because it directly implements the rule “keep values that appear once.”

But it uses extra memory:

```text
Time:  O(n)
Space: O(number of distinct values)
```

The question is whether the sorted order lets us avoid the hash map.

It does.

Since duplicates are adjacent, we do not need to remember counts for all values. At any moment, we only need to know whether the current run has one node or multiple nodes.

### 4. The Key Observation

Suppose we are standing at the first node of a run:

```text
current
  |
  v
3 -> 3 -> 3 -> 4 -> 5
```

Because the list is sorted, all nodes with value `3` are right here, consecutively.

So we can determine whether `3` should be kept by looking only at the next nodes:

```text
current.next has the same value? yes -> skip all 3s
current.next has a different value? no duplicate -> keep this 3
```

This creates a local decision:

```text
Look at one run.
Decide whether to keep it or remove it.
Move to the next run.
```

No value that appears later can suddenly become another `3`, because the list is sorted. Once we pass the run of `3`s, the decision about `3` is final.

### 5. Why a Dummy Node Is Useful

Deleting duplicate runs can change the head of the list.

For example:

```text
1 -> 1 -> 1 -> 2 -> 3
```

The original head belongs to a duplicate run and must be removed. The returned head should be `2`.

If we try to handle this without a dummy node, we need special cases for removing duplicates at the front.

A dummy node gives us a stable node before the real list:

```text
dummy -> 1 -> 1 -> 1 -> 2 -> 3
```

Now even if the first real nodes are removed, we can update:

```text
dummy.next = first node after removed run
```

At the end, the answer is always:

```python
return dummy.next
```

The dummy node is not part of the answer. It is a pointer-management tool that makes deleting the head no different from deleting any later run.

### 6. Pointer Roles and Invariant

Use two main pointers:

```text
prev
current
```

Their roles are different.

`prev` points to the last node in the already-processed answer prefix:

```text
dummy -> kept nodes ... -> prev
```

`current` points to the first unprocessed node:

```text
current -> remaining nodes not yet classified
```

The invariant before each iteration is:

```text
1. Every node before current has already been classified.
2. dummy.next through prev contains exactly the unique-value nodes from the processed prefix.
3. prev.next points to current, the first unprocessed node.
```

In picture form:

```text
dummy -> answer prefix -> prev -> current -> unknown suffix
```

The algorithm repeatedly examines the run starting at `current`.

There are two cases.

#### Case A: `current` Starts a Duplicate Run

If:

```python
current.next is not None and current.val == current.next.val
```

then `current.val` appears more than once.

Because the list is sorted, all nodes with that value must be consecutive. Skip them all:

```python
duplicate_value = current.val
while current and current.val == duplicate_value:
    current = current.next
```

Now `current` points to the first node after the duplicate run.

Reconnect `prev` around the removed run:

```python
prev.next = current
```

`prev` does not move, because no node from that run was kept.

#### Case B: `current` Is Unique

If `current` does not have a next node with the same value, then its run length is `1`.

So this node belongs in the answer.

Move both pointers forward:

```python
prev = current
current = current.next
```

Now the answer prefix has grown by one node, and the invariant is restored.

### 7. Detailed Algorithm

1. Create a dummy node whose `next` points to `head`.
2. Set `prev = dummy`.
3. Set `current = head`.
4. While `current` is not `None`:
   - If `current.next` exists and has the same value as `current`, then `current` begins a duplicate run.
   - Store that duplicate value.
   - Advance `current` until all nodes with that value are skipped.
   - Set `prev.next = current` to remove the entire duplicate run.
   - Otherwise, the current node is unique, so advance `prev` to `current` and advance `current` one step.
5. Return `dummy.next`.

The important distinction is:

```text
Unique run:      move prev forward.
Duplicate run:   do not move prev; change prev.next to skip the run.
```

### 8. Pseudocode

```python
def deleteDuplicates(head):
    dummy = ListNode(0)
    dummy.next = head

    prev = dummy
    current = head

    while current:
        if current.next and current.val == current.next.val:
            duplicate_value = current.val

            while current and current.val == duplicate_value:
                current = current.next

            prev.next = current
        else:
            prev = current
            current = current.next

    return dummy.next
```

Some implementations use `prev.next` as the scanning pointer instead of a separate `current`. That is also valid. The first-principles idea is the same: keep `prev` at the last confirmed answer node, and decide whether the next run should be attached or skipped.

### 9. Example Walkthrough: `[1, 2, 3, 3, 4, 4, 5]`

Initial list:

```text
dummy -> 1 -> 2 -> 3 -> 3 -> 4 -> 4 -> 5
prev = dummy
current = 1
```

#### Step 1: Value `1`

`current` is `1`.

The next value is `2`, so `1` is not duplicated.

Keep it:

```text
dummy -> 1 -> 2 -> 3 -> 3 -> 4 -> 4 -> 5
         prev
              current
```

#### Step 2: Value `2`

`current` is `2`.

The next value is `3`, so `2` is not duplicated.

Keep it:

```text
dummy -> 1 -> 2 -> 3 -> 3 -> 4 -> 4 -> 5
              prev
                   current
```

#### Step 3: Value `3`

`current` is the first `3`.

The next value is also `3`, so this is a duplicate run.

Skip all `3`s:

```text
dummy -> 1 -> 2 -> 3 -> 3 -> 4 -> 4 -> 5
              prev              current
```

Now reconnect:

```text
dummy -> 1 -> 2 -> 4 -> 4 -> 5
              prev    current
```

`prev` stays at `2`, because no `3` is kept.

#### Step 4: Value `4`

`current` is the first `4`.

The next value is also `4`, so this is another duplicate run.

Skip all `4`s:

```text
dummy -> 1 -> 2 -> 4 -> 4 -> 5
              prev         current
```

Reconnect:

```text
dummy -> 1 -> 2 -> 5
              prev    current
```

#### Step 5: Value `5`

`current` is `5`.

There is no next node, so `5` is unique.

Keep it:

```text
dummy -> 1 -> 2 -> 5
                   prev
                         current = None
```

Return:

```text
1 -> 2 -> 5
```

### 10. Example Walkthrough: `[1, 1, 1, 2, 3]`

Initial list:

```text
dummy -> 1 -> 1 -> 1 -> 2 -> 3
prev = dummy
current = first 1
```

The first `1` has another `1` after it, so `1` is duplicated.

Skip every `1`:

```text
dummy -> 1 -> 1 -> 1 -> 2 -> 3
prev                    current
```

Reconnect:

```text
dummy -> 2 -> 3
prev     current
```

Now `current` is `2`.

The next value is `3`, so `2` is unique. Keep it:

```text
dummy -> 2 -> 3
         prev
              current
```

Now `current` is `3`.

There is no next node, so `3` is unique. Keep it:

```text
dummy -> 2 -> 3
              prev
                   current = None
```

Return:

```text
2 -> 3
```

This example shows why the dummy node matters: the original head was removed, but `dummy.next` still gives the correct new head.

### 11. Correctness

We prove that the algorithm returns exactly the nodes whose values appear once in the original sorted list.

#### Invariant

Before each loop iteration:

```text
1. All nodes before current have been fully classified.
2. The list from dummy.next through prev contains exactly the nodes with unique values among the classified nodes.
3. prev.next points to current, the first unclassified node.
```

#### Initialization

Before the first iteration, no real node has been classified.

```text
dummy -> head
prev = dummy
current = head
```

The answer prefix is empty, which is correct for an empty classified prefix. Also, `prev.next` points to the first unclassified node. Therefore, the invariant holds initially.

#### Maintenance

At each iteration, `current` starts the next unclassified run.

If `current` has the same value as `current.next`, then the run length is at least `2`. Because the input list is sorted, every node with that value is consecutive in this run. The algorithm advances `current` past the entire run and sets `prev.next = current`. Therefore, all nodes with that duplicated value are excluded from the answer, which is exactly what the problem requires. `prev` remains the last kept node, and `prev.next` again points to the first unclassified node.

If `current` does not have the same value as `current.next`, then the run beginning at `current` has length exactly `1`. Because the list is sorted, the same value cannot appear later after a larger value. Therefore, `current` is a valid answer node. The algorithm advances `prev` to `current` and advances `current` to the next unclassified node. The answer prefix has gained exactly one correct node, and the invariant is restored.

#### Termination

The loop ends when `current` is `None`, so every node has been classified.

By the invariant, the list starting at `dummy.next` contains exactly the unique-value nodes from the entire original list, in their original order. That is precisely the required output.

### 12. Complexity

Let `n` be the number of nodes in the list.

Each node is visited a constant number of times. A node inside a duplicate run may be advanced over by the inner loop, but once passed, it is never processed again.

So the time complexity is:

```text
O(n)
```

The algorithm uses only a few pointers and one dummy node, regardless of input size.

So the auxiliary space complexity is:

```text
O(1)
```

### 13. Common Pitfalls

- Keeping one copy of a duplicated value. This problem removes **all** nodes whose value is duplicated.
- Moving `prev` after skipping a duplicate run. `prev` should stay at the last confirmed kept node.
- Forgetting the head can be deleted. Use a dummy node so duplicate runs at the front are handled naturally.
- Skipping only two duplicate nodes instead of the entire run. A value may appear three or more times.
- Checking `current.next.val` without first checking that `current.next` exists.
- Leaving old links attached after rebuilding a list in a different style. If you detach and append nodes manually, ensure the final tail points to `None`.
- Treating the problem like an array deduplication task. In a linked list, the operation is pointer rewiring, not overwriting positions.

### 14. First-Principles Summary

The sorted list turns duplicates into contiguous runs. The problem is therefore not “delete a node when it equals the previous node,” because that would leave one copy behind. The problem is “classify each run by its length.”

A dummy node gives a stable predecessor for the answer even when the original head must be removed. `prev` always marks the last node known to belong in the answer. `current` always marks the first unclassified node. For each run, either attach the single unique node by moving `prev`, or skip the whole duplicate run by changing `prev.next`.

That invariant is the entire solution:

```text
dummy -> unique processed prefix -> prev -> unclassified suffix
```

Once every run has been classified, `dummy.next` is the head of the list containing exactly the values that appeared once.

## Implementation
See `solutions/linked_list/p082_remove_duplicates_from_sorted_list_ii.py`.

## Tests
See `tests/linked_list/test_p082_remove_duplicates_from_sorted_list_ii.py`.

## Examples

### Example 1
- Input: `{'head': [1, 2, 3, 3, 4, 4, 5]}`
- Output: `[1, 2, 5]`

### Example 2
- Input: `{'head': [1, 1, 1, 2, 3]}`
- Output: `[2, 3]`

## Follow-up Practice
- Trace the algorithm on a list where every value is duplicated: `[1, 1, 2, 2, 3, 3]`.
- Trace the algorithm on a list with no duplicates: `[1, 2, 3, 4]`.
- Explain why `prev` moves only when the current run is unique.
- Explain why sorted order allows the algorithm to forget a value after its run ends.
