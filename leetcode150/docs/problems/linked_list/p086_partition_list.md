# 86. Partition List

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/partition-list/
- Official Group: Linked List
- Pattern Group: Linked List
- Patterns: linked-list

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given the head of a singly linked list and an integer `x`, rearrange the list so that:

```text
all nodes with value < x come before all nodes with value >= x
```

The important detail is that the relative order inside each group must stay the same.

For example:

```text
head = 1 -> 4 -> 3 -> 2 -> 5 -> 2
x = 3
```

The nodes whose values are less than `3` are encountered in this order:

```text
1, 2, 2
```

The nodes whose values are greater than or equal to `3` are encountered in this order:

```text
4, 3, 5
```

So the required answer is:

```text
1 -> 2 -> 2 -> 4 -> 3 -> 5
```

This is not asking us to sort the list. In a sorted list, `3` would come before `4`, and `2`s might be arranged only by value. Here, we are only making one boundary around `x`:

```text
< x region | >= x region
```

Within each region, the original order is preserved.

That makes the problem a stable partition of a linked list.

### 2. Start From the Brute-Force Baseline

If the input were an array, the most direct stable approach would be:

1. Make one pass and collect every value `< x`.
2. Make another pass and collect every value `>= x`.
3. Concatenate the two collected lists.

For a linked list, a similarly simple approach is to create new nodes:

```python
before_values = []
after_values = []

current = head
while current:
    if current.val < x:
        before_values.append(current.val)
    else:
        after_values.append(current.val)
    current = current.next

build a new linked list from before_values + after_values
```

This captures the right idea, but it is not ideal:

- It uses `O(n)` extra storage.
- It creates a new list instead of rearranging the existing nodes.
- It throws away the main advantage of linked lists: links can be changed without moving node contents.

The brute-force baseline teaches us the desired shape of the answer:

```text
stable list of small nodes + stable list of large nodes
```

The optimized solution keeps exactly that shape, but builds the two lists directly with pointers.

### 3. The Key Observation

Each node belongs to exactly one of two groups:

```text
node.val < x      -> before list
node.val >= x     -> after list
```

There is no interaction between nodes beyond preserving their encounter order.

So when we scan the original list from left to right, we can immediately append the current node to the tail of the correct group.

Because appending to the tail preserves order, the first node assigned to a group stays before the second node assigned to that group, and so on.

This is the core first-principles reduction:

> Stable partition does not require sorting, searching, or repeated insertion. It only requires maintaining two output chains in input order, then connecting them.

### 4. Why Two Lists Are Cleaner Than In-Place Insertion

A tempting approach is to walk the list and move every `< x` node toward the front.

That sounds direct, but it creates several pointer hazards:

- What if the head itself must change?
- What if a small node appears after many large nodes?
- What if moving a node causes us to lose the rest of the list?
- What if the insertion point and current pointer overlap?

For example:

```text
4 -> 1 -> 3 -> 2, x = 3
```

When seeing `1`, we need to remove it from after `4` and move it to the front. Later, when seeing `2`, we need to remove it from after `3` and insert it after `1`. This can be done, but the code has to carefully track previous nodes and head changes.

The two-list method avoids that complexity. Instead of moving nodes backward into the already-scanned part, every node moves forward into one of two new chains:

```text
before chain: nodes < x, in original order
after chain:  nodes >= x, in original order
```

At the end:

```text
before_tail.next = after_head
```

This is still an in-place linked-list rearrangement because we reuse the original nodes. We are not copying values into new nodes.

### 5. The Two-List Pointer Invariant

Use two dummy nodes:

```text
before_dummy -> start of nodes with value < x
after_dummy  -> start of nodes with value >= x
```

And two tails:

```text
before_tail = last node in the before chain
after_tail  = last node in the after chain
```

After processing some prefix of the original list, maintain this invariant:

```text
before_dummy.next ... before_tail
    contains exactly the processed nodes with value < x,
    in their original order.

after_dummy.next ... after_tail
    contains exactly the processed nodes with value >= x,
    in their original order.

Every unprocessed node is still reachable from current.
```

The local decision for each node is simple:

```text
if current.val < x:
    append current to before chain
else:
    append current to after chain
```

Appending means:

```text
tail.next = current
tail = current
```

The invariant is powerful because it says the partially built answer is already correct for everything we have processed. The remaining work is just the same problem on the unprocessed suffix.

### 6. The Important Safety Detail: Save `next`

When using original nodes, each node already has a `next` pointer from the old list.

Before rewiring `current`, save the next node:

```python
next_node = current.next
```

Then append `current` to the correct chain and move to `next_node`.

Many implementations do not explicitly detach each node during the loop, but the final line must terminate the `after` chain:

```python
after_tail.next = None
```

Why is that necessary?

Because the old `next` pointers may still point into the original list. If the final node in the `after` chain used to point to a node that was moved into the `before` chain, failing to clear `after_tail.next` can create a cycle or attach extra nodes after the answer.

Example:

```text
2 -> 1, x = 2
```

Processing creates:

```text
before: 1
after:  2 -> 1   # old pointer still exists temporarily
```

If we connect `before` to `after` without cutting the after tail, we can get:

```text
1 -> 2 -> 1 -> 2 -> ...
```

So the final termination is not cosmetic. It is what prevents stale links from leaking into the output.

### 7. Detailed Algorithm

1. Create `before_dummy` and `after_dummy`.
2. Set `before_tail = before_dummy` and `after_tail = after_dummy`.
3. Scan the original list with `current`.
4. For each node:
   - Save `next_node = current.next`.
   - If `current.val < x`, append it after `before_tail` and advance `before_tail`.
   - Otherwise, append it after `after_tail` and advance `after_tail`.
   - Move `current = next_node`.
5. Cut off the end of the `after` chain with `after_tail.next = None`.
6. Connect the two chains with `before_tail.next = after_dummy.next`.
7. Return `before_dummy.next`.

The dummy nodes are not part of the answer. They exist only to make empty-chain cases easy.

For example, if no node is `< x`, then `before_dummy.next` stays empty until we connect it to the `after` chain. If every node is `< x`, then `after_dummy.next` is `None`, and connecting the before tail to it naturally ends the list.

### 8. Code

```python
class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        before_dummy = ListNode(0)
        after_dummy = ListNode(0)

        before_tail = before_dummy
        after_tail = after_dummy

        current = head
        while current:
            next_node = current.next

            if current.val < x:
                before_tail.next = current
                before_tail = current
            else:
                after_tail.next = current
                after_tail = current

            current = next_node

        after_tail.next = None
        before_tail.next = after_dummy.next

        return before_dummy.next
```

If writing in an environment where `ListNode` is already provided by LeetCode, the dummy construction uses that provided class.

### 9. Walkthrough: `1 -> 4 -> 3 -> 2 -> 5 -> 2`, `x = 3`

Start:

```text
before: empty
after:  empty
current = 1
```

Process `1`:

```text
1 < 3, append to before

before: 1
after:  empty
```

Process `4`:

```text
4 >= 3, append to after

before: 1
after:  4
```

Process `3`:

```text
3 >= 3, append to after

before: 1
after:  4 -> 3
```

Notice that `3` is not less than `x`, so it belongs in the second group.

Process `2`:

```text
2 < 3, append to before

before: 1 -> 2
after:  4 -> 3
```

Process `5`:

```text
5 >= 3, append to after

before: 1 -> 2
after:  4 -> 3 -> 5
```

Process final `2`:

```text
2 < 3, append to before

before: 1 -> 2 -> 2
after:  4 -> 3 -> 5
```

Terminate the after chain:

```text
after: 4 -> 3 -> 5 -> None
```

Connect before to after:

```text
1 -> 2 -> 2 -> 4 -> 3 -> 5
```

This preserves the original order of the small nodes `1, 2, 2` and the original order of the large nodes `4, 3, 5`.

### 10. Walkthrough: `2 -> 1`, `x = 2`

Start:

```text
before: empty
after:  empty
```

Process `2`:

```text
2 >= 2, append to after

before: empty
after:  2
```

Process `1`:

```text
1 < 2, append to before

before: 1
after:  2
```

Now connect:

```text
1 -> 2
```

This example is small, but it exposes the main danger: the original node `2` used to point to `1`. We must ensure the final `after_tail.next` is `None`, otherwise the old link can corrupt the final structure.

### 11. Correctness

We prove that the algorithm returns exactly the required partitioned list.

#### Lemma 1: The before chain contains exactly the processed nodes whose values are `< x`, in original order.

Initially, no nodes have been processed, and the before chain is empty, so the claim is true.

When processing a node:

- If its value is `< x`, the algorithm appends it to the end of the before chain.
- If its value is `>= x`, the before chain is unchanged.

Because nodes are scanned in original order and new qualifying nodes are always appended at the tail, their relative order in the before chain is the same as in the input.

Therefore, after every iteration, the before chain contains exactly the processed `< x` nodes in original order.

#### Lemma 2: The after chain contains exactly the processed nodes whose values are `>= x`, in original order.

The proof is symmetric to Lemma 1.

Initially, the after chain is empty. When processing a node with value `>= x`, the algorithm appends it to the after tail. Nodes with value `< x` do not change the after chain. Since scanning is left to right and appending is always at the tail, the after chain preserves the original order of all processed `>= x` nodes.

#### Lemma 3: Every input node appears in exactly one of the two chains.

For each processed node, exactly one of these conditions is true:

```text
node.val < x
node.val >= x
```

So each node is appended to exactly one chain. The scan advances through every original node once, using the saved `next_node` to continue even after links are rewired.

Thus, after the loop, every original node appears in exactly one chain.

#### Lemma 4: The final list has no stale suffix or cycle from old links.

After all nodes are distributed, the algorithm sets:

```python
after_tail.next = None
```

This removes any old outgoing pointer from the last node of the after chain. Then it sets:

```python
before_tail.next = after_dummy.next
```

So the final list is the before chain followed by the after chain, and the after chain ends at `None`.

#### Theorem: The returned list is the required partition of the input list.

By Lemma 1, the first part of the returned list contains exactly all nodes with values `< x`, in original order. By Lemma 2, the second part contains exactly all nodes with values `>= x`, in original order. By Lemma 3, no node is missing or duplicated. By Lemma 4, the final links form a proper terminated list.

Therefore, the returned list satisfies the problem requirements.

### 12. Complexity

Let `n` be the number of nodes in the list.

- Time: `O(n)`, because each node is visited once and appended once.
- Auxiliary space: `O(1)`, because the algorithm uses a constant number of pointers and dummy nodes.

The output reuses the original nodes. The dummy nodes are helper sentinels, not storage proportional to the input size.

### 13. Common Pitfalls

- **Sorting the list**: The problem asks for a partition around `x`, not a full sort.
- **Putting `x` in the wrong group**: Nodes equal to `x` belong in the `>= x` group.
- **Losing the next node**: Save `next_node = current.next` before changing links if your implementation detaches or rewires aggressively.
- **Forgetting to terminate the after chain**: `after_tail.next = None` prevents old links from creating cycles or extra suffixes.
- **Breaking stability**: Prepending nodes to a chain reverses their order. Append to the tail instead.
- **Overcomplicating head changes**: Dummy nodes avoid special cases when the first real answer node is not the original head.
- **Creating new value nodes unnecessarily**: The natural linked-list solution reuses nodes and changes `next` pointers.

### 14. First-Principles Summary

The required output has only one structural rule:

```text
all < x nodes first, all >= x nodes second
```

And one stability rule:

```text
inside each group, preserve original order
```

Those two rules imply the solution:

1. Scan the original list once.
2. Append each node to the tail of the group it belongs to.
3. Join the two groups.
4. Terminate the final list cleanly.

The invariant is the whole algorithm:

```text
processed small nodes are already correct in before
processed large nodes are already correct in after
unprocessed nodes remain reachable from current
```

When the scan ends, there is no hidden work left. The before chain is exactly the first half of the answer, the after chain is exactly the second half, and connecting them gives the required stable partition.

## Implementation
See `solutions/linked_list/p086_partition_list.py`.

## Tests
See `tests/linked_list/test_p086_partition_list.py`.

## Examples

### Example 1
- Input: `{'head': [1, 4, 3, 2, 5, 2], 'x': 3}`
- Output: `[1, 2, 2, 4, 3, 5]`

### Example 2
- Input: `{'head': [2, 1], 'x': 2}`
- Output: `[1, 2]`

## Follow-up Practice

- Trace `2 -> 1` and explain why `after_tail.next = None` is required.
- Try inputs where all nodes are `< x` and where no nodes are `< x`.
- Explain why appending to tails preserves stability but prepending does not.
