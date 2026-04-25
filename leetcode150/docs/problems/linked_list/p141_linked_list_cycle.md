# 141. Linked List Cycle

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/linked-list-cycle/
- Official Group: Linked List
- Pattern Group: Linked List
- Patterns: linked-list

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given the head of a singly linked list.

Each node has one outgoing pointer:

```text
node.next
```

That pointer either leads to another node or to `None`.

The question is:

> If we keep following `next` pointers starting from `head`, will we eventually revisit a node we have already seen?

If yes, the list has a cycle.

If no, the traversal eventually reaches `None`.

The input examples mention `pos`, but `pos` is not passed to the function on LeetCode. It is only a way to describe how the hidden test builder connects the tail:

```text
pos = -1  means the tail points to None
pos = 1   means the tail points back to the node at index 1
```

So the function does not need to find `pos`, return the cycle entry, or modify the list. It only returns a boolean:

```text
True  if a cycle exists
False if no cycle exists
```

### 2. What a Linked List Traversal Can Observe

In an array, we can jump to any index.

In a singly linked list, we cannot.

From a node, the only structural information available is:

```text
Where does next point?
```

Starting from `head`, the path is forced:

```text
head -> head.next -> head.next.next -> ...
```

There are only two possible outcomes.

#### Outcome A: The Path Ends


```text
a -> b -> c -> None
```

After enough `next` steps, the traversal reaches `None`.

That proves there is no cycle reachable from `head`.

#### Outcome B: The Path Loops

```text
a -> b -> c -> d
          ^    |
          |____|
```

After entering the loop, following `next` never reaches `None`.

Instead, the traversal keeps revisiting the same cycle nodes forever.

So the problem is fundamentally about distinguishing:

```text
eventually reaches None
vs.
eventually repeats a node
```

### 3. Brute-Force Baseline: Remember Every Visited Node

The most direct solution is to store the identity of every node we visit.

Algorithm:

1. Start at `head`.
2. If the current node is already in a set, we found a cycle.
3. Otherwise, add it to the set.
4. Move to `current.next`.
5. If we reach `None`, there is no cycle.

Pseudocode:

```python
def hasCycle(head):
    seen = set()
    current = head

    while current is not None:
        if current in seen:
            return True

        seen.add(current)
        current = current.next

    return False
```

This is easy to reason about:

- If a cycle exists, eventually we walk into a node that was already visited.
- If no cycle exists, each visited node is new until the traversal reaches `None`.

Complexity:

```text
Time:  O(n)
Space: O(n)
```

The time is good, but the extra space is avoidable.

The first-principles question becomes:

> Can we detect repetition without storing all visited nodes?

### 4. Key Observation: A Cycle Is a Track With No Exit

Imagine two runners moving along the linked list.

One runner moves one node at a time.

The other runner moves two nodes at a time.

Call them:

```text
slow = moves 1 step per round
fast = moves 2 steps per round
```

If the list has no cycle, the faster runner reaches the end first:

```text
fast == None
or
fast.next == None
```

That means there is no loop to keep it inside the list.

If the list has a cycle, both runners eventually enter the cycle. Once they are inside, the faster runner cannot escape because there is no `None` inside the cycle.

Inside the cycle, the faster runner gains one node on the slower runner every round:

```text
fast moves 2
slow moves 1
relative gain = 1
```

Because the cycle has a finite number of nodes, repeatedly gaining one position must eventually make the two runners land on the same node.

This is Floyd's cycle detection algorithm, often called the tortoise-and-hare algorithm.

### 5. The Fast/Slow Pointer Invariant

At the start of each loop iteration, maintain this invariant:

```text
slow and fast are nodes reachable from head by following next pointers.
fast has taken twice as many steps as slow.
If fast can still move two steps, the search can safely continue.
```

The guard must ensure the two-step move is valid:

```python
while fast is not None and fast.next is not None:
```

Then the update is:

```python
slow = slow.next
fast = fast.next.next
```

After the update, there are two cases:

```text
slow is fast  -> both pointers are on the same node, so a cycle exists
slow is not fast -> continue
```

The comparison must be node identity, not node value.

Two different nodes may store the same value, but that does not mean the list cycles. A cycle means the exact same node object is revisited.

### 6. Detailed Algorithm

Handle all cases with the same loop:

1. Initialize both pointers at `head`.
2. Continue only while `fast` can move two steps.
3. Move `slow` one step.
4. Move `fast` two steps.
5. If they point to the same node, return `True`.
6. If the loop stops, `fast` reached the end, so return `False`.

Code:

```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = head
        fast = head

        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next

            if slow is fast:
                return True

        return False
```

Equivalent pseudocode:

```text
slow = head
fast = head

while fast exists and fast.next exists:
    slow = slow.next
    fast = fast.next.next

    if slow and fast are the same node:
        return True

return False
```

### 7. Walkthrough: Example 1

Input description:

```text
head = [3, 2, 0, -4]
pos = 1
```

The values describe nodes:

```text
index:  0   1   2    3
value:  3   2   0   -4
```

Because `pos = 1`, the tail points back to the node at index `1`:

```text
3 -> 2 -> 0 -> -4
     ^         |
     |_________|
```

Start:

```text
slow = 3
fast = 3
```

#### Round 1

Move:

```text
slow = 2        one step from 3
fast = 0        two steps from 3
```

They are not the same node.

#### Round 2

Move:

```text
slow = 0        one step from 2
fast = 2        two steps from 0: 0 -> -4 -> 2
```

They are not the same node.

#### Round 3

Move:

```text
slow = -4       one step from 0
fast = -4       two steps from 2: 2 -> 0 -> -4
```

Now:

```text
slow is fast
```

Both pointers are on the exact same `-4` node, so the algorithm returns:

```text
True
```

### 8. Walkthrough: Example 3

Input description:

```text
head = [1]
pos = -1
```

The list is:

```text
1 -> None
```

Start:

```text
slow = 1
fast = 1
```

Before entering the loop, check:

```text
fast is not None       yes
fast.next is not None  no
```

The loop never runs because `fast` cannot move two steps.

Therefore the traversal has an end, and the algorithm returns:

```text
False
```

### 9. Why Meeting Proves a Cycle

If `slow` and `fast` meet after at least one move, then both pointers refer to the same node reached by following `next` from `head`.

Could this happen in an acyclic linked list?

No.

In an acyclic singly linked list, there is exactly one forward path and no way to go backward. Since `fast` always stays ahead of `slow` after the first move, it cannot loop around and collide with `slow`. The only thing ahead of it is the tail and then `None`.

So a meeting means `fast` must have wrapped around through a cycle.

Therefore returning `True` on a meeting is sound.

### 10. Why a Cycle Guarantees a Meeting

Suppose a cycle exists.

Eventually `slow` enters the cycle because it keeps following the only path from `head`.

`fast` also enters the cycle, unless it meets `slow` earlier. Once both are inside the cycle, neither can reach `None`.

Let the cycle length be `k`.

Measure the distance from `slow` to `fast` around the cycle as a number from `0` to `k - 1`.

After each round:

```text
slow moves 1
fast moves 2
```

So the distance changes by one position modulo `k`.

That sequence must eventually hit `0`:

```text
distance = 0
```

Distance `0` means both pointers are on the same node.

Therefore, if a reachable cycle exists, the algorithm must eventually return `True`.

### 11. Why Returning False Is Correct

The loop stops only when:

```text
fast is None
or
fast.next is None
```

That means the two-step runner has found the end of the list.

If any cycle were reachable from `head`, there would be no `None` after entering it.

So reaching the end proves that the reachable structure is acyclic.

Therefore returning `False` after the loop is correct.

### 12. Complexity

Let `n` be the number of distinct nodes reachable from `head` before either reaching `None` or repeating.

Time complexity:

```text
O(n)
```

In an acyclic list, `fast` reaches the tail after a linear number of pointer moves.

In a cyclic list, both pointers enter the cycle after a linear number of moves, and then meet after at most one full cycle length more.

Space complexity:

```text
O(1)
```

Only two pointers are stored, regardless of list size.

### 13. Common Pitfalls

#### Comparing Node Values Instead of Node Identity

Wrong idea:

```python
if slow.val == fast.val:
    return True
```

This is incorrect because different nodes can have equal values:

```text
1 -> 1 -> 1 -> None
```

That list has repeated values but no cycle.

Use identity:

```python
if slow is fast:
    return True
```

#### Starting With an Immediate Equality Check

Both pointers start at `head`, so this would be wrong:

```python
slow = head
fast = head

if slow is fast:
    return True
```

For any non-empty list, that check is true before movement. The pointers must move first, then compare.

#### Using an Unsafe Loop Guard

This can crash:

```python
while fast is not None:
    fast = fast.next.next
```

If `fast.next` is `None`, then `fast.next.next` is invalid.

The safe guard is:

```python
while fast is not None and fast.next is not None:
```

#### Thinking `pos` Is an Argument

In LeetCode's function signature, `pos` is not passed to `hasCycle`.

It is only used by the judge to create the linked structure before calling your function.

The algorithm should inspect pointers, not an index.

#### Modifying the List

Some approaches mark nodes by changing values or links.

That is unnecessary and risky. The fast/slow method detects the cycle without changing the input structure.

### 14. First-Principles Summary

This problem follows from a few basic facts:

```text
1. From any linked-list node, the next step is completely determined by node.next.
2. A traversal from head either reaches None or eventually repeats a node.
3. Repeating a node is exactly what it means for a reachable cycle to exist.
4. Remembering all visited nodes detects repetition but costs O(n) space.
5. Two pointers moving at different speeds detect repetition without memory.
6. If there is no cycle, the faster pointer reaches None.
7. If there is a cycle, the faster pointer gains one node per round and must meet the slower pointer inside the finite loop.
```

So the whole algorithm is:

> Move one pointer one step at a time and another pointer two steps at a time. If they meet, there is a cycle. If the fast pointer reaches the end, there is no cycle.

## Implementation
See `solutions/linked_list/p141_linked_list_cycle.py`.

## Tests
See `tests/linked_list/test_p141_linked_list_cycle.py`.

## Examples

### Example 1
- Input: `{'head': [3, 2, 0, -4], 'pos': 1}`
- Output: `True`

### Example 2
- Input: `{'head': [1, 2], 'pos': 0}`
- Output: `True`

### Example 3
- Input: `{'head': [1], 'pos': -1}`
- Output: `False`

## Follow-up Practice

- Draw the exact node arrows, not just values.
- Walk the fast and slow pointers round by round.
- Test `head = None`, a one-node acyclic list, and a one-node self-cycle.
- Practice explaining why the fast pointer must eventually catch the slow pointer inside a finite cycle.
