# 138. Copy List with Random Pointer

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/copy-list-with-random-pointer/
- Official Group: Linked List
- Pattern Group: Linked List
- Patterns: linked-list

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

We are given the head of a linked list where each node has two pointers:

```text
next   -> the usual pointer to the next node in list order
random -> a pointer to any node in the same list, or null
```

The task is to return the head of a **deep copy** of the entire list.

That means every original node must get a brand-new cloned node with the same value, and the cloned nodes must reproduce the same pointer structure:

```text
original.next   points to original neighbor
clone.next      must point to that neighbor's clone

original.random points to some original target
clone.random    must point to that target's clone
```

The output list must not reuse original nodes. If the original list is later mutated, the copied list should still be independent.

So the problem is not just "copy node values." It is:

> Recreate a graph-like pointer structure whose nodes happen to be arranged by a linked-list `next` chain.

The `next` pointers give us a convenient way to visit all nodes. The `random` pointers are the hard part because they may point forward, backward, to the same node, or to `null`.

### 2. Why a Simple One-Pass Copy Is Not Enough

For an ordinary singly linked list, copying is easy:

```python
clone.val = original.val
clone.next = copy(original.next)
```

But here, while creating a clone, we may need to assign its `random` pointer to a node whose clone does not exist yet.

For example:

```text
original: A -> B -> C
random:   A.random = C
```

When we are standing at `A`, the clone of `C` might not have been created yet if we copy from left to right.

This is the first key constraint:

> A copied pointer cannot point to an original node; it must point to the clone of that original node.

Therefore, whenever we see an original node `x`, we need a reliable way to answer:

```text
where is x's clone?
```

The entire problem reduces to maintaining that original-to-clone relationship.

### 3. Brute-Force Baseline

A direct but inefficient baseline is:

1. Traverse the original list and create a cloned `next` chain with the same values.
2. For each original node, find which original node its `random` points to.
3. Find the corresponding cloned node at the same position in the copied list.
4. Assign the cloned node's `random` pointer.

Conceptually:

```python
original_nodes = all nodes in original list
clone_nodes = all nodes in copied list

for i, original in enumerate(original_nodes):
    if original.random is None:
        clone_nodes[i].random = None
    else:
        j = index where original_nodes[j] is original.random
        clone_nodes[i].random = clone_nodes[j]
```

This is correct, but if we search linearly for each `random` target, the random-pointer assignment can cost `O(n^2)` time.

The repeated work is obvious:

> We keep rediscovering the same mapping from original nodes to their copied nodes.

So the improvement is to store that mapping explicitly, or to encode it temporarily inside the linked list itself.

### 4. Key Observation: Random Pointers Need Identity, Not Value

Node values are not enough to identify random targets.

Two different nodes may have the same value:

```text
[3] -> [3] -> [3]
```

If a `random` pointer points to the second `3`, the clone must point to the clone of that exact second node, not just any node whose value is `3`.

So the mapping must be based on node identity:

```text
original node object -> cloned node object
```

This gives the central invariant for the standard hash-map solution:

```text
For every original node already discovered,
clone_of[original] is the unique clone node with the same value.
```

Once this invariant is true for every node, pointer assignment is mechanical:

```text
clone_of[original].next   = clone_of[original.next]
clone_of[original].random = clone_of[original.random]
```

with `None` mapped to `None`.

### 5. Approach 1: Hash Map From Original Nodes to Clones

The cleanest solution uses two passes.

#### Pass 1: Create all clone nodes


Walk through the original `next` chain.

For each original node, create one cloned node with the same value and store it:

```text
clone_of[original] = Node(original.val)
```

After this pass, every original node has a clone, but the clone pointers may not be connected yet.

The invariant is:

```text
Every original node reachable from head has exactly one clone in clone_of.
```

#### Pass 2: Wire copied pointers

Walk through the original list again.

For each original node:

```python
clone = clone_of[original]
clone.next = clone_of.get(original.next)
clone.random = clone_of.get(original.random)
```

This works because `original.next` and `original.random`, if non-null, are also original nodes whose clones were created in pass 1.

#### Python-style code

LeetCode provides a `Node` class similar to:

```python
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
```

Then the hash-map implementation is:

```python
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if head is None:
            return None

        clone_of = {}

        current = head
        while current:
            clone_of[current] = Node(current.val)
            current = current.next

        current = head
        while current:
            clone = clone_of[current]
            clone.next = clone_of.get(current.next)
            clone.random = clone_of.get(current.random)
            current = current.next

        return clone_of[head]
```

This is usually the most readable solution and is the best one to write first in an interview unless constant auxiliary space is required.

### 6. Approach 2: Interleaving Clones Inside the Original List

There is also an `O(1)` auxiliary-space solution. It avoids a hash map by placing each clone immediately after its original node.

Instead of storing:

```text
clone_of[original]
```

we make the list itself encode the mapping:

```text
original.next is original's clone
```

For an original list:

```text
A -> B -> C -> None
```

we first transform it into:

```text
A -> A' -> B -> B' -> C -> C' -> None
```

Now each original node's clone can be found in constant time:

```text
A' = A.next
B' = B.next
C' = C.next
```

This is the interleaving invariant:

```text
For every original node x, x.next is x's clone,
and x.next.next is the next original node.
```

Once this invariant holds, random pointers become easy. If:

```text
x.random = y
```

then `y`'s clone is immediately after `y`:

```text
y clone = y.next
```

Therefore:

```text
x clone random = x.random.next
```

or in code:

```python
x.next.random = x.random.next
```

when `x.random` is not `None`.

### 7. Detailed Interleaving Algorithm

The interleaving solution has three passes.

#### Pass 1: Insert each clone after its original

For each original node `current`:

```text
before:
current -> next_original

after:
current -> current_clone -> next_original
```

Code:

```python
current = head
while current:
    next_original = current.next
    clone = Node(current.val)
    current.next = clone
    clone.next = next_original
    current = next_original
```

The important safety detail is saving `next_original` before changing `current.next`. Otherwise we lose the rest of the original list.

#### Pass 2: Assign clone random pointers

Now each original node `current` is followed by its clone `current.next`.

If `current.random` is `None`, the clone's random is also `None`.

If `current.random` points to some original node `target`, then `target.next` is `target`'s clone.

Code:

```python
current = head
while current:
    clone = current.next
    if current.random:
        clone.random = current.random.next
    current = clone.next
```

The step `current = clone.next` skips from one original node to the next original node.

#### Pass 3: Separate the two lists

At this point the mixed list contains both originals and clones:

```text
A -> A' -> B -> B' -> C -> C'
```

We must restore the original list:

```text
A -> B -> C
```

and extract the copied list:

```text
A' -> B' -> C'
```

For each original node:

```text
clone = original.next
next_original = clone.next

original.next = next_original
clone.next = next_original.next if next_original else None
```

Code:

```python
current = head
copy_head = head.next

while current:
    clone = current.next
    next_original = clone.next

    current.next = next_original
    clone.next = next_original.next if next_original else None

    current = next_original

return copy_head
```

Full implementation:

```python
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if head is None:
            return None

        current = head
        while current:
            next_original = current.next
            clone = Node(current.val)
            current.next = clone
            clone.next = next_original
            current = next_original

        current = head
        while current:
            clone = current.next
            if current.random:
                clone.random = current.random.next
            current = clone.next

        current = head
        copy_head = head.next
        while current:
            clone = current.next
            next_original = clone.next

            current.next = next_original
            clone.next = next_original.next if next_original else None

            current = next_original

        return copy_head
```

### 8. Detailed Example Walkthrough

Use Example 1:

```text
Input: [[7, null], [13, 0], [11, 4], [10, 2], [1, 0]]
```

This means the list order is:

```text
index:  0   1    2    3   4
value:  7   13   11   10  1
next:   0 -> 1 -> 2 -> 3 -> 4
```

The random pointers are:

```text
node 0 value 7  random -> null
node 1 value 13 random -> node 0 value 7
node 2 value 11 random -> node 4 value 1
node 3 value 10 random -> node 2 value 11
node 4 value 1  random -> node 0 value 7
```

Name the original nodes by position:

```text
A(7) -> B(13) -> C(11) -> D(10) -> E(1)
```

with random pointers:

```text
A.random = null
B.random = A
C.random = E
D.random = C
E.random = A
```

#### After pass 1: interleave clones

Create one clone after each original:

```text
A -> A' -> B -> B' -> C -> C' -> D -> D' -> E -> E'
```

Now the mapping is stored in the list:

```text
A.next = A'
B.next = B'
C.next = C'
D.next = D'
E.next = E'
```

#### During pass 2: assign random pointers

For `A`:

```text
A.random = null
A'.random = null
```

For `B`:

```text
B.random = A
A's clone is A.next = A'
B'.random = A'
```

For `C`:

```text
C.random = E
E's clone is E.next = E'
C'.random = E'
```

For `D`:

```text
D.random = C
C's clone is C.next = C'
D'.random = C'
```

For `E`:

```text
E.random = A
A's clone is A.next = A'
E'.random = A'
```

#### After pass 3: detach clones

Restore the original list:

```text
A -> B -> C -> D -> E
```

Extract the copied list:

```text
A' -> B' -> C' -> D' -> E'
```

The copied list has the same values and random-index structure:

```text
[[7, null], [13, 0], [11, 4], [10, 2], [1, 0]]
```

But every node in the copied list is new.

### 9. Correctness Argument

We prove the interleaving algorithm returns a deep copy of the original list.

#### Lemma 1: After pass 1, every original node has exactly one clone immediately after it.

For each original node `x`, pass 1 creates a new node `x'` with `x'.val == x.val`, rewires `x.next` to `x'`, and sets `x'.next` to the original successor of `x`.

The traversal then moves to that saved original successor, so each original node is processed once. Therefore every original node has exactly one clone immediately after it.

#### Lemma 2: After pass 2, every clone has the correct `random` pointer.

Consider any original node `x` and its clone `x' = x.next`.

If `x.random` is `None`, the algorithm leaves `x'.random` as `None`, which is correct.

If `x.random` is some original node `y`, Lemma 1 says `y.next` is exactly `y`'s clone. The algorithm assigns:

```text
x'.random = x.random.next = y.next = y'
```

So `x'` points to the clone of the same target that `x` points to.

#### Lemma 3: After pass 3, the original list is restored and the cloned list has correct `next` pointers.

Before separation, each local structure looks like:

```text
x -> x' -> next_original
```

Pass 3 sets:

```text
x.next = next_original
```

which restores the original list's `next` pointer.

It also sets:

```text
x'.next = next_original.next
```

when `next_original` exists. By Lemma 1, `next_original.next` is the clone of the next original node. Therefore each clone points to the next clone. For the final original node, the clone's `next` becomes `None`, which is correct.

#### Theorem: The returned list is a deep copy of the input list.

By Lemma 1, there is one new cloned node for every original node with the same value. By Lemma 2, every cloned `random` pointer points to the clone of the corresponding original random target. By Lemma 3, the cloned `next` pointers reproduce the original list order, and the original list is restored.

Therefore the returned head points to an independent copied list with the same value, `next`, and `random` structure as the input.

### 10. Complexity

For the hash-map solution:

- Time: `O(n)` because each node is visited a constant number of times.
- Space: `O(n)` for the original-to-clone map.

For the interleaving solution:

- Time: `O(n)` because the algorithm performs three linear passes.
- Auxiliary Space: `O(1)` excluding the newly created output nodes.

The copied nodes themselves are required by the problem and are not counted as auxiliary space.

### 11. Common Pitfalls

- **Using node values as map keys:** values are not unique, so the map must use original node identity.
- **Pointing clone randoms to original nodes:** `clone.random = original.random` is a shallow copy bug.
- **Forgetting `None` random pointers:** `current.random.next` crashes when `current.random` is `None`.
- **Losing the original successor during interleaving:** save `next_original` before changing `current.next`.
- **Walking into clones as if they were originals:** after interleaving, advance from an original by two steps: `current = clone.next`.
- **Not restoring the original list:** the interleaving method temporarily mutates the input structure and must detach it cleanly.
- **Returning the original head:** the result must be the copied head, `head.next` after pass 1.
- **Assuming random only points forward:** random pointers may point backward, to self, or to any node.

### 12. First-Principles Summary

The list has two kinds of relationships:

```text
next   gives traversal order
random gives arbitrary identity-based references
```

Copying values is easy; copying references is hard because every copied reference must target the copied version of an original node.

So the fundamental requirement is a way to answer:

```text
given original node x, where is clone node x'?
```

The hash-map solution answers this with explicit storage:

```text
clone_of[x] = x'
```

The interleaving solution answers this by changing the temporary list shape:

```text
x.next = x'
```

Once that relationship exists, both `next` and `random` assignments follow directly. The whole problem is about preserving an original-to-clone invariant until every pointer in the copied structure has been rebuilt.

## Implementation
See `solutions/linked_list/p138_copy_list_with_random_pointer.py`.

## Tests
See `tests/linked_list/test_p138_copy_list_with_random_pointer.py`.

## Examples

### Example 1
- Input: `{'head': [[7, None], [13, 0], [11, 4], [10, 2], [1, 0]]}`
- Output: `[[7, None], [13, 0], [11, 4], [10, 2], [1, 0]]`

### Example 2
- Input: `{'head': [[1, 1], [2, 1]]}`
- Output: `[[1, 1], [2, 1]]`

### Example 3
- Input: `{'head': [[3, None], [3, 0], [3, None]]}`
- Output: `[[3, None], [3, 0], [3, None]]`

## Follow-up Practice

- Trace a self-random node where `node.random = node`.
- Trace duplicate values to confirm identity, not value, determines random targets.
- Implement the hash-map version first, then the interleaving version.
- After interleaving, draw both the mixed list and the detached copied list.
