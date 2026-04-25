# 637. Average of Levels in Binary Tree

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/average-of-levels-in-binary-tree/
- Official Group: Binary Tree BFS
- Pattern Group: Binary Tree BFS
- Patterns: binary-tree-bfs, tree-traversal

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given the `root` of a binary tree.

For every depth level in the tree, return the average value of all nodes on that level.

The root is level `0`.
Its children are level `1`.
Their children are level `2`, and so on.

For example, consider this tree:

```text
        3
       / \
      9   20
         /  \
        15   7
```

The levels are:

```text
level 0: [3]
level 1: [9, 20]
level 2: [15, 7]
```

So the averages are:

```text
level 0 average = 3 / 1 = 3.0
level 1 average = (9 + 20) / 2 = 14.5
level 2 average = (15 + 7) / 2 = 11.0
```

The answer is:

```text
[3.0, 14.5, 11.0]
```

So the problem is not asking for a traversal order as the final output.
It is asking for one number per depth:

```text
For each level, compute: sum of node values on that level / number of nodes on that level
```

That means each level must be treated as its own group.

### 2. Start From the Brute-Force/Baseline Idea

A direct way to think about the problem is:

1. Visit every node.
2. Record which level that node belongs to.
3. For each level, store a running sum and count.
4. After all nodes are visited, divide each level's sum by its count.

For example, using DFS, we could pass a `depth` value down the recursion:

```python
sums[depth] += node.val
counts[depth] += 1
```

At the end:

```python
answer[depth] = sums[depth] / counts[depth]
```

This baseline is already correct and takes `O(n)` time, because every node must be inspected at least once.

But it makes us store level information separately because DFS moves down one branch before finishing a level.
A DFS traversal might visit nodes in this order:

```text
3, 9, 20, 15, 7
```

That order does not naturally isolate:

```text
[9, 20]
```

as one complete level before moving deeper.

So the first-principles question is:

> Is there a traversal whose natural unit of work is exactly one level?

Yes: breadth-first search.

### 3. The Key Observation

A binary tree level is defined by distance from the root.

Breadth-first search visits nodes in increasing distance from the root:

```text
root first,
then all nodes one edge away,
then all nodes two edges away,
then all nodes three edges away,
...
```

That is exactly the grouping this problem needs.

So instead of asking:

```text
How do I attach a level number to each node?
```

we can ask:

```text
How do I process the tree one complete level at a time?
```

A queue gives us that ability.

The queue stores nodes that have been discovered but not yet processed. If we process the queue carefully, the queue can represent the current level.

### 4. The Queue/Level Invariant

The central invariant is:

```text
At the start of each outer loop iteration, the queue contains exactly the nodes of the next level to process, from left to right.
```

This invariant is the whole solution.

If the queue contains exactly one level, then:

1. The number of nodes in that level is `len(queue)`.
2. We can process exactly that many nodes.
3. Their values form exactly one level's sum.
4. Their children become exactly the next level.

The important detail is this:

```text
Freeze the current queue size before processing the level.
```

Why?

Because while processing level `d`, we enqueue children from level `d + 1`.
If we keep looping until the queue is empty, we will accidentally mix multiple levels together.

For the sample tree, start with:

```text
queue = [3]
```

Before processing level `0`, record:

```text
level_size = 1
```

Now process exactly one node: `3`.
During that processing, enqueue `9` and `20`:

```text
queue = [9, 20]
```

Those nodes are not part of level `0`; they are the next level.
Because we froze `level_size`, the current level ends after `3`.

Now the invariant is true again:

```text
At the start of the next iteration, queue = [9, 20]
```

which is exactly level `1`.

### 5. Detailed Algorithm

If the tree is empty, there are no levels, so return an empty list.

Otherwise:

1. Create an empty answer list.
2. Put the root node into a queue.
3. While the queue is not empty:
   - Record `level_size = len(queue)`.
   - Set `level_sum = 0`.
   - Repeat `level_size` times:
     - Remove one node from the front of the queue.
     - Add its value to `level_sum`.
     - If it has a left child, add that child to the back of the queue.
     - If it has a right child, add that child to the back of the queue.
   - Append `level_sum / level_size` to the answer.
4. Return the answer.

The queue is first-in, first-out:

```text
nodes discovered earlier are processed earlier
```

That preserves left-to-right order inside each level, although the average itself does not depend on left-to-right order. The order still matters for maintaining the clean BFS structure.

### 6. Pseudocode

```python
from collections import deque


def averageOfLevels(root):
    if root is None:
        return []

    answer = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level_sum = 0

        for _ in range(level_size):
            node = queue.popleft()
            level_sum += node.val

            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)

        answer.append(level_sum / level_size)

    return answer
```

The implementation should use a real queue, such as `collections.deque`, because removing from the front of a Python list with `pop(0)` shifts all remaining elements and can make the traversal slower than necessary.

### 7. Detailed Example Walkthrough

Use the first official example:

```text
root = [3, 9, 20, None, None, 15, 7]
```

This represents:

```text
        3
       / \
      9   20
         /  \
        15   7
```

Initialize:

```text
answer = []
queue = [3]
```

#### Level 0

At the start of the loop:

```text
queue = [3]
level_size = 1
level_sum = 0
```

Process exactly `1` node.

Remove `3`:

```text
level_sum = 3
```

Add its children:

```text
queue = [9, 20]
```

The level is finished because we processed exactly the original `level_size` nodes.

Average:

```text
3 / 1 = 3.0
```

Update answer:

```text
answer = [3.0]
```

#### Level 1

At the start of the next loop:

```text
queue = [9, 20]
level_size = 2
level_sum = 0
```

Process exactly `2` nodes.

Remove `9`:

```text
level_sum = 9
queue = [20]
```

`9` has no children in this example, so nothing is added.

Remove `20`:

```text
level_sum = 29
queue = []
```

Add `20`'s children:

```text
queue = [15, 7]
```

The level is finished after processing the two original nodes, `9` and `20`.

Average:

```text
29 / 2 = 14.5
```

Update answer:

```text
answer = [3.0, 14.5]
```

#### Level 2

At the start of the next loop:

```text
queue = [15, 7]
level_size = 2
level_sum = 0
```

Process exactly `2` nodes.

Remove `15`:

```text
level_sum = 15
queue = [7]
```

Remove `7`:

```text
level_sum = 22
queue = []
```

Neither node has children.

Average:

```text
22 / 2 = 11.0
```

Update answer:

```text
answer = [3.0, 14.5, 11.0]
```

The queue is now empty, so there are no more levels.

Return:

```text
[3.0, 14.5, 11.0]
```

### 8. Why the Algorithm Is Correct

We prove that the algorithm returns exactly the average value of every level.

#### Lemma 1: At the start of each outer loop iteration, the queue contains exactly one tree level.

Initially, the queue contains only the root.
The root is exactly level `0`, so the invariant is true before the first iteration.

Now assume the invariant is true at the start of some iteration. The queue contains exactly all nodes at level `d`.
The algorithm records `level_size`, then removes exactly those `level_size` nodes.
While removing them, it appends their children to the back of the queue.
Every child of a level `d` node is at level `d + 1`.
Also, every level `d + 1` node must be a child of some level `d` node in a binary tree connected from the root.
Therefore, after the `level_size` removals are complete, the queue contains exactly the nodes at level `d + 1`.

So the invariant is preserved.

#### Lemma 2: During each iteration, `level_sum` is the sum of exactly the nodes on that level.

By Lemma 1, the queue contains exactly the current level at the start of the iteration.
The algorithm freezes `level_size` and processes exactly that many nodes.
For each processed node, it adds that node's value exactly once to `level_sum`.
It does not add values from newly enqueued children during the same iteration, because those children are not processed until a later outer loop iteration.

Therefore, `level_sum` is exactly the sum of the current level's node values.

#### Lemma 3: The appended value for each iteration is the correct average for that level.

For a level with `level_size` nodes and total value `level_sum`, the average is:

```text
level_sum / level_size
```

By Lemma 2, the algorithm's `level_sum` is the correct sum.
By Lemma 1, `level_size` is the correct number of nodes on the level.
Therefore, the appended value is the correct average for that level.

#### Theorem: The algorithm returns the required list of level averages.

By Lemma 1, each outer loop iteration corresponds to exactly one level, in top-to-bottom order.
By Lemma 3, each appended value is the correct average for that level.
When the queue becomes empty, all reachable tree nodes have been processed, so all levels have been handled.
Thus the returned list contains exactly the average of every level in the binary tree.

### 9. Complexity

Let `n` be the number of nodes in the tree.
Let `w` be the maximum number of nodes on any one level, also called the maximum width of the tree.

#### Time Complexity: `O(n)`

Every node is enqueued once and dequeued once.
For each node, the algorithm performs constant work:

```text
read value,
check left child,
check right child,
possibly append children
```

So the total time is:

```text
O(n)
```

#### Space Complexity: `O(w)`

The queue stores nodes waiting to be processed.
At its largest, it can hold up to about one full level, plus children being collected for the next level during processing.
This is bounded by the maximum width of the tree up to constant factors.

So the auxiliary space is:

```text
O(w)
```

In the worst case, a very wide tree can have `w = O(n)`, so worst-case space is `O(n)`.
For a narrow skewed tree, `w` can be `1`, so the queue stays small.

### 10. Common Pitfalls

#### Pitfall: Not freezing the level size

Incorrect idea:

```python
while queue:
    node = queue.popleft()
    # add node.val to the current level
    # append children
```

This processes until the whole tree is empty, not until the current level is finished.
It loses the boundary between levels.

The fix is:

```python
level_size = len(queue)
for _ in range(level_size):
    ...
```

#### Pitfall: Dividing by the wrong count

The average for a level must divide by the number of nodes on that level, not by the total number of nodes seen so far.

For this tree:

```text
        3
       / \
      9   20
```

The second average is:

```text
(9 + 20) / 2
```

not:

```text
(3 + 9 + 20) / 3
```

#### Pitfall: Letting children affect the current average

Children are discovered while processing the current level, but they do not belong to the current level.

That is why newly appended children must wait until the next outer loop iteration before their values are added to a level sum.

#### Pitfall: Using list `pop(0)` as a queue

A Python list can simulate a queue, but `pop(0)` is inefficient because it shifts the rest of the list left.

Prefer:

```python
from collections import deque
```

and use:

```python
queue.popleft()
```

#### Pitfall: Forgetting the empty tree case

If `root` is `None`, there are no levels.
The correct result is:

```text
[]
```

Trying to put `None` in the queue and access `node.val` later would cause an error.

#### Pitfall: Assuming integer division

The result requires averages, so the output values are floats.
In Python 3, `/` performs true division:

```python
level_sum / level_size
```

Do not use floor division:

```python
level_sum // level_size
```

### 11. First-Principles Summary

The problem asks for one aggregate value per depth level.

A level is not an arbitrary group; it is the set of nodes at the same distance from the root.
Breadth-first search is designed to process nodes by increasing distance from the root.

The queue gives the algorithm a precise invariant:

```text
At the start of each loop, the queue contains exactly the next level.
```

Once that invariant is established, the rest follows mechanically:

```text
freeze queue length,
process exactly that many nodes,
sum their values,
enqueue their children,
append sum / count
```

The algorithm is efficient because every node contributes to exactly one level sum and is never revisited.

## Implementation
See `solutions/binary_tree_bfs/p637_average_of_levels_in_binary_tree.py`.

## Tests
See `tests/binary_tree_bfs/test_p637_average_of_levels_in_binary_tree.py`.

## Examples

### Example 1
- Input: `{'root': [3, 9, 20, None, None, 15, 7]}`
- Output: `[3.0, 14.5, 11.0]`

### Example 2
- Input: `{'root': [3, 9, 20, 15, 7]}`
- Output: `[3.0, 14.5, 11.0]`

## Follow-up Practice
- Trace the queue level by level.
- Implement with `deque` instead of a list pop from the front.
- Modify the level aggregation to produce sums, averages, or right-side values.
