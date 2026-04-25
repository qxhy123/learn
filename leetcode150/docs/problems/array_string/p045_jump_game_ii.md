# 45. Jump Game II

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/jump-game-ii/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: greedy, interval-expansion

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an array `nums`.

At index `i`, the value `nums[i]` tells you the maximum jump length from that index:

```text
from i, you may jump to any index in:

i + 1, i + 2, ..., i + nums[i]
```

You start at index `0`.

Return the minimum number of jumps needed to reach the last index.

The problem guarantees that the last index can be reached.

For example:

```text
nums = [2, 3, 1, 1, 4]
```

From index `0`, you can jump at most `2` steps, so your first jump can land at index `1` or index `2`.

If you jump to index `1`, where `nums[1] = 3`, you can then jump directly to index `4`, the last index:

```text
0 -> 1 -> 4
```

That uses `2` jumps, and no solution can use fewer than `2` because index `0` cannot reach index `4` directly.

So the answer is:

```text
2
```

The real problem is:

> Among all possible paths from index `0` to the last index, find the path with the fewest edges.

Each jump is one edge. We want the shortest path in a special graph whose nodes are array indices.

---

### 2. Start From the Brute Force View

The most literal model is a graph:

```text
node i = array index i
edge i -> j exists if 1 <= j - i <= nums[i]
```

Then the question becomes:

```text
What is the shortest number of edges from node 0 to node n - 1?
```

A breadth-first search would be correct:

1. Start with index `0`.
2. All indices reachable in one jump form the next BFS layer.
3. All indices reachable from those form the next layer.
4. The first layer that contains the last index gives the minimum jump count.

For `nums = [2, 3, 1, 1, 4]`, the BFS layers are:

```text
0 jumps: {0}
1 jump : {1, 2}
2 jumps: {3, 4}
```

The last index appears in the `2`-jump layer, so the answer is `2`.

This is conceptually clean, but a direct BFS can examine many edges. From index `i`, there can be up to `nums[i]` outgoing edges, so in the worst case this can become `O(n^2)`.

There is also a dynamic programming baseline:

```python
dp[0] = 0
dp[i] = minimum jumps needed to reach i

for i in range(n):
    for next_i in range(i + 1, min(n, i + nums[i] + 1)):
        dp[next_i] = min(dp[next_i], dp[i] + 1)
```

This is also correct because it relaxes every possible jump, but it still may inspect `O(n^2)` transitions.

The key is to keep the BFS idea while avoiding explicit enumeration of every edge.

---

### 3. Key Observation: BFS Layers Are Contiguous Intervals

In a normal graph, a BFS layer can be an arbitrary set of nodes.

Here, the graph comes from an array, and jumps only move to the right across index ranges.

If some set of positions is reachable in `k` jumps, then the useful information is not every individual position separately. What matters is the interval of positions we are currently scanning, and the farthest position that any of them can reach with one more jump.

Suppose all positions in the current BFS layer lie between:

```text
layer_start ... current_end
```

While scanning this layer, every index `i` can extend the next layer as far as:

```text
i + nums[i]
```

So the farthest index reachable with one more jump is:

```text
farthest = max(i + nums[i]) over all i in the current layer
```

Once we finish scanning every index up to `current_end`, we have finished considering all ways to make the current number of jumps. At that exact moment, we must take one more jump, and the next layer extends to `farthest`.

This is the greedy insight:

> Do not decide the exact landing index immediately. Scan every index reachable with the current number of jumps, remember the farthest next reach, and only increase the jump count when the current reachable frontier is exhausted.

This is still BFS, but compressed into a single left-to-right pass.

---

### 4. Why Greedy Is Safe Here

The word "greedy" can be misleading if it sounds like:

```text
Always jump to the locally largest nums[i].
```

That is not the algorithm.

The safe greedy decision is different:

```text
Among all indices reachable with the current number of jumps,
compute the farthest boundary reachable with one more jump.
```

We are not committing to one landing index too early. We are postponing that choice until the entire current layer has been inspected.

For example:

```text
nums = [2, 3, 1, 1, 4]
```

After one jump from index `0`, you may be at index `1` or index `2`.

If you only looked at one candidate, you might miss the better continuation. But if you scan the whole current layer:

```text
index 1 reaches 1 + 3 = 4
index 2 reaches 2 + 1 = 3
```

then you know the best next boundary is `4`.

This is exactly what BFS would discover, just without storing a queue.

---

### 5. State and Invariant

Maintain three pieces of state:

```text
jumps       = number of jumps already committed
current_end = farthest index reachable using exactly jumps jumps after finishing prior layers
farthest    = farthest index reachable using jumps + 1 jumps from indices scanned so far
```

The scan index `i` moves from left to right.

The central invariant is:

```text
Before crossing current_end, every scanned index i is reachable using at most jumps jumps.
After processing such an index, farthest includes every position reachable by one more jump from scanned indices in the current layer.
```

When `i < current_end`, we are still inside the same jump layer. We update:

```text
farthest = max(farthest, i + nums[i])
```

but we do not increase `jumps` yet, because there may still be another reachable index in this same layer that extends farther.

When `i == current_end`, the current layer has been fully scanned. There are no more positions reachable with only `jumps` jumps that can improve the next boundary. Therefore we commit one more jump:

```text
jumps += 1
current_end = farthest
```

Now `current_end` is the boundary of the next BFS layer.

---

### 6. Detailed Algorithm

Handle the smallest case first:

```text
If nums has length 1, we are already at the last index, so the answer is 0.
```

Then scan from left to right, but stop before the last index.

Why stop before the last index?

Because the goal is to reach the last index, not jump away from it. Counting a jump when `i` is already the last index can produce an extra jump.

Algorithm:

1. Set:

```text
jumps = 0
current_end = 0
farthest = 0
```

2. For each index `i` from `0` to `n - 2`:

```text
farthest = max(farthest, i + nums[i])
```

3. If `i == current_end`, the current jump layer is exhausted:

```text
jumps += 1
current_end = farthest
```

4. If `current_end >= n - 1`, the last index is reachable with `jumps` jumps, so we can stop.

5. Return `jumps`.

The loop is linear because every index is processed at most once.

---

### 7. Pseudocode

```python
def jump(nums):
    n = len(nums)
    if n <= 1:
        return 0

    jumps = 0
    current_end = 0
    farthest = 0

    for i in range(n - 1):
        farthest = max(farthest, i + nums[i])

        if i == current_end:
            jumps += 1
            current_end = farthest

            if current_end >= n - 1:
                break

    return jumps
```

The LeetCode version can be written as:

```python
from typing import List


class Solution:
    def jump(self, nums: List[int]) -> int:
        jumps = 0
        current_end = 0
        farthest = 0

        for i in range(len(nums) - 1):
            farthest = max(farthest, i + nums[i])

            if i == current_end:
                jumps += 1
                current_end = farthest

                if current_end >= len(nums) - 1:
                    break

        return jumps
```

No queue is needed because the queue's layers have collapsed into interval boundaries.

---

### 8. Detailed Example Walkthrough

Use:

```text
nums = [2, 3, 1, 1, 4]
```

Start:

```text
jumps = 0
current_end = 0
farthest = 0
```

At index `0`:

```text
nums[0] = 2
farthest = max(0, 0 + 2) = 2
```

Because:

```text
i == current_end == 0
```

we have finished scanning all positions reachable in `0` jumps. We must spend one jump:

```text
jumps = 1
current_end = 2
```

Meaning:

```text
With 1 jump, every useful candidate position is within indices 1 through 2.
```

At index `1`:

```text
nums[1] = 3
farthest = max(2, 1 + 3) = 4
```

Index `1` can reach the last index, but we do not increment `jumps` immediately because we have not yet finished scanning the current layer. We only record that the next layer can reach at least index `4`.

At index `2`:

```text
nums[2] = 1
farthest = max(4, 2 + 1) = 4
```

Now:

```text
i == current_end == 2
```

So all positions reachable in `1` jump have been scanned. We commit the second jump:

```text
jumps = 2
current_end = 4
```

Since:

```text
current_end >= last index
```

the answer is:

```text
2
```

The compressed layer view is:

```text
0 jumps: reachable through index 0
1 jump : reachable through index 2
2 jumps: reachable through index 4
```

So the minimum number of jumps is `2`.

---

### 9. Another Example: `[2, 3, 0, 1, 4]`

Start:

```text
jumps = 0
current_end = 0
farthest = 0
```

At index `0`:

```text
farthest = max(0, 0 + 2) = 2
i == current_end
jumps = 1
current_end = 2
```

The first jump can reach indices `1` and `2`.

At index `1`:

```text
farthest = max(2, 1 + 3) = 4
```

At index `2`:

```text
farthest = max(4, 2 + 0) = 4
i == current_end
jumps = 2
current_end = 4
```

The last index is reachable, so the answer is:

```text
2
```

One optimal path is:

```text
0 -> 1 -> 4
```

---

### 10. Correctness

We prove that the algorithm returns the minimum number of jumps.

#### Lemma 1: At the start of each layer, `current_end` is the farthest index reachable using `jumps` jumps.

Initially:

```text
jumps = 0
current_end = 0
```

This is true because before making any jump, only index `0` is reachable.

Assume it is true for some value of `jumps`.

The algorithm scans every reachable index in the current layer before increasing `jumps`. While scanning those indices, it computes:

```text
farthest = max(i + nums[i])
```

over all scanned indices `i` reachable with `jumps` jumps.

Any index reachable with `jumps + 1` jumps must be reached by first standing on some index reachable with `jumps` jumps, then making one jump from that index. Therefore no `jumps + 1` destination can be beyond `farthest`.

Also, by definition of the maximum, `farthest` itself is reachable with one more jump from some scanned index.

So after the layer is exhausted and the algorithm sets:

```text
current_end = farthest
```

the new `current_end` is exactly the farthest index reachable using `jumps + 1` jumps.

By induction, the lemma holds for every layer.

#### Lemma 2: The algorithm increments `jumps` only when another jump is necessary.

The algorithm increments `jumps` only when:

```text
i == current_end
```

At that moment, every index reachable with the current number of jumps has already been scanned. If the last index has not yet been reached by the current layer, then no path using the current number of jumps can finish. A further jump is necessary.

Thus every increment corresponds to moving from one BFS layer to the next.

#### Lemma 3: The first time `current_end` reaches the last index, `jumps` is minimal.

By Lemma 1, after each layer transition, `current_end` is the farthest index reachable with exactly the current number of committed jumps.

If `current_end >= n - 1`, then the last index is reachable with `jumps` jumps.

Before this transition, the previous `current_end` was less than `n - 1`, meaning the last index was not reachable with fewer jumps.

Therefore the first such `jumps` value is the minimum possible.

#### Conclusion

The algorithm simulates BFS layers exactly, but represents each layer by its right boundary. Since BFS finds the shortest number of edges in an unweighted graph, and each jump is one edge, the returned `jumps` is the minimum number of jumps needed to reach the last index.

---

### 11. Complexity

Let `n = len(nums)`.

Time complexity:

```text
O(n)
```

Each index before the last is scanned at most once.

Space complexity:

```text
O(1)
```

The algorithm stores only `jumps`, `current_end`, `farthest`, and the loop index.

---

### 12. Common Pitfalls

- Iterating through the last index and counting an extra jump. The loop should usually stop at `n - 2`.
- Incrementing `jumps` every time `farthest` improves. A jump should be counted only when the current layer ends.
- Confusing `current_end` and `farthest`. `current_end` is the boundary reachable with the current number of jumps; `farthest` is the boundary being built for one more jump.
- Greedily jumping to the index with the largest `nums[i]` instead of the index that produces the farthest `i + nums[i]`.
- Forgetting that `nums[i]` is a maximum jump length, not an exact jump length.
- Returning `1` for a one-element array. If you start on the last index, the minimum number of jumps is `0`.
- Adding complicated path reconstruction. The problem asks only for the jump count, not the actual sequence of indices.
- Designing for unreachable inputs as the main case. LeetCode 45 guarantees reachability; defensive code is possible, but the core proof relies on that guarantee.

---

### 13. First-Principles Summary

The brute-force way to think about Jump Game II is shortest path search over array indices. From each index, every allowed jump is an outgoing edge. A normal BFS would find the minimum number of jumps, but it may enumerate too many edges.

The array structure makes the BFS layers compressible. All we need from the current layer is its right boundary, `current_end`, and all we need to build the next layer is the farthest destination seen so far, `farthest`.

The algorithm is therefore:

```text
scan all indices reachable with the current jump count
record the farthest index reachable with one more jump
when the current reachable boundary is exhausted, spend one jump
move the boundary to the farthest recorded reach
```

This works because the algorithm never ignores a candidate index reachable with the current number of jumps. It waits until the whole layer has been scanned, then advances to the best possible next boundary. That is exactly BFS behavior, expressed as a greedy interval expansion.

## Implementation

See `solutions/array_string/p045_jump_game_ii.py`.

## Tests

See `tests/array_string/test_p045_jump_game_ii.py`.

## Examples

### Example 1
- Input: `{'nums': [2, 3, 1, 1, 4]}`
- Output: `2`

### Example 2
- Input: `{'nums': [2, 3, 0, 1, 4]}`
- Output: `2`

## Follow-up Practice
- Trace `current_end` and `farthest` after each index.
- Compare the compressed greedy scan with an explicit BFS layer simulation.
- Test the boundary case `nums = [0]`, where the answer is `0`.
