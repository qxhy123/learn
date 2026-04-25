# 200. Number of Islands

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/number-of-islands/
- Official Group: Graph General
- Pattern Group: Graph General
- Patterns: graph-general

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given a rectangular grid of characters:

```text
'1' means land
'0' means water
```

An **island** is a group of land cells connected horizontally or vertically.

Diagonal contact does **not** connect land.

So this shape is one island:

```text
1 1 0
0 1 0
0 1 1
```

Every `1` can reach every other `1` in that island by moving only up, down, left, or right through land cells.

But this grid has two islands:

```text
1 0
0 1
```

The two land cells touch only diagonally, and diagonal movement is not allowed.

So the problem is not asking for the number of land cells. It is asking:

> How many separate connected groups of land exist in the grid?

This is a connected-components problem hidden inside a grid.

The grid is the graph:

- Each land cell is a graph vertex.
- Two land cells have an edge if they are adjacent horizontally or vertically.
- The answer is the number of connected components among land vertices.

### 2. Start From the Brute Force Baseline

A very direct idea is:

1. For every land cell, try to discover all land cells connected to it.
2. Store that discovered group as one island.
3. Compare it against previously discovered groups to avoid double-counting.

For example, from each `1`, we could run a search and collect all reachable land cells:

```python
islands = []

for each cell in grid:
    if cell is land:
        component = all land reachable from cell
        if component is not already one of islands:
            islands.append(component)
```

This is conceptually correct, but wasteful.

If one island contains 1,000 cells, then starting a fresh search from every cell in that island repeatedly rediscovers the same 1,000 cells. The repeated work is enormous because cells that belong to an already-known island are treated as if they might start a new island.

The key improvement is simple:

> Once a land cell has been assigned to an island, it never needs to start another island search.

That single idea turns repeated rediscovery into one traversal per island.

### 3. The Key Observation

When scanning the grid row by row, suppose we arrive at a land cell that has not been visited before.

There are only two possibilities:

1. It belongs to an island we have already counted.
2. It is the first cell we have encountered from a new island.

If we maintain `visited` correctly, possibility 1 cannot happen for an unvisited land cell.

Why?

Because when we first counted any island, we immediately searched through that entire island and marked every reachable land cell as visited. Therefore, if the current land cell belonged to a previously counted island, it would already be visited.

So the important conclusion is:

> Every unvisited land cell found during the outer scan is the start of exactly one new island.

That gives the algorithm:

1. Scan every cell.
2. When you see an unvisited `1`, increment the island count.
3. Search from that cell and mark the whole island as visited.

### 4. The Graph/Search Invariant

The invariant is the heart of the solution:

```text
After finishing a search from a newly found land cell,
every land cell in that island is marked visited,
and no water cell or separate island is marked because of that search.
```

This invariant works because the search is allowed to move only to valid neighboring land cells:

```text
(row - 1, col)  up
(row + 1, col)  down
(row, col - 1)  left
(row, col + 1)  right
```

The search does not cross water, does not leave the grid, and does not move diagonally.

So a DFS or BFS starting from one land cell explores exactly the connected component containing that land cell.

That is exactly one island.

### 5. Detailed Algorithm

Use either DFS or BFS. The counting logic is the same.

Maintain:

```text
rows, cols     grid dimensions
visited        cells already assigned to an island
islands        number of islands found so far
```

Then:

1. Initialize `islands = 0`.
2. Scan each position `(row, col)` in the grid.
3. If `grid[row][col] == '0'`, ignore it because water cannot start an island.
4. If `(row, col)` is already visited, ignore it because it already belongs to a counted island.
5. Otherwise, this is an unvisited land cell.
6. Increment `islands`.
7. Run DFS or BFS from `(row, col)`:
   - Mark the starting cell visited.
   - Repeatedly inspect its four neighbors.
   - For each neighbor, continue only if it is inside the grid, is land, and is unvisited.
   - Mark accepted neighbors visited and keep expanding from them.
8. After the search finishes, the entire island has been consumed.
9. Continue scanning the rest of the grid.
10. Return `islands`.

A practical implementation may use a separate `visited` set, or it may mutate the grid by changing visited land cells from `'1'` to `'0'`. Both approaches use the same invariant. A separate `visited` set preserves the input; grid mutation saves extra visited storage but changes the input grid.

### 6. DFS Pseudocode

Recursive DFS version:

```python
def numIslands(grid):
    if not grid:
        return 0

    rows = len(grid)
    cols = len(grid[0])
    visited = set()
    islands = 0

    def dfs(row, col):
        if row < 0 or row == rows:
            return
        if col < 0 or col == cols:
            return
        if grid[row][col] == '0':
            return
        if (row, col) in visited:
            return

        visited.add((row, col))

        dfs(row - 1, col)
        dfs(row + 1, col)
        dfs(row, col - 1)
        dfs(row, col + 1)

    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == '1' and (row, col) not in visited:
                islands += 1
                dfs(row, col)

    return islands
```

Iterative BFS version:

```python
from collections import deque


def numIslands(grid):
    if not grid:
        return 0

    rows = len(grid)
    cols = len(grid[0])
    visited = set()
    islands = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for row in range(rows):
        for col in range(cols):
            if grid[row][col] != '1' or (row, col) in visited:
                continue

            islands += 1
            queue = deque([(row, col)])
            visited.add((row, col))

            while queue:
                current_row, current_col = queue.popleft()

                for row_delta, col_delta in directions:
                    next_row = current_row + row_delta
                    next_col = current_col + col_delta

                    if next_row < 0 or next_row == rows:
                        continue
                    if next_col < 0 or next_col == cols:
                        continue
                    if grid[next_row][next_col] != '1':
                        continue
                    if (next_row, next_col) in visited:
                        continue

                    visited.add((next_row, next_col))
                    queue.append((next_row, next_col))

    return islands
```

The BFS marks a neighbor as visited when it is enqueued, not when it is dequeued. That prevents the same cell from being placed in the queue multiple times by different neighboring land cells.

### 7. Detailed Example Walkthrough

Consider Example 2:

```text
grid = [
  ["1", "1", "0", "0", "0"],
  ["1", "1", "0", "0", "0"],
  ["0", "0", "1", "0", "0"],
  ["0", "0", "0", "1", "1"]
]
```

Write coordinates as `(row, col)`.

Initial state:

```text
islands = 0
visited = {}
```

#### Scan starts at `(0, 0)`

`grid[0][0]` is land and unvisited.

So `(0, 0)` starts a new island:

```text
islands = 1
```

Search from `(0, 0)`.

Reachable land cells through four-direction movement are:

```text
(0, 0), (0, 1), (1, 0), (1, 1)
```

The search marks all four visited.

Visited now contains:

```text
{(0, 0), (0, 1), (1, 0), (1, 1)}
```

The outer scan continues.

#### Scan reaches `(0, 1)`

`grid[0][1]` is land, but it is already visited.

Do not increment `islands`.

This is exactly why the visited structure matters: `(0, 1)` is part of the first island, not a second island.

#### Scan reaches `(1, 0)` and `(1, 1)`

Both are land, but both are already visited.

Still:

```text
islands = 1
```

#### Scan reaches `(2, 2)`

`grid[2][2]` is land and unvisited.

It cannot be connected to the first island because all paths from it to the top-left land block would require crossing water or moving diagonally.

So it starts a new island:

```text
islands = 2
```

Search from `(2, 2)`.

Its four neighbors are water or out of the component:

```text
(1, 2) is 0
(3, 2) is 0
(2, 1) is 0
(2, 3) is 0
```

So this island contains only:

```text
(2, 2)
```

Mark it visited.

#### Scan reaches `(3, 3)`

`grid[3][3]` is land and unvisited.

It starts another new island:

```text
islands = 3
```

Search from `(3, 3)`.

It can move right to `(3, 4)`, which is also land.

So this island contains:

```text
(3, 3), (3, 4)
```

Both are marked visited.

The scan finishes. The final answer is:

```text
3
```

### 8. Why the Algorithm Is Correct

We prove that the algorithm returns exactly the number of islands.

#### Lemma 1: Each search marks exactly one island.

A search starts only from a land cell. During the search, it moves only to neighboring cells that are inside the grid, are land, and are connected by an allowed horizontal or vertical step. Therefore every marked cell is reachable from the starting land cell through valid land moves.

The search also tries all four allowed directions from every reached land cell. Therefore if another land cell is reachable from the starting cell through valid moves, the search will eventually follow that path and mark it.

So the search marks all and only the land cells in the starting cell's island.

#### Lemma 2: The algorithm increments the count at most once per island.

After the first time the scan finds any cell of an island, the algorithm runs a search from that cell. By Lemma 1, that search marks every cell in that island visited. Later, when the outer scan reaches any other cell from the same island, it is already visited, so the algorithm does not increment the count again.

Therefore no island is counted more than once.

#### Lemma 3: The algorithm increments the count at least once per island.

Every island contains at least one land cell. The outer loops scan every grid cell, so they eventually reach some land cell from that island. If the island has not been counted yet, that cell is unvisited, and the algorithm increments the count. If it is already visited, then it was visited by a previous search from the same island, which means the island was already counted.

Therefore every island is counted at least once.

#### Theorem: The returned count equals the number of islands.

By Lemma 2, the algorithm never overcounts an island. By Lemma 3, the algorithm never misses an island. Therefore the final value of `islands` is exactly the number of islands in the grid.

### 9. Complexity

Let:

```text
m = number of rows
n = number of columns
```

The outer scan visits every cell once.

Each land cell is added to `visited` once. Its four neighbors are inspected during DFS or BFS. Water cells may be checked as neighbors multiple times from adjacent land cells, but each check is constant-time and bounded by the four-direction edges of the grid.

So the time complexity is:

```text
O(m * n)
```

The space complexity depends on the implementation:

- With a `visited` set, space is `O(m * n)` in the worst case.
- Recursive DFS also uses call-stack space up to `O(m * n)` in the worst case.
- Iterative BFS uses a queue that can hold up to `O(m * n)` cells in the worst case.
- If the grid is mutated in place and recursion/queue space is ignored separately, visited storage can be avoided, but traversal stack or queue space may still be needed.

### 10. Common Pitfalls

- **Counting land cells instead of islands:** The answer increases only when an unvisited land cell starts a new component, not for every `1`.
- **Allowing diagonal movement:** Only up, down, left, and right connect cells. Diagonal `1`s are separate unless connected through another valid path.
- **Marking visited too late:** In BFS, mark a cell visited when enqueuing it. Otherwise multiple neighbors can enqueue the same cell repeatedly.
- **Forgetting bounds checks:** Neighbor coordinates can be outside the grid at edges and corners.
- **Comparing against integer `1` instead of string `'1'`:** The LeetCode input uses characters, so checks should match `'1'` and `'0'`.
- **Mutating the grid unexpectedly:** Turning visited land into water is valid, but callers will see the changed grid. Use `visited` if preserving input matters.
- **Recursive DFS depth:** A huge snake-shaped island can exceed Python's recursion limit. Iterative DFS or BFS avoids that risk.
- **Not handling an empty grid defensively:** Some environments guarantee a non-empty grid, but a robust helper can return `0` for an empty input.

### 11. First-Principles Summary

The problem becomes simple once the grid is viewed as a graph.

```text
land cell = vertex
horizontal/vertical land adjacency = edge
island = connected component
answer = number of connected components
```

The outer scan finds possible component starts. The search consumes one whole component at a time. The visited set is what connects these two pieces: it guarantees that once an island has been counted, none of its cells can trigger another count.

So the core rule is:

> Count an island when you first discover unvisited land, then immediately mark the entire reachable land mass so it cannot be counted again.

That is the complete first-principles reason DFS or BFS solves Number of Islands.

## Implementation
See `solutions/graph_general/p200_number_of_islands.py`.

## Tests
See `tests/graph_general/test_p200_number_of_islands.py`.

## Examples

### Example 1
- Input: `{'grid': [['1', '1', '1', '1', '0'], ['1', '1', '0', '1', '0'], ['1', '1', '0', '0', '0'], ['0', '0', '0', '0', '0']]}`
- Output: `1`

### Example 2
- Input: `{'grid': [['1', '1', '0', '0', '0'], ['1', '1', '0', '0', '0'], ['0', '0', '1', '0', '0'], ['0', '0', '0', '1', '1']]}`
- Output: `3`

## Follow-up Practice
- Trace the first island in Example 2 and list cells in the order your DFS or BFS visits them.
- Rewrite the solution once using a `visited` set and once by mutating land cells to water.
- Explain why diagonal land cells do not merge into one island.
- Compare recursive DFS, iterative DFS, and BFS for a grid containing one very large island.
