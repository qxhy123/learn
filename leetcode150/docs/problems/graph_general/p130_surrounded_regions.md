# 130. Surrounded Regions

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/surrounded-regions/
- Official Group: Graph General
- Pattern Group: Graph General
- Patterns: graph-general, grid-graph, dfs, bfs, flood-fill

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an `m x n` board of characters:

```text
'X' = blocked / wall cell
'O' = open cell
```

You must modify the board **in place** so that every `O` region that is completely surrounded by `X` becomes `X`.

A region is formed by connecting `O` cells in the four cardinal directions:

```text
up, down, left, right
```

Diagonal contact does not connect two cells.

The important word is **surrounded**.

An `O` cell should be captured only if its entire connected component cannot reach the outside edge of the board.

For example:

```text
X X X X
X O O X
X X O X
X O X X
```

The `O` cells at positions:

```text
(1, 1), (1, 2), (2, 2)
```

form one connected component. That component is enclosed by `X` on all sides, so it becomes `X`.

The `O` cell at:

```text
(3, 1)
```

is already on the border. It is not surrounded, because it touches the outside of the board directly. It stays `O`.

The final board is:

```text
X X X X
X X X X
X X X X
X O X X
```

So the real problem is:

> Change every `O` component that is not connected to the border into `X`, while preserving every `O` component that can reach the border.

---

### 2. Model the Board as a Graph

A grid is a graph if we treat each cell as a vertex.

For this problem:

```text
vertex = one board cell containing 'O'
edge   = four-directional adjacency between two 'O' cells
```

For a cell `(r, c)`, its possible neighbors are:

```text
(r - 1, c)
(r + 1, c)
(r, c - 1)
(r, c + 1)
```

Only neighbors inside the board matter, and only neighbors containing `O` are part of the same region.

Now the word **surrounded** becomes a graph reachability property:

```text
An O is safe if it can reach any border O through O cells.
An O is captured if it cannot reach any border O through O cells.
```

This is not primarily about counting regions or checking every wall around every cell. It is about reachability from the boundary.

---

### 3. Start From the Brute Force Baseline

The direct way to solve the problem is to examine each `O` component separately.

For every unvisited `O`:

1. Run DFS or BFS to collect its whole connected component.
2. Track whether any cell in that component lies on the border.
3. If the component never touches the border, flip all cells in the component to `X`.
4. If the component touches the border, leave it as `O`.

Conceptually:

```python
visited = set()

for each cell (r, c):
    if board[r][c] == 'O' and (r, c) not in visited:
        component = []
        touches_border = False

        search from (r, c):
            mark each reached O as visited
            add it to component
            if it is on the border:
                touches_border = True

        if not touches_border:
            for each cell in component:
                board[cell] = 'X'
```

This is correct, because it makes a decision for each connected component.

Its time complexity is still `O(m * n)` if implemented carefully, because every cell is visited at most once. But it has a practical inconvenience: for every component, we may need to store all cells until we know whether the component is captured.

The deeper question is:

> Can we avoid deciding component-by-component and instead identify all cells that must never be captured?

Yes. Work backward from the border.

---

### 4. The Key Observation: Only Border-Reachable `O`s Survive

A surrounded region cannot touch the border.

Why?

If an `O` is on the border, then it is open to the outside of the board, so it cannot be surrounded.

If an interior `O` is connected to a border `O`, then it also has a path to the outside:

```text
interior O -> ... -> border O -> outside
```

So it cannot be surrounded either.

Therefore:

```text
Every O connected to a border O is safe.
Every remaining O is captured.
```

This flips the problem from:

```text
Find all captured regions.
```

to the easier problem:

```text
Find all safe regions, then capture everything else.
```

That is the central first-principles move.

The border is the only place where an `O` component can escape. So instead of starting searches from every interior `O`, start searches from border `O`s and mark all `O`s reachable from them.

---

### 5. The Search Invariant

We maintain a temporary mark, usually something like `S` for safe:

```text
'S' = this cell was originally 'O' and is connected to the border
```

The invariant during DFS or BFS is:

```text
Every cell marked S is an original O that can reach the border through original O cells.
```

Initially, this is true for border `O` cells, because they are on the border themselves.

When the search moves from a marked safe cell to a neighboring `O`, that neighbor is also safe:

```text
neighbor O -> current safe cell -> ... -> border
```

So marking the neighbor preserves the invariant.

At the end of all border searches, the invariant gives us exactly what we need:

```text
S cells are the O cells that must remain O.
Unmarked O cells are not connected to the border, so they must become X.
```

This is why a graph traversal solves the problem cleanly.

---

### 6. Detailed Algorithm

Use the board itself as the visited structure by temporarily changing safe `O`s to a sentinel character.

Let:

```text
rows = number of rows
cols = number of columns
```

Algorithm:

1. If the board is empty, return immediately.
2. Traverse the first and last row.
   - Whenever a border cell contains `O`, flood-fill from it and mark reachable `O`s as safe.
3. Traverse the first and last column.
   - Whenever a border cell contains `O`, flood-fill from it and mark reachable `O`s as safe.
4. Scan the whole board.
   - Change every remaining `O` to `X` because it is surrounded.
   - Change every temporary safe marker back to `O`.

The flood-fill can be DFS or BFS.

For iterative DFS:

```python
def mark_safe(start_r, start_c):
    if board[start_r][start_c] != 'O':
        return

    stack = [(start_r, start_c)]
    board[start_r][start_c] = 'S'

    while stack:
        r, c = stack.pop()

        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == 'O':
                board[nr][nc] = 'S'
                stack.append((nr, nc))
```

Then:

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        if not board or not board[0]:
            return

        rows, cols = len(board), len(board[0])

        def mark_safe(start_r: int, start_c: int) -> None:
            if board[start_r][start_c] != 'O':
                return

            stack = [(start_r, start_c)]
            board[start_r][start_c] = 'S'

            while stack:
                r, c = stack.pop()

                for nr, nc in (
                    (r - 1, c),
                    (r + 1, c),
                    (r, c - 1),
                    (r, c + 1),
                ):
                    if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == 'O':
                        board[nr][nc] = 'S'
                        stack.append((nr, nc))

        for c in range(cols):
            mark_safe(0, c)
            mark_safe(rows - 1, c)

        for r in range(rows):
            mark_safe(r, 0)
            mark_safe(r, cols - 1)

        for r in range(rows):
            for c in range(cols):
                if board[r][c] == 'O':
                    board[r][c] = 'X'
                elif board[r][c] == 'S':
                    board[r][c] = 'O'
```

The method returns `None` because the problem asks us to mutate `board` in place.

---

### 7. Walk Through the Main Example

Start with:

```text
row 0: X X X X
row 1: X O O X
row 2: X X O X
row 3: X O X X
```

Coordinates of `O` cells:

```text
(1, 1), (1, 2), (2, 2), (3, 1)
```

First, look only at border cells.

The border consists of:

```text
row 0
row 3
column 0
column 3
```

Among those border cells, the only `O` is:

```text
(3, 1)
```

Flood-fill from `(3, 1)`.

Its neighbors are:

```text
(2, 1) = X
(4, 1) = outside
(3, 0) = X
(3, 2) = X
```

So only `(3, 1)` is marked safe:

```text
X X X X
X O O X
X X O X
X S X X
```

Now all border-reachable `O`s have been marked.

Scan every cell:

- `(1, 1)` is still `O`, so it is not border-reachable. Flip it to `X`.
- `(1, 2)` is still `O`, so it is not border-reachable. Flip it to `X`.
- `(2, 2)` is still `O`, so it is not border-reachable. Flip it to `X`.
- `(3, 1)` is `S`, so restore it to `O`.

Final board:

```text
X X X X
X X X X
X X X X
X O X X
```

Notice that we never had to explicitly prove that the middle component was surrounded by checking all walls around it. Once all escapable `O`s were marked, every remaining `O` was trapped by definition.

---

### 8. A Border-Connected Interior Example

Consider:

```text
X O X X
X O O X
X X O X
X X X X
```

The `O` at `(0, 1)` is on the border.

Flood-fill from it reaches:

```text
(0, 1) -> (1, 1) -> (1, 2) -> (2, 2)
```

So even though `(1, 1)`, `(1, 2)`, and `(2, 2)` are interior cells, they are not captured. They have a path to the outside through `(0, 1)`.

After marking safe cells:

```text
X S X X
X S S X
X X S X
X X X X
```

The final restore step turns every `S` back into `O`, and nothing is flipped.

This example is the reason local reasoning like “an interior `O` has many nearby `X`s” is not enough. The property depends on the whole connected component.

---

### 9. Correctness

We prove the algorithm produces exactly the required final board.

#### Lemma 1: Every cell marked safe is not surrounded.

A cell is marked safe only when it is reached by a flood-fill that starts from a border `O` and moves only through `O` cells.

Therefore every marked cell has a path of `O` cells to a border `O`.

A region with such a path is connected to the outside of the board, so it is not surrounded.

So every marked safe cell should remain `O`.

#### Lemma 2: Every unmarked `O` is surrounded.

After the algorithm runs flood-fill from every border `O`, suppose some remaining unmarked `O` were not surrounded.

If it were not surrounded, its connected component would have to touch the border. That means there would be a path of `O` cells from this cell to some border `O`.

But the algorithm starts a flood-fill from every border `O`, and flood-fill reaches every `O` in the same connected component.

So this cell would have been marked safe, which contradicts that it is unmarked.

Therefore every remaining unmarked `O` is surrounded and should be flipped to `X`.

#### Lemma 3: The final scan assigns the correct value to every cell.

During the final scan:

- Original `X` cells remain `X`.
- Safe-marked cells are restored to `O` by Lemma 1.
- Remaining unmarked `O` cells are changed to `X` by Lemma 2.

Thus every cell has exactly the required final value.

#### Theorem: The algorithm correctly solves Surrounded Regions.

By Lemma 3, after the final scan, all and only surrounded `O` regions have been captured, and all border-connected `O` regions have been preserved. Therefore the board is correct.

---

### 10. Complexity

Let:

```text
m = number of rows
n = number of columns
```

Each cell can be marked safe at most once. The final scan also visits each cell once.

So the time complexity is:

```text
O(m * n)
```

The extra space depends on the traversal implementation.

For iterative DFS or BFS, the stack or queue can hold up to `O(m * n)` cells in the worst case, for example when the whole board is `O`.

So auxiliary space is:

```text
O(m * n)
```

If recursive DFS is used, the recursion call stack can also be `O(m * n)` in the worst case, and may exceed Python's recursion limit on large boards. Iterative DFS or BFS avoids that risk.

The temporary safe marker is stored inside the board, so it does not require a separate visited set.

---

### 11. Common Pitfalls

#### Pitfall: Starting from interior cells first

You can solve the problem by exploring every component, but the cleanest approach starts from border `O`s. The border is the source of all cells that must be preserved.

#### Pitfall: Treating diagonal cells as connected

Only four directions count. These two `O`s are not connected:

```text
O X
X O
```

A diagonal path cannot save a region.

#### Pitfall: Flipping `O` to `X` during the border search

During the first phase, do not capture cells. You are only identifying safe cells. If you flip too early, you can destroy paths that later searches need.

#### Pitfall: Forgetting to restore the temporary marker

The output board must contain only `X` and `O`. Any temporary marker such as `S`, `#`, or `E` must be changed back to `O` at the end.

#### Pitfall: Marking visited too late

When using a stack or queue, mark a neighbor as soon as you add it. If you wait until popping it, the same cell can be added many times by different neighbors.

#### Pitfall: Recursive DFS depth in Python

A large board full of `O`s can create a very deep recursion chain. Prefer iterative DFS or BFS unless the constraints are small enough and recursion depth is managed deliberately.

#### Pitfall: Returning a new board

LeetCode expects in-place mutation. The function should not return a transformed board as the answer.

---

### 12. First-Principles Summary

The problem looks like “find surrounded regions,” but the easier invariant is about the opposite set:

```text
Find O cells that cannot be surrounded.
```

An `O` cannot be surrounded exactly when it can reach the border through other `O`s.

So:

1. Treat `O` cells as graph vertices.
2. Treat four-directional adjacency as graph edges.
3. Start traversal from every border `O`.
4. Mark every reachable `O` as safe.
5. Flip all unmarked `O`s to `X`.
6. Restore safe marks back to `O`.

The core invariant is:

```text
marked safe == originally O and border-reachable
```

Once that invariant is established, the final conversion is forced:

```text
border-reachable O -> keep
not border-reachable O -> capture
```

That is the whole problem reduced to graph reachability.

## Implementation

See `solutions/graph_general/p130_surrounded_regions.py`.

## Tests

See `tests/graph_general/test_p130_surrounded_regions.py`.

## Examples

### Example 1

Input board:

```text
[
  ["X", "X", "X", "X"],
  ["X", "O", "O", "X"],
  ["X", "X", "O", "X"],
  ["X", "O", "X", "X"]
]
```

Output board after in-place modification:

```text
[
  ["X", "X", "X", "X"],
  ["X", "X", "X", "X"],
  ["X", "X", "X", "X"],
  ["X", "O", "X", "X"]
]
```

The middle `O` component is surrounded, but the bottom-row `O` touches the border.

### Example 2

Input board:

```text
[
  ["X"]
]
```

Output board after in-place modification:

```text
[
  ["X"]
]
```

There are no `O` cells to capture.

### Preserved Raw Example Data

- Input: `{'raw': '[["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]\n[["X"]]'}`
- Output: `'See official examples'`

## Follow-up Practice

- Trace the border flood-fill on a board where every cell is `O`.
- Explain why an interior `O` connected to a border `O` must remain unchanged.
- Rewrite the search once with BFS and once with iterative DFS.
- Try the component-by-component brute force approach, then compare its invariant with the border-reachable invariant.
