# 289. Game of Life

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/game-of-life/
- Official Group: Matrix
- Pattern Group: Matrix
- Patterns: matrix

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given a rectangular board where each cell is either:

```text
0 = dead
1 = live
```

Every cell changes at the same time according to the number of live neighbors it has.

A neighbor is any of the eight surrounding positions:

```text
above-left   above   above-right
left         cell    right
below-left   below   below-right
```

The next state of a cell is determined only by:

```text
its current value
+
the number of currently live neighbors
```

The rules are:

1. A live cell with fewer than `2` live neighbors dies.
2. A live cell with `2` or `3` live neighbors lives.
3. A live cell with more than `3` live neighbors dies.
4. A dead cell with exactly `3` live neighbors becomes live.

The important phrase is:

> The next generation is computed simultaneously.

That means when deciding the next state of `board[r][c]`, every neighbor must be interpreted as it was in the original board, not as it may already have been changed during the scan.

So the real problem is:

> Update every cell in-place while still being able to read every cell's original state until all decisions have been made.

---

### 2. Start From the Brute Force Baseline

The most direct safe solution is to create a copy of the board.

1. Copy the original board.
2. For each cell in the original copy:
   - count its live neighbors in the copy;
   - apply the Game of Life rule;
   - write the result into the real board.

Conceptually:

```python
copy = [row[:] for row in board]

for r in range(rows):
    for c in range(cols):
        live_neighbors = count_live_neighbors(copy, r, c)

        if copy[r][c] == 1:
            board[r][c] = 1 if live_neighbors in (2, 3) else 0
        else:
            board[r][c] = 1 if live_neighbors == 3 else 0
```

This is easy to reason about because all reads come from `copy`, and all writes go to `board`.

Correctness is straightforward, but the extra copy costs:

```text
O(m * n) extra space
```

where `m` is the number of rows and `n` is the number of columns.

The follow-up asks for an in-place solution, so we need to remove the copy without losing the old information.

---

### 3. The Key Observation: Each Cell Needs Two Bits of Information

During the update, each cell temporarily needs to answer two questions:

```text
What was my original state?
What should my next state be?
```

The board normally stores only one bit of information:

```text
0 or 1
```

But while computing the transition, we can use extra integer values as temporary encodings.

There are only four possible transitions:

```text
original 0 -> next 0
original 0 -> next 1
original 1 -> next 0
original 1 -> next 1
```

The unchanged cases can keep their normal values:

```text
0 = originally dead, next dead
1 = originally live, next live
```

For the two changing cases, use marker values:

```text
2 = originally live, next dead
3 = originally dead, next live
```

This gives us a compact transition table:

```text
stored value   original state   next state
0              dead             dead
1              live             live
2              live             dead
3              dead             live
```

The exact marker numbers are not magical. What matters is that they let us recover both meanings while the first pass is still running.

---

### 4. The In-Place State Encoding Invariant

The central invariant is:

```text
At all times during the first pass, every cell's original state can still be recovered from its stored value.
```

With the encoding above:

```text
originally live  <=>  value is 1 or 2
originally dead  <=>  value is 0 or 3
```

This is the whole trick.

Suppose we scan left to right, top to bottom. By the time we reach a cell, some earlier neighbors may already have been marked as `2` or `3`.

If we naïvely counted only value `1` as live, we would make a mistake:

```text
2 means the cell is going to die, but it was live originally.
```

Since all updates are simultaneous, a `2` neighbor must still count as live when computing the current cell's next state.

Similarly:

```text
3 means the cell is going to become live, but it was dead originally.
```

A `3` neighbor must not count as live during the first pass.

So the neighbor-counting rule during the first pass is:

```text
count a neighbor as originally live if board[nr][nc] in (1, 2)
```

As long as we use that rule, marking earlier cells does not corrupt later decisions.

---

### 5. Why One Pass Is Not Enough

After the first pass, the board may contain temporary states:

```text
0, 1, 2, 3
```

But the final board must contain only:

```text
0, 1
```

So the algorithm naturally has two passes:

1. First pass: decide every transition and store it using temporary markers.
2. Second pass: collapse each marker to its final state.

The collapse rule is:

```text
0 -> 0
1 -> 1
2 -> 0
3 -> 1
```

Equivalently:

```text
next state is 1 exactly when value is 1 or 3
next state is 0 exactly when value is 0 or 2
```

---

### 6. Detailed Algorithm

Let:

```text
rows = len(board)
cols = len(board[0])
```

For each cell `(r, c)`:

1. Count the number of originally live neighbors.
2. If the cell was originally live:
   - it survives only if the count is `2` or `3`;
   - otherwise mark it as `2`, meaning live-to-dead.
3. If the cell was originally dead:
   - it becomes live only if the count is exactly `3`;
   - mark it as `3`, meaning dead-to-live.
4. Leave unchanged cells as `0` or `1`.

To count neighbors, try all eight direction offsets:

```text
(-1, -1), (-1, 0), (-1, 1),
( 0, -1),          ( 0, 1),
( 1, -1), ( 1, 0), ( 1, 1)
```

For each neighbor coordinate `(nr, nc)`, first check bounds:

```text
0 <= nr < rows
0 <= nc < cols
```

Then count it as live if:

```text
board[nr][nc] == 1 or board[nr][nc] == 2
```

After the first pass, run a second pass over every cell:

```text
if board[r][c] == 2:
    board[r][c] = 0
elif board[r][c] == 3:
    board[r][c] = 1
```

The problem statement asks for the board to be modified in-place, so the method does not need to return a new board.

---

### 7. Pseudocode

```python
def gameOfLife(board):
    rows = len(board)
    cols = len(board[0])

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def was_live(value):
        return value == 1 or value == 2

    for r in range(rows):
        for c in range(cols):
            live_neighbors = 0

            for dr, dc in directions:
                nr = r + dr
                nc = c + dc

                if 0 <= nr < rows and 0 <= nc < cols:
                    if was_live(board[nr][nc]):
                        live_neighbors += 1

            if board[r][c] == 1:
                if live_neighbors < 2 or live_neighbors > 3:
                    board[r][c] = 2      # live -> dead
            else:
                if live_neighbors == 3:
                    board[r][c] = 3      # dead -> live

    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 2:
                board[r][c] = 0
            elif board[r][c] == 3:
                board[r][c] = 1
```

A compact finalization pass is also possible:

```python
board[r][c] = 1 if board[r][c] in (1, 3) else 0
```

The more explicit version is often easier to debug.

---

### 8. Detailed Walkthrough of Example 1

Input:

```text
board = [
  [0, 1, 0],
  [0, 0, 1],
  [1, 1, 1],
  [0, 0, 0]
]
```

We compute the next state of every cell from the original board.

#### Row 0

Cell `(0, 0)` is dead.

Its valid neighbors are:

```text
(0, 1) = 1
(1, 0) = 0
(1, 1) = 0
```

It has `1` live neighbor, so it stays dead:

```text
0 -> 0
```

Cell `(0, 1)` is live.

Its valid neighbors are:

```text
(0, 0) = 0
(0, 2) = 0
(1, 0) = 0
(1, 1) = 0
(1, 2) = 1
```

It has `1` live neighbor, so it dies from underpopulation:

```text
1 -> 0, store 2 temporarily
```

Cell `(0, 2)` is dead.

Even though `(0, 1)` now stores `2`, it was originally live, so it still counts as live during this first pass.

Its valid neighbors are originally:

```text
(0, 1) = live
(1, 1) = dead
(1, 2) = live
```

It has `2` live neighbors, not `3`, so it stays dead:

```text
0 -> 0
```

At this point the board may look partially marked:

```text
[
  [0, 2, 0],
  [0, 0, 1],
  [1, 1, 1],
  [0, 0, 0]
]
```

The `2` is exactly why the invariant matters: it tells later cells that `(0, 1)` was originally live.

#### Row 1

Cell `(1, 0)` is dead.

Its originally live neighbors are:

```text
(0, 1), (2, 0), (2, 1)
```

That is `3` live neighbors, so it becomes live:

```text
0 -> 1, store 3 temporarily
```

Cell `(1, 1)` is dead.

Its originally live neighbors are:

```text
(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)
```

That is `5` live neighbors, so it remains dead:

```text
0 -> 0
```

Cell `(1, 2)` is live.

Its originally live neighbors are:

```text
(0, 1), (2, 1), (2, 2)
```

That is `3` live neighbors, so it survives:

```text
1 -> 1
```

#### Row 2

Cell `(2, 0)` is live.

Its originally live neighbors are:

```text
(2, 1)
```

The cell `(1, 0)` may now store `3`, but `3` means originally dead, so it does not count.

With only `1` live neighbor, `(2, 0)` dies:

```text
1 -> 0, store 2 temporarily
```

Cell `(2, 1)` is live.

Its originally live neighbors are:

```text
(1, 2), (2, 0), (2, 2)
```

That is `3`, so it survives:

```text
1 -> 1
```

Cell `(2, 2)` is live.

Its originally live neighbors are:

```text
(1, 2), (2, 1)
```

That is `2`, so it survives:

```text
1 -> 1
```

#### Row 3

Cell `(3, 0)` is dead.

Its originally live neighbors are:

```text
(2, 0), (2, 1)
```

That is `2`, so it stays dead.

Cell `(3, 1)` is dead.

Its originally live neighbors are:

```text
(2, 0), (2, 1), (2, 2)
```

That is `3`, so it becomes live:

```text
0 -> 1, store 3 temporarily
```

Cell `(3, 2)` is dead.

Its originally live neighbors are:

```text
(2, 1), (2, 2)
```

That is `2`, so it stays dead.

After the first pass, one possible temporary board is:

```text
[
  [0, 2, 0],
  [3, 0, 1],
  [2, 1, 1],
  [0, 3, 0]
]
```

Now collapse temporary states:

```text
2 -> 0
3 -> 1
```

Final board:

```text
[
  [0, 0, 0],
  [1, 0, 1],
  [0, 1, 1],
  [0, 1, 0]
]
```

---

### 9. Correctness Argument

We prove that the algorithm updates the board to exactly the next Game of Life generation.

#### Lemma 1: During the first pass, the original state of every cell is recoverable.

The algorithm uses the following encoding:

```text
0 = originally dead, next dead
1 = originally live, next live
2 = originally live, next dead
3 = originally dead, next live
```

Therefore a cell was originally live exactly when its value is `1` or `2`, and originally dead exactly when its value is `0` or `3`.

The first pass only changes `1` to `2` and `0` to `3`, so every value remains in this encoding. Thus the original state is always recoverable.

#### Lemma 2: For each cell, the algorithm counts exactly its originally live neighbors.

For every cell, the algorithm checks all eight possible neighbor offsets and ignores coordinates outside the board.

For every valid neighbor, it counts the neighbor as live exactly when the stored value is `1` or `2`.

By Lemma 1, values `1` and `2` are exactly the cells that were live in the original board. Therefore the algorithm counts exactly the originally live neighbors.

#### Lemma 3: The first pass stores the correct next state for every cell.

Consider any cell.

By Lemma 2, the algorithm computes the correct number of live neighbors from the original board.

If the cell was originally live, the algorithm keeps it live only for neighbor counts `2` or `3`; otherwise it marks it live-to-dead. This is exactly the Game of Life rule for live cells.

If the cell was originally dead, the algorithm marks it dead-to-live only for neighbor count `3`; otherwise it remains dead. This is exactly the Game of Life rule for dead cells.

Therefore after the first pass, every cell's stored value encodes the correct next state.

#### Lemma 4: The second pass writes exactly the encoded next states.

The second pass maps:

```text
0 -> 0
1 -> 1
2 -> 0
3 -> 1
```

These are exactly the next states represented by the encoding.

#### Theorem: The final board is the correct next Game of Life generation.

By Lemma 3, after the first pass each cell encodes its correct next state. By Lemma 4, the second pass replaces every encoding with that next state. Therefore the final board is exactly the next generation required by the problem.

---

### 10. Complexity

Let:

```text
m = number of rows
n = number of columns
```

The first pass visits every cell once and checks at most eight neighbors per cell:

```text
O(8 * m * n) = O(m * n)
```

The second pass visits every cell once:

```text
O(m * n)
```

Total time:

```text
O(m * n)
```

The algorithm uses only a fixed list of eight directions and a few counters.

Extra space:

```text
O(1)
```

The output is written into the input board itself.

---

### 11. Common Pitfalls

#### Counting the next state instead of the original state

If a neighbor is marked as `2`, it is going to become dead, but it was originally live. It must still count as live during the first pass.

If a neighbor is marked as `3`, it is going to become live, but it was originally dead. It must not count as live during the first pass.

The safe rule is:

```text
originally live means value in (1, 2)
```

#### Updating directly to `0` or `1`

If you immediately change a live cell to `0`, later neighbors cannot know that it used to be live.

If you immediately change a dead cell to `1`, later neighbors may incorrectly count it as originally live.

Temporary markers are what make simultaneous update possible in-place.

#### Forgetting diagonal neighbors

Each cell has up to eight neighbors, not four.

The four diagonal positions are part of the rules.

#### Mixing row and column bounds

Use:

```text
0 <= nr < rows
0 <= nc < cols
```

Do not compare a row index with `cols` or a column index with `rows`.

#### Returning a new board instead of mutating

The LeetCode function is expected to modify `board` in-place. Creating and returning a separate board may be logically correct but does not satisfy the in-place requirement.

#### Mishandling tiny boards

A `1x1` board has no neighbors.

So:

```text
[[1]] -> [[0]]
[[0]] -> [[0]]
```

A single row or single column still uses the same eight-direction loop; the boundary check naturally removes invalid neighbors.

---

### 12. First-Principles Summary

The hard part is not applying the Game of Life rules. The rules are local and direct once the live-neighbor count is known.

The hard part is simultaneity:

```text
Every decision must read the old board,
but every result must be written into the same board.
```

The first-principles solution is to store both pieces of information in each cell temporarily:

```text
original state + next state
```

Using markers:

```text
2 = live -> dead
3 = dead -> live
```

preserves the invariant:

```text
value in (1, 2) means originally live
```

That invariant makes neighbor counting correct even after earlier cells have been marked. Once every transition is encoded, a final pass collapses all cells back to normal `0` and `1` values.

This turns a copy-based simulation into an in-place simulation with the same `O(m * n)` time and only `O(1)` extra space.

## Implementation
See `solutions/matrix/p289_game_of_life.py`.

## Tests
See `tests/matrix/test_p289_game_of_life.py`.

## Examples

### Example 1
- Input: `{'board': [[0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]}`
- Output: `[[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0]]`

### Example 2
- Input: `{'board': [[1, 1], [1, 0]]}`
- Output: `[[1, 1], [1, 1]]`
