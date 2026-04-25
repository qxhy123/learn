# 52. N-Queens II

- Difficulty: Hard
- LeetCode: https://leetcode.com/problems/n-queens-ii/
- Official Group: Backtracking
- Pattern Group: Backtracking
- Patterns: backtracking, constraint-pruning

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

We are given an integer `n`.

An `n x n` chessboard has:

```text
n rows
n columns
```

A queen attacks every square in:

```text
its row
its column
its two diagonals
```

The problem asks:

> How many different ways are there to place exactly `n` queens on an `n x n` board so that no two queens attack each other?

For `n = 4`, the answer is `2`.

One valid board is:

```text
.Q..
...Q
Q...
..Q.
```

Another valid board is:

```text
..Q.
Q...
...Q
.Q..
```

So:

```text
totalNQueens(4) = 2
```

This is different from LeetCode 51, N-Queens, where we must return the actual boards. Here we only need the count.

That difference matters: we do not need to store strings for every solution. We only need to count how many complete valid placements exist.

---

### 2. Start From the Board Constraints

A placement is valid if every queen is isolated from every other queen by three rules:

1. No two queens share a row.
2. No two queens share a column.
3. No two queens share a diagonal.

The row rule is the easiest one to exploit.

Since we need exactly `n` queens on `n` rows, and no two queens may share a row, every valid solution must have:

```text
exactly one queen in each row
```

So instead of asking:

> Which `n` squares of the board should contain queens?

we can ask the smaller question:

> For each row, which column should receive that row's queen?

For example, for `n = 4`, this placement:

```text
.Q..
...Q
Q...
..Q.
```

can be represented as:

```text
row 0 -> column 1
row 1 -> column 3
row 2 -> column 0
row 3 -> column 2
```

or simply:

```text
[1, 3, 0, 2]
```

That representation already guarantees one queen per row. The remaining work is to prevent column and diagonal conflicts.

---

### 3. The Brute-Force Baseline

A very direct brute-force approach is:

1. For row `0`, choose any column from `0` to `n - 1`.
2. For row `1`, choose any column from `0` to `n - 1`.
3. Continue until all rows have a chosen column.
4. Check whether the resulting placement is valid.
5. Count it if it is valid.

In pseudocode:

```python
count = 0

for col0 in range(n):
    for col1 in range(n):
        for col2 in range(n):
            ...
                placement = [col0, col1, col2, ...]
                if placement_is_valid(placement):
                    count += 1
```

There are `n` choices for each of `n` rows, so this explores:

```text
n^n
```

possible row-to-column assignments.

Most of these assignments are obviously impossible. For example, if row `0` and row `1` both choose column `2`, the placement can never become valid, no matter what later rows do.

So the first improvement is:

> Do not wait until the board is complete to discover a conflict. Reject a choice as soon as it conflicts with queens already placed.

That is the reason backtracking fits this problem.

---

### 4. The Key Observation: Build One Safe Row at a Time

Suppose we place queens from top to bottom.

At some moment, we have already placed queens in rows:

```text
0, 1, 2, ..., row - 1
```

Now we are deciding where to place the queen in:

```text
row
```

Because previous rows already contain one queen each, the only question is:

> Which columns in this row are not attacked by any previous queen?

If a column is safe, we place a queen there and recurse to the next row.

If a column is not safe, we skip it immediately.

This creates a decision tree where each level corresponds to one row:

```text
level 0: choose column for row 0
level 1: choose column for row 1
level 2: choose column for row 2
...
level n: all rows filled, found one valid solution
```

The important property is that we never create a partial board that already has attacking queens.

---

### 5. How to Test Whether a Square Is Attacked

For a square `(row, col)`, it is unsafe if any previous queen uses:

```text
the same column
or the same main diagonal
or the same anti-diagonal
```

The column rule is direct:

```text
same column -> same col
```

The diagonal rules need a compact way to identify diagonals.

#### Main Diagonals

A main diagonal runs from top-left to bottom-right.

Along such a diagonal, this value stays constant:

```text
row - col
```

Example:

```text
(0, 1) -> -1
(1, 2) -> -1
(2, 3) -> -1
```

All those squares are on the same main diagonal.

#### Anti-Diagonals

An anti-diagonal runs from top-right to bottom-left.

Along such a diagonal, this value stays constant:

```text
row + col
```

Example:

```text
(0, 2) -> 2
(1, 1) -> 2
(2, 0) -> 2
```

All those squares are on the same anti-diagonal.

So a square `(row, col)` is safe exactly when:

```text
col       is not already used
row - col is not already used
row + col is not already used
```

This lets us check safety in constant time with three sets.

---

### 6. Recursion State and Invariant

The recursive state can be:

```text
row        = the next row to fill
columns    = columns already occupied by previous queens
diag_down  = row - col values already occupied by previous queens
diag_up    = row + col values already occupied by previous queens
```

At the start of a recursive call `backtrack(row)`, the invariant is:

> Rows `0` through `row - 1` each contain exactly one queen, and no two of those queens attack each other. The three sets describe exactly the columns and diagonals occupied by those queens.

This invariant is the whole algorithm.

If it is true when we enter `backtrack(row)`, then for every candidate column `col`:

- If `col`, `row - col`, or `row + col` is already used, placing a queen at `(row, col)` would break the invariant, so we skip it.
- Otherwise, placing a queen at `(row, col)` preserves the invariant for `backtrack(row + 1)`.

When `row == n`, the invariant says rows `0` through `n - 1` have all been filled safely. That is one complete valid arrangement, so we add `1` to the answer.

---

### 7. Detailed Algorithm

The algorithm is:

1. Create three empty sets:
   - `columns`
   - `diag_down` for `row - col`
   - `diag_up` for `row + col`
2. Define a recursive function `backtrack(row)`.
3. If `row == n`, return `1` because a full valid placement has been built.
4. Otherwise, initialize `total = 0`.
5. For each `col` from `0` to `n - 1`:
   - Compute `down = row - col`.
   - Compute `up = row + col`.
   - If any of `col`, `down`, or `up` is already used, skip this column.
   - Otherwise:
     - Add `col` to `columns`.
     - Add `down` to `diag_down`.
     - Add `up` to `diag_up`.
     - Recursively count solutions from `row + 1`.
     - Remove the same three values to restore the previous state.
6. Return `total`.

The add-recursive-call-remove pattern is what makes this backtracking rather than greedy search.

We are not committing permanently to one column. We are temporarily exploring the world where that column was chosen, then undoing it so the next column can be explored from a clean state.

---

### 8. Pseudocode

```python
def totalNQueens(n):
    columns = set()
    diag_down = set()  # row - col
    diag_up = set()    # row + col

    def backtrack(row):
        if row == n:
            return 1

        total = 0

        for col in range(n):
            down = row - col
            up = row + col

            if col in columns:
                continue
            if down in diag_down:
                continue
            if up in diag_up:
                continue

            columns.add(col)
            diag_down.add(down)
            diag_up.add(up)

            total += backtrack(row + 1)

            columns.remove(col)
            diag_down.remove(down)
            diag_up.remove(up)

        return total

    return backtrack(0)
```

A direct implementation can use a nonlocal `answer` counter instead of returning counts. Returning counts often makes the recursion easier to reason about because each call answers:

> How many valid completions exist from this partial board?

---

### 9. Walkthrough for `n = 4`

Let rows and columns be zero-indexed.

We start with:

```text
row = 0
columns = {}
diag_down = {}
diag_up = {}
```

#### Try row 0, column 0

Place a queen at `(0, 0)`.

```text
Q...
....
....
....
```

Used constraints:

```text
columns = {0}
diag_down = {0 - 0 = 0}
diag_up = {0 + 0 = 0}
```

Now move to row `1`.

- Column `0` is blocked by the same column.
- Column `1` is blocked because `1 - 1 = 0`, same main diagonal.
- Column `2` is safe.

Place `(1, 2)`:

```text
Q...
..Q.
....
....
```

Now row `2` has no safe column:

- Column `0` is used.
- Column `1` is attacked by `(1, 2)` on an anti-diagonal.
- Column `2` is used.
- Column `3` is attacked by `(1, 2)` on a main diagonal.

So this branch cannot produce a solution. We undo `(1, 2)` and try the next choice in row `1`.

#### Continue row 1, column 3

From only `(0, 0)` placed, try `(1, 3)`:

```text
Q...
...Q
....
....
```

In row `2`, column `1` is safe, so place `(2, 1)`:

```text
Q...
...Q
.Q..
....
```

But then row `3` has no safe column. This branch also fails.

Now all options after `(0, 0)` have failed, so we undo `(0, 0)`.

#### Try row 0, column 1

Place `(0, 1)`:

```text
.Q..
....
....
....
```

For row `1`, columns `0`, `1`, and `2` are attacked or occupied, but column `3` is safe.

Place `(1, 3)`:

```text
.Q..
...Q
....
....
```

For row `2`, column `0` is safe.

Place `(2, 0)`:

```text
.Q..
...Q
Q...
....
```

For row `3`, column `2` is safe.

Place `(3, 2)`:

```text
.Q..
...Q
Q...
..Q.
```

Now `row == 4`, meaning all four rows are filled. This is one valid solution, so the count increases by `1`.

The recursion then backtracks and keeps searching.

#### Try row 0, column 2

A symmetric branch eventually finds the second valid solution:

```text
..Q.
Q...
...Q
.Q..
```

#### Try row 0, column 3

This branch produces no additional solution.

After every branch is explored, the total is:

```text
2
```

So `totalNQueens(4)` returns `2`.

---

### 10. Why the Algorithm Is Correct

We prove correctness using the recursion invariant.

#### Invariant

At the start of `backtrack(row)`, rows `0` through `row - 1` contain exactly one queen each, no two of those queens attack each other, and the three sets contain exactly their occupied columns and diagonals.

#### Initialization

Before the first call, `row = 0` and no queens have been placed.

The invariant is true because there are no conflicting queens, and all three sets are empty.

#### Maintenance

Assume the invariant is true at the start of `backtrack(row)`.

For a candidate column `col`, the algorithm checks:

```text
col not in columns
row - col not in diag_down
row + col not in diag_up
```

If any check fails, then placing a queen at `(row, col)` would conflict with an existing queen, so skipping that choice cannot remove a valid solution.

If all checks pass, the new queen does not share a column or diagonal with any previous queen. It also cannot share a row because previous queens are only in earlier rows and this is the only queen placed in the current row.

After adding the three constraint values, the invariant is true for `backtrack(row + 1)`.

After the recursive call returns, removing those same values restores the exact state that existed before trying `(row, col)`, so later choices are explored independently.

#### Completion

When `row == n`, the invariant says rows `0` through `n - 1` each contain one queen and no two queens attack each other. Therefore the algorithm counts one valid solution.

#### Exhaustiveness

Take any valid N-Queens solution. It has exactly one queen in every row. When the algorithm reaches each row, the column used by that solution will pass the safety checks because the solution has no conflicts. Therefore the algorithm has a branch that follows exactly that solution and counts it.

#### No Double Counting

Each complete branch chooses exactly one column for each row. That sequence of column choices uniquely identifies one board. The recursion visits each such sequence at most once, so no valid board is counted twice.

Therefore the final count is exactly the number of valid N-Queens arrangements.

---

### 11. Complexity

Let `n` be the board size.

The recursion places at most one queen per row, so the maximum recursion depth is:

```text
O(n)
```

At each row, the algorithm tries up to `n` columns, and each safety check is `O(1)` because it uses sets.

A loose upper bound is:

```text
O(n^n)
```

because there are at most `n` choices across `n` rows.

A tighter common bound is:

```text
O(n!)
```

because the `columns` set prevents using the same column twice, so after choosing columns for earlier rows, fewer columns remain available.

Diagonal pruning usually reduces the actual search much further, but the number of explored states is still exponential.

Space complexity is:

```text
O(n)
```

The recursion stack has depth `n`, and each of the three sets stores at most `n` values.

Because this version only returns a count, it does not need output storage proportional to the number of solutions.

---

### 12. Common Pitfalls

#### Forgetting That This Problem Returns a Count

For N-Queens II, we do not need to build every board as strings.

Building boards is still correct, but it does unnecessary work. The recursive search can simply return or increment a count when `row == n`.

#### Checking Diagonals by Scanning the Board

A slower approach scans previous queen positions to see whether a new queen is diagonal to them.

That works, but it makes each safety check more expensive. The first-principles diagonal identities:

```text
row - col
row + col
```

turn diagonal checks into constant-time set lookups.

#### Using the Wrong Diagonal Formula

For main diagonals, use:

```text
row - col
```

For anti-diagonals, use:

```text
row + col
```

Mixing these up is less dangerous than forgetting one entirely, but using only one formula will miss conflicts.

#### Not Undoing State

After exploring a choice, all three sets must be restored:

```python
columns.remove(col)
diag_down.remove(row - col)
diag_up.remove(row + col)
```

If one value is not removed, later branches will incorrectly think a column or diagonal is still occupied.

#### Counting Too Early

A partial placement is not a solution.

Only count when:

```text
row == n
```

That is the point where every row has a queen and the invariant guarantees the board is valid.

#### Thinking Symmetry Is Required

The board has symmetries, and advanced solutions can use symmetry to reduce work.

But symmetry is not required for a clean accepted solution. The essential idea is the row-by-row invariant with column and diagonal pruning.

---

### 13. First-Principles Summary

The problem looks like a chess problem, but the core is a constrained counting problem.

A valid board must have one queen per row, so the natural decision is:

```text
choose one column for the next row
```

A choice is safe only if it does not reuse:

```text
column
row - col diagonal
row + col diagonal
```

Backtracking works because it explores every safe partial placement, abandons a branch immediately when a queen would be attacked, and restores state after each temporary choice.

The invariant is:

> The rows already processed contain non-attacking queens, and the constraint sets exactly describe what future queens must avoid.

Once that invariant is clear, the algorithm follows directly: try each safe column, recurse to the next row, undo the choice, and count every time all rows are filled.

## Implementation

See `solutions/backtracking/p052_n_queens_ii.py`.

## Tests

See `tests/backtracking/test_p052_n_queens_ii.py`.

## Examples

### Example 1
- Input: `{'n': 4}`
- Output: `2`

### Example 2
- Input: `{'n': 1}`
- Output: `1`

## Follow-up Practice

- Trace the sets for `n = 4` until the first solution is counted.
- Explain why `row - col` identifies one diagonal direction.
- Explain why the algorithm never needs to place more than one queen in a row.
- Rewrite the recursion so it uses a nonlocal counter instead of returning counts.
