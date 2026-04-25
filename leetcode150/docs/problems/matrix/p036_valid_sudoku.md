# 36. Valid Sudoku

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/valid-sudoku/
- Official Group: Matrix
- Pattern Group: Matrix
- Patterns: matrix

## First-Principles Explanation

### What The Problem Is Asking
You are given a partially filled `9 x 9` Sudoku board. Each cell contains either:

- a digit from `'1'` through `'9'`, or
- `'.'`, meaning the cell is empty.

The task is to decide whether the digits that are already placed obey the Sudoku rules:

1. No row may contain the same digit twice.
2. No column may contain the same digit twice.
3. No `3 x 3` box may contain the same digit twice.

The board does **not** need to be solvable. We are not asked to fill empty cells, search for a completed board, or prove that a solution exists. We only validate the current placements. Empty cells are ignored because they do not place any digit into a row, column, or box.

So the problem is really a duplicate-detection problem over three different groupings of the same cells: rows, columns, and boxes.

### Brute-Force Baseline
A direct baseline is to check every group independently:

1. For each of the 9 rows, scan its 9 cells and detect repeated digits.
2. For each of the 9 columns, scan its 9 cells and detect repeated digits.
3. For each of the 9 boxes, scan its 9 cells and detect repeated digits.

That works well because the board size is fixed. The main inconvenience is that it repeats similar logic three times and requires careful box iteration.

A row check might look like:

```text
for each row:
    seen = empty set
    for each cell in that row:
        if cell is '.': skip it
        if cell is already in seen: return false
        add cell to seen
```

Then the same idea is repeated for columns and boxes.

This baseline is already `O(1)` in LeetCode terms because the board is always `9 x 9`, but conceptually it scans a constant number of groups of constant size. If generalized to an `n x n` Sudoku-like board, it is linear in the number of cells per pass.

### Key Observation
Every filled cell creates exactly three obligations at once.

If digit `d` appears at coordinate `(row, col)`, then:

- row `row` must not have seen `d` before,
- column `col` must not have seen `d` before,
- the `3 x 3` box containing `(row, col)` must not have seen `d` before.

That means we do not need three separate validation passes. While scanning each filled cell once, we can record which digits have already appeared in each row, column, and box. If the current digit is already recorded in any of those three places, the board is invalid immediately.

This is the whole problem: maintain three duplicate-detection tables while walking through the board.

### Row, Column, And Box Invariant
After processing some prefix of the board scan, maintain this invariant:

- `rows[r]` contains exactly the digits already seen in row `r`.
- `cols[c]` contains exactly the digits already seen in column `c`.
- `boxes[b]` contains exactly the digits already seen in box `b`.

For a cell `(r, c)` with digit `d`, the placement is valid relative to previously scanned cells if and only if:

```text
d not in rows[r]
d not in cols[c]
d not in boxes[box_index(r, c)]
```

The box index is determined by integer division:

```text
box_index = (r // 3) * 3 + (c // 3)
```

Why this formula works:

- `r // 3` tells which band of three rows the cell is in: top `0`, middle `1`, bottom `2`.
- `c // 3` tells which stack of three columns the cell is in: left `0`, middle `1`, right `2`.
- Multiplying the row band by `3` and adding the column stack numbers the boxes from left to right, top to bottom.

The nine boxes therefore have these indices:

```text
0 0 0 | 1 1 1 | 2 2 2
0 0 0 | 1 1 1 | 2 2 2
0 0 0 | 1 1 1 | 2 2 2
------+-------+------
3 3 3 | 4 4 4 | 5 5 5
3 3 3 | 4 4 4 | 5 5 5
3 3 3 | 4 4 4 | 5 5 5
------+-------+------
6 6 6 | 7 7 7 | 8 8 8
6 6 6 | 7 7 7 | 8 8 8
6 6 6 | 7 7 7 | 8 8 8
```

### Detailed Algorithm
Use three arrays of sets:

- `rows = [set() for _ in range(9)]`
- `cols = [set() for _ in range(9)]`
- `boxes = [set() for _ in range(9)]`

Then scan all coordinates:

1. For each row index `r` from `0` to `8`:
2. For each column index `c` from `0` to `8`:
3. Read `digit = board[r][c]`.
4. If `digit == '.'`, continue because empty cells impose no Sudoku constraint.
5. Compute `box = (r // 3) * 3 + (c // 3)`.
6. If `digit` already appears in `rows[r]`, return `False`.
7. If `digit` already appears in `cols[c]`, return `False`.
8. If `digit` already appears in `boxes[box]`, return `False`.
9. Otherwise, add `digit` to all three sets.
10. If the scan finishes without finding a duplicate, return `True`.

The order of checks does not matter as long as all three are performed before adding the current digit. Adding first would make every filled cell look like a duplicate of itself.

### Pseudocode

```text
function isValidSudoku(board):
    rows = array of 9 empty sets
    cols = array of 9 empty sets
    boxes = array of 9 empty sets

    for r from 0 to 8:
        for c from 0 to 8:
            digit = board[r][c]

            if digit == '.':
                continue

            box = (r // 3) * 3 + (c // 3)

            if digit in rows[r]:
                return false
            if digit in cols[c]:
                return false
            if digit in boxes[box]:
                return false

            add digit to rows[r]
            add digit to cols[c]
            add digit to boxes[box]

    return true
```

### Python Implementation Shape

```python
from typing import List


class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]

        for r in range(9):
            for c in range(9):
                digit = board[r][c]
                if digit == ".":
                    continue

                box = (r // 3) * 3 + (c // 3)
                if digit in rows[r] or digit in cols[c] or digit in boxes[box]:
                    return False

                rows[r].add(digit)
                cols[c].add(digit)
                boxes[box].add(digit)

        return True
```

This code treats digits as strings, which matches the input. There is no need to convert `'5'` to integer `5`; equality and set membership work directly on strings.

### Detailed Example Walkthrough
Consider the first example board:

```text
5 3 . | . 7 . | . . .
6 . . | 1 9 5 | . . .
. 9 8 | . . . | . 6 .
------+-------+------
8 . . | . 6 . | . . 3
4 . . | 8 . 3 | . . 1
7 . . | . 2 . | . . 6
------+-------+------
. 6 . | . . . | 2 8 .
. . . | 4 1 9 | . . 5
. . . | . 8 . | . 7 9
```

Start with all sets empty.

1. Cell `(0, 0)` contains `'5'`.
   - Row `0` has not seen `'5'`.
   - Column `0` has not seen `'5'`.
   - Box `(0 // 3) * 3 + (0 // 3) = 0` has not seen `'5'`.
   - Add `'5'` to `rows[0]`, `cols[0]`, and `boxes[0]`.

2. Cell `(0, 1)` contains `'3'`.
   - Row `0` currently contains `{'5'}`, so `'3'` is allowed.
   - Column `1` is empty.
   - Box `0` currently contains `{'5'}`, so `'3'` is allowed.
   - Add `'3'` to row `0`, column `1`, and box `0`.

3. Cell `(0, 2)` contains `'.'`.
   - Skip it. Empty cells do not create duplicates.

4. Cell `(0, 4)` contains `'7'`.
   - Row `0` currently contains `{'5', '3'}`.
   - Column `4` is empty.
   - Box `(0 // 3) * 3 + (4 // 3) = 1` is empty.
   - Add `'7'` to row `0`, column `4`, and box `1`.

Continue this way. When the scan reaches `(1, 3)` with `'1'`, it is stored in row `1`, column `3`, and box `1`. When it reaches `(1, 4)` with `'9'`, it is stored in row `1`, column `4`, and box `1`. No digit ever appears twice in the same row, column, or box, so the algorithm finishes and returns `True`.

Now compare the second example. Its first row begins with `'8'`, and row `3` also begins with `'8'`:

```text
8 3 . | . 7 . | . . .
6 . . | 1 9 5 | . . .
. 9 8 | . . . | . 6 .
------+-------+------
8 . . | . 6 . | . . 3
...
```

The duplicate is in column `0`:

1. At `(0, 0)`, the algorithm sees `'8'` and records it in `cols[0]`.
2. At `(3, 0)`, the algorithm sees another `'8'`.
3. Before adding it, the algorithm checks `cols[0]` and finds that `'8'` is already present.
4. That violates the column invariant, so the algorithm returns `False` immediately.

It does not matter that the two `'8'` values are in different `3 x 3` boxes. A Sudoku placement must satisfy all three constraints, and one violated constraint is enough to reject the board.

### Correctness
We prove that the algorithm returns `True` if and only if the given board is valid.

First, suppose the algorithm returns `False`. It does so only when processing a filled cell `(r, c)` with digit `d` and finding `d` already in `rows[r]`, `cols[c]`, or `boxes[box]`. By the invariant, that means a previously processed cell in the same row, column, or `3 x 3` box already contained `d`. Therefore the board violates one of the Sudoku rules, so the board is invalid.

Second, suppose the board is invalid. Then some digit appears at least twice in a row, column, or box. Consider the later of those duplicate cells in the scan order. When the algorithm reaches that later cell, the earlier duplicate has already been inserted into the corresponding row, column, or box set. The membership check will find the digit and return `False`. Therefore an invalid board cannot pass the scan.

Finally, if the algorithm completes the scan and returns `True`, then no filled cell ever duplicated a digit already seen in its row, column, or box. Since every filled cell was checked against exactly the three Sudoku constraints that apply to it, every row, column, and box contains no repeated digit. Therefore the board is valid.

### Complexity
The board has exactly `81` cells.

- Time: `O(81)`, usually written as `O(1)` for the fixed-size LeetCode problem. If generalized to an `n x n` board, the scan is `O(n^2)`.
- Space: `O(27 * 9)`, also `O(1)` for the fixed-size board, because there are 9 row sets, 9 column sets, and 9 box sets, each holding at most 9 digits.

### Common Pitfalls
- Checking rows and columns but forgetting the `3 x 3` boxes.
- Computing the box index incorrectly; use `(r // 3) * 3 + (c // 3)`, not `r // 3 + c // 3`.
- Treating `'.'` as a digit; empty cells must be skipped.
- Adding the digit to a set before checking membership, which makes the current cell collide with itself.
- Trying to solve the Sudoku instead of validating only the current board.
- Converting digits unnecessarily and then comparing integers to strings inconsistently.
- Assuming each row must contain all digits `1` through `9`; partial boards are allowed, so missing digits are fine.

### First-Principles Summary
A valid partial Sudoku board is one where every placed digit is unique within each of its three constraint groups: its row, its column, and its `3 x 3` box. Each cell can be validated locally if the scan carries the right history: digits already seen per row, per column, and per box. The moment a digit repeats in any one of those histories, the board is invalid. If all filled cells can be inserted without a repeat, the board satisfies exactly the rules the problem asks us to check.

## Implementation
See `solutions/matrix/p036_valid_sudoku.py`.

## Tests
See `tests/matrix/test_p036_valid_sudoku.py`.

## Examples

### Example 1
- Input: `{'board': [['5', '3', '.', '.', '7', '.', '.', '.', '.'], ['6', '.', '.', '1', '9', '5', '.', '.', '.'], ['.', '9', '8', '.', '.', '.', '.', '6', '.'], ['8', '.', '.', '.', '6', '.', '.', '.', '3'], ['4', '.', '.', '8', '.', '3', '.', '.', '1'], ['7', '.', '.', '.', '2', '.', '.', '.', '6'], ['.', '6', '.', '.', '.', '.', '2', '8', '.'], ['.', '.', '.', '4', '1', '9', '.', '.', '5'], ['.', '.', '.', '.', '8', '.', '.', '7', '9']]}`
- Output: `True`

### Example 2
- Input: `{'board': [['8', '3', '.', '.', '7', '.', '.', '.', '.'], ['6', '.', '.', '1', '9', '5', '.', '.', '.'], ['.', '9', '8', '.', '.', '.', '.', '6', '.'], ['8', '.', '.', '.', '6', '.', '.', '.', '3'], ['4', '.', '.', '8', '.', '3', '.', '.', '1'], ['7', '.', '.', '.', '2', '.', '.', '.', '6'], ['.', '6', '.', '.', '.', '.', '2', '8', '.'], ['.', '.', '.', '4', '1', '9', '.', '.', '5'], ['.', '.', '.', '.', '8', '.', '.', '7', '9']]}`
- Output: `False`

## Follow-up Practice
- Rewrite the solution using one set of triples like `('row', r, digit)`, `('col', c, digit)`, and `('box', box, digit)`.
- Trace the box-index formula for all cells in the top-left, center, and bottom-right boxes.
- Add custom tests where the only violation is in a row, only in a column, and only in a box.
