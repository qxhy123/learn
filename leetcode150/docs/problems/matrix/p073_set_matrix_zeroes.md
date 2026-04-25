# 73. Set Matrix Zeroes

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/set-matrix-zeroes/
- Official Group: Matrix
- Pattern Group: Matrix
- Patterns: matrix

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given an `m x n` matrix, if any cell is originally `0`, then every cell in that same row and every cell in that same column must become `0`.

The update must happen as if all zero positions were discovered from the original matrix at the same time.

For example:

```text
matrix = [
  [1, 1, 1],
  [1, 0, 1],
  [1, 1, 1]
]
```

The zero is at coordinate `(1, 1)`.

So row `1` must become all zeroes:

```text
[0, 0, 0]
```

and column `1` must become all zeroes:

```text
matrix[0][1], matrix[1][1], matrix[2][1]
```

The final matrix is:

```text
[
  [1, 0, 1],
  [0, 0, 0],
  [1, 0, 1]
]
```

The real problem is:

> Find every row and column that contained an original zero, then zero exactly the union of those rows and columns.

The word **original** is the trap. If we write zeroes while we are still discovering zeroes, we can accidentally treat a zero we just wrote as if it had existed in the input.

---

### 2. Why We Cannot Naively Zero Immediately

A tempting approach is:

1. Scan the matrix.
2. When `matrix[row][col] == 0`, immediately zero that entire row and column.
3. Continue scanning.

This is wrong because the scan becomes contaminated by its own writes.

Consider:

```text
matrix = [
  [1, 0, 1],
  [1, 1, 1],
  [1, 1, 1]
]
```

The original zero is only at `(0, 1)`.

If we immediately zero row `0` and column `1`, the matrix becomes:

```text
[
  [0, 0, 0],
  [1, 0, 1],
  [1, 0, 1]
]
```

Now the scan may later see the newly written zero at `(1, 1)` and decide to zero row `1`, even though row `1` did not contain an original zero.

That would spread zeroes farther than the problem allows.

So the first-principles requirement is:

```text
Discovery phase must be separated from mutation phase.
```

We need to remember which rows and columns should be zeroed before performing the destructive update.

---

### 3. Brute-Force Baseline

The simplest correct solution is to use extra storage.

Scan the whole matrix and record:

```text
zero_rows = every row index that contains an original zero
zero_cols = every column index that contains an original zero
```

Then scan the matrix again. For every coordinate `(row, col)`:

```text
if row in zero_rows or col in zero_cols:
    matrix[row][col] = 0
```

Conceptually:

```python
rows = set()
cols = set()

for row in range(m):
    for col in range(n):
        if matrix[row][col] == 0:
            rows.add(row)
            cols.add(col)

for row in range(m):
    for col in range(n):
        if row in rows or col in cols:
            matrix[row][col] = 0
```

This is correct because it records all decisions before changing the matrix.

Its cost is:

```text
Time:  O(m * n)
Space: O(m + n)
```

The follow-up challenge is usually to do it in constant extra space.

That means we still need markers, but we cannot allocate separate row and column marker arrays.

---

### 4. Key Observation: The Matrix Already Has Marker Space

To decide the final value of `matrix[row][col]`, we only need two yes/no facts:

```text
Should this row be zeroed?
Should column col be zeroed?
```

Instead of storing those facts in external sets, we can store them inside the matrix itself.

Where can a row-level marker live?

A natural place is the first cell of that row:

```text
matrix[row][0]
```

Where can a column-level marker live?

A natural place is the first cell of that column:

```text
matrix[0][col]
```

So when we find an original zero at `(row, col)`, we can mark:

```text
matrix[row][0] = 0   # this row must become zero
matrix[0][col] = 0   # this column must become zero
```

Then later, for any interior cell `(row, col)`, we can decide:

```text
if matrix[row][0] == 0 or matrix[0][col] == 0:
    matrix[row][col] = 0
```

This gives us the same information as `zero_rows` and `zero_cols`, but reuses the first row and first column as marker arrays.

---

### 5. The First Row and First Column Need Special Care

The marker idea creates one overlap problem.

The cell `matrix[0][0]` belongs to both:

```text
first row marker area
first column marker area
```

If `matrix[0][0] == 0`, what does that mean?

It could mean:

```text
row 0 must be zeroed
```

or it could mean:

```text
column 0 must be zeroed
```

or both.

One cell cannot reliably store two independent boolean flags.

So we keep separate boolean variables for the first row and first column before using them as marker space:

```text
first_row_zero = whether row 0 originally contains a zero
first_col_zero = whether column 0 originally contains a zero
```

After these two facts are saved, the rest of the matrix can safely use row `0` and column `0` as marker storage.

This is the central invariant of the in-place solution.

---

### 6. Marker Invariant

After the marking pass finishes, the algorithm maintains this invariant:

```text
For every row r > 0:
    matrix[r][0] == 0
    if and only if row r contained an original zero somewhere.

For every column c > 0:
    matrix[0][c] == 0
    if and only if column c contained an original zero somewhere.

first_row_zero tells whether row 0 originally contained a zero.
first_col_zero tells whether column 0 originally contained a zero.
```

The invariant deliberately excludes row `0` from the row markers and column `0` from the column markers, because those two boundary lines are being reused as storage.

Once this invariant is true, every interior cell can be decided locally:

```text
matrix[r][c] should be zero
if row r was marked or column c was marked.
```

That is:

```text
matrix[r][0] == 0 or matrix[0][c] == 0
```

---

### 7. Detailed Algorithm

Let:

```text
m = number of rows
n = number of columns
```

The in-place algorithm has four phases.

#### Phase 1: Remember Whether the First Row and First Column Need Zeroing

Check row `0`:

```python
first_row_zero = any(matrix[0][col] == 0 for col in range(n))
```

Check column `0`:

```python
first_col_zero = any(matrix[row][0] == 0 for row in range(m))
```

These checks must happen before row `0` and column `0` are used as markers.

#### Phase 2: Mark Rows and Columns Using the First Row and First Column

Scan only the interior cells:

```text
rows 1 through m - 1
cols 1 through n - 1
```

When an interior zero is found at `(row, col)`, mark its row and column:

```python
matrix[row][0] = 0
matrix[0][col] = 0
```

Do not zero the whole row or column yet. Only mark.

#### Phase 3: Zero Interior Cells According to the Markers

Scan the same interior region again.

For each interior cell `(row, col)`:

```python
if matrix[row][0] == 0 or matrix[0][col] == 0:
    matrix[row][col] = 0
```

At this point, all non-boundary cells are correct.

#### Phase 4: Zero the First Row and First Column If Needed

Finally, apply the saved boundary flags:

```python
if first_row_zero:
    set every cell in row 0 to 0

if first_col_zero:
    set every cell in column 0 to 0
```

This phase must come last. If we zero the first row too early, we destroy the column markers. If we zero the first column too early, we destroy the row markers.

---

### 8. Pseudocode

```python
def setZeroes(matrix):
    m = len(matrix)
    n = len(matrix[0])

    first_row_zero = False
    for col in range(n):
        if matrix[0][col] == 0:
            first_row_zero = True
            break

    first_col_zero = False
    for row in range(m):
        if matrix[row][0] == 0:
            first_col_zero = True
            break

    for row in range(1, m):
        for col in range(1, n):
            if matrix[row][col] == 0:
                matrix[row][0] = 0
                matrix[0][col] = 0

    for row in range(1, m):
        for col in range(1, n):
            if matrix[row][0] == 0 or matrix[0][col] == 0:
                matrix[row][col] = 0

    if first_row_zero:
        for col in range(n):
            matrix[0][col] = 0

    if first_col_zero:
        for row in range(m):
            matrix[row][0] = 0
```

On LeetCode, the method mutates `matrix` in-place and does not need to return a value.

Some local test scaffolds may compare the returned matrix. In that situation, returning `matrix` after mutation is harmless in Python, but the core algorithm is still an in-place mutation.

---

### 9. Detailed Example Walkthrough

Use the second official example:

```text
matrix = [
  [0, 1, 2, 0],
  [3, 4, 5, 2],
  [1, 3, 1, 5]
]
```

Here:

```text
m = 3
n = 4
```

#### Step 1: Check the First Row

The first row is:

```text
[0, 1, 2, 0]
```

It contains zeroes, so:

```text
first_row_zero = True
```

#### Step 2: Check the First Column

The first column is:

```text
[0, 3, 1]
```

It contains a zero, so:

```text
first_col_zero = True
```

#### Step 3: Mark Interior Zeroes

The interior cells are rows `1..2` and columns `1..3`:

```text
[4, 5, 2]
[3, 1, 5]
```

There are no zeroes in this interior region.

So the matrix remains:

```text
[
  [0, 1, 2, 0],
  [3, 4, 5, 2],
  [1, 3, 1, 5]
]
```

But notice that the first row already contains a zero at column `3`. Since row `0` is the column marker area, this means:

```text
column 3 must become zero
```

The first column contains a zero at row `0`, but whether column `0` itself must be zeroed is stored in:

```text
first_col_zero = True
```

#### Step 4: Zero Interior Cells from Markers

For each interior cell, check its row marker and column marker.

Cell `(1, 1)`:

```text
row marker:    matrix[1][0] = 3
column marker: matrix[0][1] = 1
```

Neither marker is zero, so it stays `4`.

Cell `(1, 2)`:

```text
matrix[1][0] = 3
matrix[0][2] = 2
```

It stays `5`.

Cell `(1, 3)`:

```text
matrix[1][0] = 3
matrix[0][3] = 0
```

Column `3` is marked, so it becomes `0`.

Row `2` behaves similarly:

```text
(2, 1) stays 3
(2, 2) stays 1
(2, 3) becomes 0
```

Now the matrix is:

```text
[
  [0, 1, 2, 0],
  [3, 4, 5, 0],
  [1, 3, 1, 0]
]
```

#### Step 5: Zero the First Row

Because:

```text
first_row_zero = True
```

row `0` becomes all zeroes:

```text
[
  [0, 0, 0, 0],
  [3, 4, 5, 0],
  [1, 3, 1, 0]
]
```

#### Step 6: Zero the First Column

Because:

```text
first_col_zero = True
```

column `0` becomes all zeroes:

```text
[
  [0, 0, 0, 0],
  [0, 4, 5, 0],
  [0, 3, 1, 0]
]
```

This is the required output.

---

### 10. Another Walkthrough With an Interior Zero

Consider:

```text
matrix = [
  [1, 1, 1],
  [1, 0, 1],
  [1, 1, 1]
]
```

First row has no zero:

```text
first_row_zero = False
```

First column has no zero:

```text
first_col_zero = False
```

Now scan the interior. At `(1, 1)`, we find zero, so mark:

```text
matrix[1][0] = 0
matrix[0][1] = 0
```

The matrix becomes:

```text
[
  [1, 0, 1],
  [0, 0, 1],
  [1, 1, 1]
]
```

Now use markers to update interior cells.

For `(1, 2)`:

```text
matrix[1][0] == 0
```

so it becomes zero.

For `(2, 1)`:

```text
matrix[0][1] == 0
```

so it becomes zero.

The interior-updated matrix is:

```text
[
  [1, 0, 1],
  [0, 0, 0],
  [1, 0, 1]
]
```

The saved flags are both false, so we do not zero the entire first row or entire first column.

Final result:

```text
[
  [1, 0, 1],
  [0, 0, 0],
  [1, 0, 1]
]
```

---

### 11. Correctness Argument

We prove that the algorithm produces exactly the matrix required by the problem.

#### Lemma 1: The saved boundary flags correctly describe the original first row and first column.

`first_row_zero` is computed before any mutation by scanning every cell in row `0`. Therefore it is true exactly when row `0` originally contained a zero.

`first_col_zero` is computed before any mutation by scanning every cell in column `0`. Therefore it is true exactly when column `0` originally contained a zero.

#### Lemma 2: After the marking phase, every non-first row that originally contained a zero is marked.

Take any row `r > 0` that originally contained a zero.

If that zero is in column `0`, then `matrix[r][0]` is already zero, so row `r` is marked.

If that zero is in some column `c > 0`, the marking scan visits `(r, c)` and sets `matrix[r][0] = 0`, so row `r` is marked.

Thus every non-first row that originally contained a zero is marked.

#### Lemma 3: After the marking phase, no non-first row is falsely marked unless it should be zeroed.

The algorithm sets `matrix[r][0] = 0` only when it finds an original zero in row `r` during the interior scan, or when `matrix[r][0]` was already an original zero.

Both cases mean row `r` originally contained a zero.

So a row marker never creates a false row-zero decision.

#### Lemma 4: After the marking phase, every non-first column that originally contained a zero is marked, and no non-first column is falsely marked.

The argument is symmetric to Lemmas 2 and 3.

For any column `c > 0`, `matrix[0][c]` is zero after marking exactly when column `c` originally contained a zero.

#### Lemma 5: Every interior cell is set to zero exactly when required.

Consider an interior cell `(r, c)` where `r > 0` and `c > 0`.

The problem requires this cell to become zero if and only if:

```text
row r originally contained a zero
or
column c originally contained a zero
```

By Lemmas 2 through 4, those two facts are represented exactly by:

```text
matrix[r][0] == 0
or
matrix[0][c] == 0
```

The algorithm uses exactly this condition when updating interior cells.

Therefore every interior cell is correct.

#### Lemma 6: Every first-row and first-column cell is set correctly.

By Lemma 1, `first_row_zero` exactly determines whether the entire first row must be zeroed.

If it is true, the algorithm zeroes every cell in row `0`. If it is false, the algorithm does not zero row `0` as a whole; individual cells in row `0` may already be zero because they are valid column markers for columns that must be zeroed.

Similarly, `first_col_zero` exactly determines whether the entire first column must be zeroed.

Therefore all boundary cells are correct.

#### Theorem: The final matrix satisfies the problem requirement.

Every cell is either an interior cell or lies in the first row or first column.

By Lemma 5, every interior cell is correct.

By Lemma 6, every first-row and first-column cell is correct.

Therefore every cell in the final matrix is zero exactly when its original row or original column contained a zero, which is precisely the required transformation.

---

### 12. Complexity

Let:

```text
m = number of rows
n = number of columns
```

The algorithm scans:

1. The first row: `O(n)`.
2. The first column: `O(m)`.
3. The interior for marking: `O(m * n)` in the worst case.
4. The interior for updating: `O(m * n)` in the worst case.
5. The first row and first column again if needed: `O(m + n)`.

So the total time is:

```text
O(m * n)
```

The extra space is only two boolean flags:

```text
O(1)
```

The matrix itself is reused as marker storage, which does not count as extra space.

---

### 13. Common Pitfalls

- **Zeroing too early:** If you zero rows and columns during discovery, newly written zeroes can spread incorrectly.
- **Forgetting the first row or first column:** They cannot be handled only by `matrix[0][0]`; two independent flags are needed.
- **Applying boundary zeroing before interior updates:** Zeroing row `0` or column `0` too soon destroys the markers before they are used.
- **Scanning from index `0` during marker-based interior updates:** The marker logic is meant for rows `1..m-1` and columns `1..n-1`; boundaries are handled separately.
- **Returning the wrong thing on LeetCode:** The official method mutates in place and returns `None`. If a local scaffold expects a returned matrix, return it only after performing the same in-place mutation.
- **Assuming `matrix[0][0]` can store everything:** It cannot distinguish “first row should be zero” from “first column should be zero.”

---

### 14. First-Principles Summary

The problem is not fundamentally about writing zeroes. It is about preserving the original zero information long enough to make all writes safely.

A row/column should be zeroed if it contained an original zero. The brute-force solution stores those row and column decisions in external sets. The optimized solution stores the same decisions inside the matrix: first column cells mark rows, and first row cells mark columns.

Because the first row and first column are both data and marker storage, their original zero status must be saved before marking begins. After that, the algorithm has a clean invariant:

```text
row marker OR column marker determines each interior cell
```

Then the first row and first column are applied last from the saved flags.

This separation of discovery, marking, interior update, and boundary update is what prevents accidental zero propagation.

## Implementation
See `solutions/matrix/p073_set_matrix_zeroes.py`.

## Tests
See `tests/matrix/test_p073_set_matrix_zeroes.py`.

## Examples

### Example 1
- Input: `{'matrix': [[1, 1, 1], [1, 0, 1], [1, 1, 1]]}`
- Output: `[[1, 0, 1], [0, 0, 0], [1, 0, 1]]`

### Example 2
- Input: `{'matrix': [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]}`
- Output: `[[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]]`

## Follow-up Practice
- Trace the algorithm on a `1 x 1` matrix containing `0`.
- Trace the algorithm on a `1 x n` matrix where the first row is also the marker row.
- Trace the algorithm on an `m x 1` matrix where the first column is also the marker column.
- Explain why the first row and first column must be zeroed after the interior cells, not before.
