# 48. Rotate Image

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/rotate-image/
- Official Group: Matrix
- Pattern Group: Matrix
- Patterns: matrix, in-place, coordinate-transform, layers

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given an `n x n` matrix, rotate it 90 degrees clockwise **in place**.

For example:

```text
matrix = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]
```

After rotating 90 degrees clockwise, the matrix becomes:

```text
[
  [7, 4, 1],
  [8, 5, 2],
  [9, 6, 3]
]
```

The phrase **in place** is the important constraint. We are not supposed to build a second `n x n` matrix and return it. The input matrix itself must be modified.

So the real problem is:

> Move every value to its 90-degree-clockwise destination without losing any value that has not been moved yet.

This is a coordinate problem, not a sorting problem, not a graph problem, and not a search problem. Every cell already has a deterministic destination. The only challenge is performing those moves safely inside the same matrix.

---

### 2. The Coordinate Rule for a Clockwise Rotation

Use zero-based coordinates:

```text
(row, col)
```

In an `n x n` matrix, a 90-degree clockwise rotation sends:

```text
(row, col) -> (col, n - 1 - row)
```

Why?

Take the top-left cell in a `3 x 3` matrix:

```text
(0, 0)
```

After rotation, it moves to the top-right corner:

```text
(0, 2)
```

The rule gives:

```text
(row, col) = (0, 0)
(col, n - 1 - row) = (0, 2)
```

Take the top-middle cell:

```text
(0, 1)
```

It moves to the right-middle cell:

```text
(1, 2)
```

The rule gives:

```text
(col, n - 1 - row) = (1, 2)
```

So the full mapping is:

```text
old matrix coordinate        new matrix coordinate
(row, col)              ->   (col, n - 1 - row)
```

If extra space were allowed, the solution would be very simple:

```python
rotated[col][n - 1 - row] = matrix[row][col]
```

But we must do this in place.

---

### 3. Brute Force Baseline With Extra Space

The simplest correct idea is to allocate a new matrix:

```python
n = len(matrix)
rotated = [[0] * n for _ in range(n)]

for row in range(n):
    for col in range(n):
        rotated[col][n - 1 - row] = matrix[row][col]

for row in range(n):
    for col in range(n):
        matrix[row][col] = rotated[row][col]
```

This is easy to reason about:

1. Every original value is read from `matrix[row][col]`.
2. It is written to its rotated position in `rotated`.
3. The completed rotated matrix is copied back.

This takes:

```text
Time:  O(n^2)
Space: O(n^2)
```

The time is already optimal because there are `n^2` cells to move or verify. The space is the part that violates the spirit of the problem.

The in-place solution asks:

> Can we apply the same coordinate rule while using only a few temporary variables?

---

### 4. Key Observation: Four Cells Form a Cycle

Apply the coordinate rule repeatedly to one coordinate:

```text
(row, col)
      -> (col, n - 1 - row)
      -> (n - 1 - row, n - 1 - col)
      -> (n - 1 - col, row)
      -> (row, col)
```

After four moves, we return to the starting coordinate. That means rotation decomposes the matrix into groups of four cells.

For a `4 x 4` matrix, look at the four corners:

```text
(0, 0) -> (0, 3) -> (3, 3) -> (3, 0) -> (0, 0)
```

The values move around this cycle:

```text
top-left     goes to top-right
top-right    goes to bottom-right
bottom-right goes to bottom-left
bottom-left  goes to top-left
```

So instead of thinking about all cells at once, we can rotate one four-cell cycle at a time.

That is the first-principles breakthrough:

> A 90-degree rotation is not a sequence of independent single-cell writes. It is a sequence of four-cell cyclic swaps.

---

### 5. Why Direct Assignment Can Corrupt Data

Suppose we try to move the top-left corner directly:

```python
matrix[0][3] = matrix[0][0]
```

Now the original value at `(0, 3)` is gone. But that value is still needed because it must move to `(3, 3)`.

This is the core danger of in-place matrix updates:

```text
writing a destination too early can destroy a value that has not moved yet
```

The fix is to rotate a full four-cell cycle using one temporary variable.

For coordinates:

```text
top    = (row, col)
right  = (col, n - 1 - row)
bottom = (n - 1 - row, n - 1 - col)
left   = (n - 1 - col, row)
```

The clockwise movement is:

```text
left   -> top
top    -> right
right  -> bottom
bottom -> left
```

One safe assignment order is:

```python
top_value = matrix[row][col]

matrix[row][col] = matrix[n - 1 - col][row]
matrix[n - 1 - col][row] = matrix[n - 1 - row][n - 1 - col]
matrix[n - 1 - row][n - 1 - col] = matrix[col][n - 1 - row]
matrix[col][n - 1 - row] = top_value
```

This writes the four rotated positions without losing any original value.

---

### 6. Layers: The Matrix Is a Set of Rings

The four-cell cycle idea tells us how to rotate one group of four cells. Now we need to visit every needed group exactly once.

A square matrix can be viewed as nested square layers:

```text
4 x 4 matrix indices:

layer 0: outer ring
(0,0) (0,1) (0,2) (0,3)
(1,0)             (1,3)
(2,0)             (2,3)
(3,0) (3,1) (3,2) (3,3)

layer 1: inner ring
      (1,1) (1,2)
      (2,1) (2,2)
```

Rotating the matrix means rotating each layer.

For layer `layer`:

```text
first = layer
last  = n - 1 - layer
```

The top edge of that layer runs from:

```text
(first, first) through (first, last)
```

But we should not rotate all cells on the top edge. If we include the top-right corner as a starting point, that corner is already part of the cycle started by the top-left corner.

So for each layer, we start cycles at:

```text
row = first
col = first, first + 1, ..., last - 1
```

That is exactly one starting cell for each four-cell cycle in the layer.

---

### 7. The Coordinate and Layer Invariant

At the start of processing a layer:

```text
all layers outside the current layer are already correctly rotated
all layers inside the current layer have not been touched yet
```

While processing the top edge of the current layer, for each offset from the left side:

```text
offset = col - first
```

we rotate exactly these four cells:

```text
top:    (first,        first + offset)
right:  (first+offset, last)
bottom: (last,         last - offset)
left:   (last-offset,  first)
```

These are the same coordinates as the general rotation rule, specialized to the current layer.

The invariant during the inner loop is:

```text
all cycles with smaller offsets in this layer are already correctly rotated
all cycles with current or larger offsets still contain their original values
```

Because each cycle contains four distinct cells and no two starting columns on the top edge produce the same cycle, rotating one cycle does not interfere with another.

That is why the algorithm can safely move left to right across the top edge of each layer.

---

### 8. Detailed Algorithm

Let:

```text
n = len(matrix)
```

There are only `n // 2` layers that need work.

- If `n` is even, all cells belong to some rotating layer.
- If `n` is odd, the center cell stays where it is.

For each layer:

1. Compute `first = layer`.
2. Compute `last = n - 1 - layer`.
3. For each `col` from `first` to `last - 1`:
   1. Compute `offset = col - first`.
   2. Save the top value.
   3. Move left into top.
   4. Move bottom into left.
   5. Move right into bottom.
   6. Move saved top into right.

Pseudocode:

```python
def rotate(matrix):
    n = len(matrix)

    for layer in range(n // 2):
        first = layer
        last = n - 1 - layer

        for col in range(first, last):
            offset = col - first

            top = matrix[first][col]

            matrix[first][col] = matrix[last - offset][first]
            matrix[last - offset][first] = matrix[last][last - offset]
            matrix[last][last - offset] = matrix[col][last]
            matrix[col][last] = top
```

This function mutates `matrix` and returns nothing, matching the LeetCode requirement.

---

### 9. Example Walkthrough: `3 x 3`

Start with:

```text
[
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]
```

Here:

```text
n = 3
n // 2 = 1
```

So there is one layer: `layer = 0`.

```text
first = 0
last  = 2
```

The inner loop uses:

```text
col = 0, 1
```

#### Cycle 1: `col = 0`

```text
offset = 0
```

The four coordinates are:

```text
top:    (0, 0) = 1
right:  (0, 2) = 3
bottom: (2, 2) = 9
left:   (2, 0) = 7
```

After clockwise rotation:

```text
left   -> top     7 moves to (0, 0)
top    -> right   1 moves to (0, 2)
right  -> bottom  3 moves to (2, 2)
bottom -> left    9 moves to (2, 0)
```

Matrix becomes:

```text
[
  [7, 2, 1],
  [4, 5, 6],
  [9, 8, 3]
]
```

#### Cycle 2: `col = 1`

```text
offset = 1
```

The four coordinates are:

```text
top:    (0, 1) = 2
right:  (1, 2) = 6
bottom: (2, 1) = 8
left:   (1, 0) = 4
```

After clockwise rotation:

```text
left   -> top     4 moves to (0, 1)
top    -> right   2 moves to (1, 2)
right  -> bottom  6 moves to (2, 1)
bottom -> left    8 moves to (1, 0)
```

Matrix becomes:

```text
[
  [7, 4, 1],
  [8, 5, 2],
  [9, 6, 3]
]
```

The center cell `5` never moves, which is correct for an odd-sized matrix.

---

### 10. Example Walkthrough: `4 x 4`

Start with:

```text
[
  [ 5,  1,  9, 11],
  [ 2,  4,  8, 10],
  [13,  3,  6,  7],
  [15, 14, 12, 16]
]
```

There are:

```text
n // 2 = 2 layers
```

#### Outer layer: `layer = 0`

```text
first = 0
last  = 3
col   = 0, 1, 2
```

The three cycles are:

```text
(0,0), (0,3), (3,3), (3,0)
(0,1), (1,3), (3,2), (2,0)
(0,2), (2,3), (3,1), (1,0)
```

After rotating the outer layer, the boundary is correct:

```text
[
  [15, 13,  2,  5],
  [14,  4,  8,  1],
  [12,  3,  6,  9],
  [16,  7, 10, 11]
]
```

#### Inner layer: `layer = 1`

```text
first = 1
last  = 2
col   = 1
```

The only inner cycle is:

```text
(1,1), (1,2), (2,2), (2,1)
```

Values before rotating that cycle:

```text
(1,1) = 4
(1,2) = 8
(2,2) = 6
(2,1) = 3
```

After rotation:

```text
3 -> (1,1)
4 -> (1,2)
8 -> (2,2)
6 -> (2,1)
```

Final matrix:

```text
[
  [15, 13,  2,  5],
  [14,  3,  4,  1],
  [12,  6,  8,  9],
  [16,  7, 10, 11]
]
```

---

### 11. Correctness Argument

We prove that the algorithm rotates the matrix 90 degrees clockwise in place.

#### Lemma 1: Each four-cell cycle is rotated correctly.

For a fixed layer and offset, the algorithm identifies four coordinates:

```text
top:    (first,        first + offset)
right:  (first+offset, last)
bottom: (last,         last - offset)
left:   (last-offset,  first)
```

A clockwise rotation sends:

```text
left   -> top
top    -> right
right  -> bottom
bottom -> left
```

The algorithm performs exactly those assignments, using a temporary variable to preserve the original top value until it is written into the right position. Therefore that cycle is rotated correctly.

#### Lemma 2: The algorithm visits every non-center cell exactly once as part of one cycle.

Every cell outside the center of an odd-sized matrix belongs to exactly one square layer. Within a layer, every rotating cycle contains exactly one cell on that layer's top edge, excluding the top-right corner. The inner loop iterates exactly over those top-edge starting cells:

```text
col = first, first + 1, ..., last - 1
```

So each cycle in the layer is processed once, and no cycle is processed twice.

#### Lemma 3: Rotating one cycle does not corrupt another unprocessed cycle.

Different offsets in the same layer produce disjoint sets of four coordinates. Different layers are also disjoint. Since the algorithm only writes the four cells of the current cycle, it cannot overwrite a value belonging to any unprocessed cycle.

#### Theorem: The final matrix is the original matrix rotated 90 degrees clockwise.

By Lemma 2, every cell that should move is included in exactly one processed cycle. By Lemma 1, each processed cycle places its four values into their correct clockwise destinations. By Lemma 3, these placements are not later corrupted by unrelated cycles. The center cell of an odd-sized matrix maps to itself and correctly remains unchanged. Therefore, when the algorithm finishes, every cell contains exactly the value required by a 90-degree clockwise rotation.

---

### 12. Complexity

Let `n` be the side length of the matrix.

Each cell is moved a constant number of times. The algorithm processes all `n^2` cells except possibly the center cell when `n` is odd.

```text
Time:  O(n^2)
Space: O(1)
```

The space is `O(1)` because the algorithm uses only a few integer variables and one temporary saved value, regardless of matrix size.

---

### 13. Common Pitfalls

#### Pitfall 1: Using the wrong coordinate transform

Clockwise rotation is:

```text
(row, col) -> (col, n - 1 - row)
```

Counterclockwise rotation is different:

```text
(row, col) -> (n - 1 - col, row)
```

Mixing these up rotates in the wrong direction.

#### Pitfall 2: Iterating too far on each layer

For a layer, the inner loop should stop before `last`:

```python
for col in range(first, last):
```

Using `range(first, last + 1)` double-processes corner cycles and breaks the result.

#### Pitfall 3: Forgetting the offset

Inside an inner layer, `col` is not the same as the distance from the layer's left edge. The offset is:

```python
offset = col - first
```

This matters when `layer > 0`.

#### Pitfall 4: Overwriting a needed value

This assignment is unsafe by itself:

```python
matrix[col][last] = matrix[first][col]
```

because the old value at `matrix[col][last]` still needs to move. Save one value and rotate all four cells as a group.

#### Pitfall 5: Returning a new matrix

LeetCode expects the input matrix to be modified in place. A helper matrix may produce the right visual answer, but it does not satisfy the intended constraint.

#### Pitfall 6: Mishandling `1 x 1`

For `n = 1`:

```text
n // 2 = 0
```

The loops do not run, and the matrix remains unchanged. That is correct.

---

### 14. First-Principles Summary

A rotation is a coordinate transformation:

```text
(row, col) -> (col, n - 1 - row)
```

Doing that with extra space is straightforward, but doing it in place requires respecting data dependencies. Applying the coordinate rule four times returns to the starting cell, so the matrix decomposes into independent four-cell cycles. Those cycles are naturally organized by square layers.

The algorithm is therefore:

```text
for each layer:
    for each top-edge starting position in that layer:
        rotate the corresponding four-cell cycle
```

The invariant is that completed outer layers and completed earlier cycles already contain their final rotated values, while unprocessed cycles still contain their original values. A single temporary variable is enough because each local operation only needs to preserve one value while the other three are shifted into place.

This is the essential matrix lesson from the problem:

> When an in-place grid update would overwrite needed information, find the cycle structure of the coordinate transform and rotate one cycle at a time.

## Implementation

See `solutions/matrix/p048_rotate_image.py`.

## Tests

See `tests/matrix/test_p048_rotate_image.py`.

## Examples

### Example 1

- Input: `{'matrix': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}`
- Output: `[[7, 4, 1], [8, 5, 2], [9, 6, 3]]`

### Example 2

- Input: `{'matrix': [[5, 1, 9, 11], [2, 4, 8, 10], [13, 3, 6, 7], [15, 14, 12, 16]]}`
- Output: `[[15, 13, 2, 5], [14, 3, 4, 1], [12, 6, 8, 9], [16, 7, 10, 11]]`

## Follow-up Practice

- Trace the coordinate rule on a `2 x 2`, `3 x 3`, and `4 x 4` matrix.
- Write out the four coordinates in one cycle before writing code.
- Explain why the inner loop stops at `last` instead of `last + 1`.
- Implement the alternative two-step method: transpose the matrix, then reverse each row.
