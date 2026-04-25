# 74. Search a 2D Matrix

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/search-a-2d-matrix/
- Official Group: Binary Search
- Pattern Group: Binary Search
- Patterns: binary-search

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given a matrix of integers and a `target` value.

The matrix has two important ordering rules:

1. Each row is sorted from left to right.
2. The first number of each row is greater than the last number of the previous row.

For example:

```text
matrix = [
  [ 1,  3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 60]
]
```

The first row is sorted:

```text
1 < 3 < 5 < 7
```

The second row is sorted:

```text
10 < 11 < 16 < 20
```

And the first value of the second row is greater than the last value of the first row:

```text
10 > 7
```

The same relationship holds between the second and third rows:

```text
23 > 20
```

The question is:

> Does `target` appear anywhere in this matrix?

Return `True` if it appears, and `False` otherwise.

This is not asking for the position of the target. It only asks whether the value exists.

---

### 2. Start From the Baseline Idea

The most direct method is to check every cell.

Conceptually:

```python
for row in matrix:
    for value in row:
        if value == target:
            return True

return False
```

This is correct because it visits every possible location where `target` could be.

If the matrix has:

```text
m rows
n columns
```

then there are:

```text
m * n
```

cells.

So the brute-force time complexity is:

```text
O(m * n)
```

The brute-force algorithm ignores the sorted structure. It treats the matrix like an unordered pile of numbers.

The real opportunity is to use the ordering rules to rule out many values at once.

---

### 3. The Key Observation: The Matrix Is One Sorted List

The matrix looks two-dimensional, but its ordering rules make it behave like one long sorted array.

Take the example matrix:

```text
[
  [ 1,  3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 60]
]
```

If we read it row by row, we get:

```text
[1, 3, 5, 7, 10, 11, 16, 20, 23, 30, 34, 60]
```

That list is fully sorted.

This happens because:

- inside a row, values increase from left to right;
- after a row ends, the next row starts with a larger value than anything in the previous row.

So the matrix does not merely contain sorted rows. The rows connect together into one global sorted order.

That means the problem can be reduced to:

> Search for `target` in a sorted array of length `m * n`, without physically building that array.

Once we see that, binary search becomes the natural tool.

---

### 4. How a Virtual 1D Index Maps Back to the Matrix

Suppose the matrix has `n` columns.

A flattened index counts positions as if all rows were laid end to end.

For this matrix:

```text
matrix = [
  [ 1,  3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 60]
]
```

there are `4` columns, so the flattened positions are:

```text
flat index:  0   1   2   3    4   5   6   7    8   9  10  11
value:       1   3   5   7   10  11  16  20   23  30  34  60
row:         0   0   0   0    1   1   1   1    2   2   2   2
col:         0   1   2   3    0   1   2   3    0   1   2   3
```

For any flattened index `i`:

```text
row = i // n
col = i % n
```

Why?

- `i // n` tells us how many complete rows come before index `i`.
- `i % n` tells us the offset inside the current row.

For example, with `n = 4`:

```text
i = 6
row = 6 // 4 = 1
col = 6 % 4 = 2
```

So flattened index `6` maps to:

```text
matrix[1][2] = 16
```

This lets us binary search over numbers `0` through `m * n - 1` while still reading actual matrix values.

---

### 5. Search Invariant

Use an inclusive binary-search interval:

```text
left ... right
```

The invariant is:

> If `target` is present in the matrix, then its flattened index is somewhere between `left` and `right`, inclusive.

Initially:

```text
left = 0
right = m * n - 1
```

So the interval covers every cell.

On each step:

1. Choose the middle flattened index `mid`.
2. Convert it into `(row, col)`.
3. Read `matrix[row][col]`.
4. Compare it with `target`.

There are three cases.

#### Case 1: `matrix[row][col] == target`

The target has been found, so return `True` immediately.

#### Case 2: `matrix[row][col] < target`

Because the flattened order is sorted, everything at index `mid` or before is less than or equal to this value.

If `matrix[row][col]` is already too small, then every flattened index:

```text
0 ... mid
```

is too small as well.

So `target` cannot be in that part.

Move the search interval rightward:

```text
left = mid + 1
```

The invariant is preserved because we only discarded positions that cannot contain `target`.

#### Case 3: `matrix[row][col] > target`

Because the flattened order is sorted, everything at index `mid` or after is greater than or equal to this value.

If `matrix[row][col]` is already too large, then every flattened index:

```text
mid ... m * n - 1
```

is too large as well.

So `target` cannot be in that part.

Move the search interval leftward:

```text
right = mid - 1
```

Again, the invariant is preserved.

When the loop ends, `left > right`. The interval of possible positions is empty. Since every possible location has been ruled out, return `False`.

---

### 6. Detailed Algorithm

Let:

```text
m = number of rows
n = number of columns
```

Then:

1. Set `left = 0`.
2. Set `right = m * n - 1`.
3. While `left <= right`:
   - Compute `mid = (left + right) // 2`.
   - Convert `mid` to matrix coordinates:
     - `row = mid // n`
     - `col = mid % n`
   - Let `value = matrix[row][col]`.
   - If `value == target`, return `True`.
   - If `value < target`, discard the left half by setting `left = mid + 1`.
   - If `value > target`, discard the right half by setting `right = mid - 1`.
4. If the loop finishes, return `False`.

The matrix itself is never modified.

We also do not need to allocate a flattened copy of the matrix. The flattened array exists only as a mental model.

---

### 7. Pseudocode

```python
def searchMatrix(matrix, target):
    m = len(matrix)
    n = len(matrix[0])

    left = 0
    right = m * n - 1

    while left <= right:
        mid = (left + right) // 2

        row = mid // n
        col = mid % n
        value = matrix[row][col]

        if value == target:
            return True

        if value < target:
            left = mid + 1
        else:
            right = mid - 1

    return False
```

This is ordinary binary search. The only extra idea is the conversion from a virtual one-dimensional index to a real two-dimensional coordinate.

---

### 8. Walkthrough: Target Exists

Use Example 1:

```text
matrix = [
  [ 1,  3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 60]
]
target = 3
```

There are:

```text
m = 3 rows
n = 4 columns
m * n = 12 cells
```

So the initial flattened search interval is:

```text
left = 0
right = 11
```

The virtual sorted array is:

```text
index:  0  1  2  3   4   5   6   7   8   9  10  11
value:  1  3  5  7  10  11  16  20  23  30  34  60
```

#### Step 1

```text
left = 0
right = 11
mid = (0 + 11) // 2 = 5
```

Convert index `5`:

```text
row = 5 // 4 = 1
col = 5 % 4 = 1
```

So:

```text
matrix[1][1] = 11
```

Compare:

```text
11 > 3
```

The target, if it exists, must be before index `5`.

Update:

```text
right = 4
```

#### Step 2

```text
left = 0
right = 4
mid = (0 + 4) // 2 = 2
```

Convert index `2`:

```text
row = 2 // 4 = 0
col = 2 % 4 = 2
```

So:

```text
matrix[0][2] = 5
```

Compare:

```text
5 > 3
```

The target must be before index `2`.

Update:

```text
right = 1
```

#### Step 3

```text
left = 0
right = 1
mid = (0 + 1) // 2 = 0
```

Convert index `0`:

```text
row = 0 // 4 = 0
col = 0 % 4 = 0
```

So:

```text
matrix[0][0] = 1
```

Compare:

```text
1 < 3
```

The target must be after index `0`.

Update:

```text
left = 1
```

#### Step 4

```text
left = 1
right = 1
mid = (1 + 1) // 2 = 1
```

Convert index `1`:

```text
row = 1 // 4 = 0
col = 1 % 4 = 1
```

So:

```text
matrix[0][1] = 3
```

Compare:

```text
3 == 3
```

Return:

```text
True
```

---

### 9. Walkthrough: Target Does Not Exist

Use Example 2:

```text
matrix = [
  [ 1,  3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 60]
]
target = 13
```

Again:

```text
left = 0
right = 11
```

#### Step 1

```text
mid = 5
matrix[5 // 4][5 % 4] = matrix[1][1] = 11
```

Compare:

```text
11 < 13
```

Everything through index `5` is too small.

Update:

```text
left = 6
```

#### Step 2

```text
left = 6
right = 11
mid = (6 + 11) // 2 = 8
matrix[8 // 4][8 % 4] = matrix[2][0] = 23
```

Compare:

```text
23 > 13
```

Everything from index `8` onward is too large.

Update:

```text
right = 7
```

#### Step 3

```text
left = 6
right = 7
mid = (6 + 7) // 2 = 6
matrix[6 // 4][6 % 4] = matrix[1][2] = 16
```

Compare:

```text
16 > 13
```

Update:

```text
right = 5
```

Now:

```text
left = 6
right = 5
```

The interval is empty.

Return:

```text
False
```

This result makes sense because `13` would have to be between `11` and `16`, but no such cell exists.

---

### 10. Correctness Argument

We prove that the algorithm returns `True` exactly when `target` appears in the matrix.

#### The Flattened Order Is Sorted

Because every row is sorted, moving right within a row never decreases the value.

Because the first value of each row is greater than the last value of the previous row, moving from the end of one row to the beginning of the next row also increases the value.

Therefore, if the matrix is read row by row, the resulting virtual array is sorted in increasing order.

#### The Invariant Holds Initially

At the start:

```text
left = 0
right = m * n - 1
```

Every matrix cell corresponds to exactly one flattened index in that range.

So if `target` exists, its index is inside the search interval.

#### Each Update Preserves the Invariant

On each iteration, the algorithm examines `mid`.

If `matrix[mid] == target`, the algorithm returns `True`, which is correct.

If `matrix[mid] < target`, then every index less than or equal to `mid` contains a value less than or equal to `matrix[mid]`, so all of those values are too small. None can be `target`. Setting `left = mid + 1` keeps every possible target position inside the remaining interval.

If `matrix[mid] > target`, then every index greater than or equal to `mid` contains a value greater than or equal to `matrix[mid]`, so all of those values are too large. None can be `target`. Setting `right = mid - 1` keeps every possible target position inside the remaining interval.

Thus, after every update, the invariant remains true:

> If `target` exists, its flattened index is between `left` and `right`.

#### Termination Gives the Correct Answer

The loop stops only when:

```text
left > right
```

At that point, there are no candidate indices left.

By the invariant, if `target` existed, it would have to be inside the interval. But the interval is empty.

Therefore, `target` does not exist in the matrix, and returning `False` is correct.

So the algorithm is correct.

---

### 11. Complexity

Let:

```text
m = number of rows
n = number of columns
```

There are `m * n` cells.

Binary search halves the remaining candidate interval on each iteration, so the number of iterations is:

```text
O(log(m * n))
```

Each iteration does constant work:

- compute `mid`;
- convert `mid` to `(row, col)`;
- read one matrix value;
- compare it with `target`.

So the time complexity is:

```text
O(log(m * n))
```

The algorithm uses only a few integer variables, so the space complexity is:

```text
O(1)
```

---

### 12. Common Pitfalls

#### Pitfall 1: Binary Searching Each Row Separately

You could binary search every row, but that costs:

```text
O(m * log n)
```

That is better than checking every cell, but it misses the stronger fact that all rows together form one sorted sequence.

The virtual-array approach uses the full ordering and runs in:

```text
O(log(m * n))
```

#### Pitfall 2: Forgetting the Column Count in the Mapping

The conversion must use the number of columns:

```python
row = mid // n
col = mid % n
```

Do not divide or mod by the number of rows.

For an `m x n` matrix, each row contains `n` elements, so every block of `n` flattened indices belongs to one row.

#### Pitfall 3: Building a Real Flattened List

This works logically:

```python
flat = []
for row in matrix:
    flat.extend(row)
```

Then binary search `flat`.

But it uses extra memory:

```text
O(m * n)
```

The virtual-index method gets the same sorted-array behavior without copying anything.

#### Pitfall 4: Mixing Boundary Styles

This tutorial uses an inclusive interval:

```text
[left, right]
```

So the loop condition is:

```python
while left <= right:
```

and updates are:

```python
left = mid + 1
right = mid - 1
```

If you instead use a half-open interval `[left, right)`, the loop condition and updates must change. Mixing the two styles often causes infinite loops or skipped candidates.

#### Pitfall 5: Treating the Matrix Like LeetCode 240

There is another common problem, “Search a 2D Matrix II,” where rows and columns are sorted, but the rows do not necessarily connect into one sorted array.

This problem is different.

Here, the first element of each row is greater than the last element of the previous row. That stronger rule is exactly what allows one global binary search over `m * n` positions.

---

### 13. First-Principles Summary

The matrix is two-dimensional in storage, but one-dimensional in order.

The row and cross-row ordering rules imply that reading the matrix left to right, top to bottom produces a sorted array.

So the core move is:

```text
replace “search a matrix” with “binary search a virtual sorted array”
```

The algorithm does not need to reshape the data. It only needs a way to translate a virtual index back to the original matrix:

```text
row = index // number_of_columns
col = index % number_of_columns
```

From there, every binary-search comparison has the usual meaning:

- if the middle value is too small, discard everything before it;
- if the middle value is too large, discard everything after it;
- if it equals the target, the answer is found.

The maintained invariant is that any possible target position remains inside the current flattened index interval. When that interval becomes empty, all possibilities have been eliminated.

## Implementation
See `solutions/binary_search/p074_search_a_2d_matrix.py`.

## Tests
See `tests/binary_search/test_p074_search_a_2d_matrix.py`.

## Examples

### Example 1
- Input: `{'matrix': [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 'target': 3}`
- Output: `True`

### Example 2
- Input: `{'matrix': [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 'target': 13}`
- Output: `False`

## Follow-up Practice
- Rewrite the flattened-index invariant before coding.
- Trace the mapping from `mid` to `(row, col)` on a non-square matrix.
- Compare the inclusive `[left, right]` version with a half-open `[left, right)` version.
- Explain why this problem allows one binary search, while a matrix with only row-wise sorting would not.
