# 427. Construct Quad Tree

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/construct-quad-tree/
- Official Group: Divide & Conquer
- Pattern Group: Divide & Conquer
- Patterns: divide-conquer, tree-traversal

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an `n x n` binary grid, where every cell is either:

```text
0
1
```

The grid size `n` is a power of two. That matters because it means every square region can be repeatedly split into exactly four equal smaller squares until each region is a single cell.

The task is to build a **quad tree** representation of the grid.

A quad tree node has six fields:

```python
val
isLeaf
topLeft
topRight
bottomLeft
bottomRight
```

There are two kinds of nodes.

A **leaf node** represents a square region where every cell has the same value:

```text
isLeaf = True
val = True   if the whole region is all 1
val = False  if the whole region is all 0
children = None
```

An **internal node** represents a square region that contains both `0` and `1` somewhere inside it:

```text
isLeaf = False
val = either True or False; LeetCode accepts either value here
topLeft = quad tree for top-left quadrant
topRight = quad tree for top-right quadrant
bottomLeft = quad tree for bottom-left quadrant
bottomRight = quad tree for bottom-right quadrant
```

So the problem is not asking us to search for a value or compute a number. It is asking us to **compress a square binary image into a tree**.

The compression rule is simple:

> If a square region is uniform, store one leaf. Otherwise split the square into four equal quadrants and repeat the same rule for each quadrant.

### 2. Why a Quad Tree Is Natural Here

A normal binary tree splits a structure into two parts.

A quad tree splits a two-dimensional square into four parts:

```text
+-------------+--------------+
| top-left    | top-right    |
+-------------+--------------+
| bottom-left | bottom-right |
+-------------+--------------+
```

For a grid region with top-left coordinate `(row, col)` and side length `size`, the four quadrants have side length:

```text
half = size // 2
```

Their coordinates are:

```text
top-left:     (row,        col)
top-right:    (row,        col + half)
bottom-left:  (row + half, col)
bottom-right: (row + half, col + half)
```

That coordinate rule is the backbone of the whole algorithm.

### 3. Start From the Baseline Idea

The most direct way to construct the tree is:

1. Given a square region, check whether all cells in that region are equal.
2. If they are all equal, return one leaf node.
3. Otherwise, split the region into four quadrants.
4. Recursively build one child node for each quadrant.
5. Return an internal node pointing to those four children.

Conceptually:

```python
build(row, col, size):
    if grid[row:row+size][col:col+size] is all one value:
        return Leaf(value)

    half = size // 2
    return Internal(
        build(row, col, half),
        build(row, col + half, half),
        build(row + half, col, half),
        build(row + half, col + half, half),
    )
```

This is already a correct first solution.

The only expensive part is the uniformity check. For every recursive region, we may scan all cells inside that region to decide whether it is a leaf.

### 4. The Key Observation

A quad tree node always represents a **square subgrid**.

Therefore every recursive call can be described completely by three numbers:

```text
row  = top row of the square
col  = left column of the square
size = side length of the square
```

We do not need to copy subgrids.

That is important. If each recursive call creates sliced grid copies, the code becomes slower and uses extra memory. The grid is fixed; the recursive state only needs to point to a region inside it.

The second observation is the leaf rule:

> A region becomes a leaf exactly when every cell in that region equals the region's first cell.

So to test a region, choose:

```text
target = grid[row][col]
```

Then scan:

```text
for r in row ... row + size - 1
    for c in col ... col + size - 1
        if grid[r][c] != target:
            not uniform
```

If no mismatch is found, the entire square is represented by one leaf node.

### 5. The Quadrant Invariant

The main invariant is:

```text
build(row, col, size) returns the root of a correct quad tree for exactly
this square region:

rows row through row + size - 1
columns col through col + size - 1
```

That invariant includes three smaller promises.

First, the call never represents a rectangle. It always represents a square.

Second, `size` is always a power of two, so if `size > 1`, splitting into `size // 2` produces four equal square quadrants.

Third, the children are attached in the exact order required by the problem:

```text
topLeft, topRight, bottomLeft, bottomRight
```

This order is easy to get wrong because the coordinate changes are similar. The invariant prevents that confusion: every child must represent one precise quadrant of the parent square.

### 6. The Detailed Algorithm

Use a recursive helper:

```python
build(row, col, size)
```

For each call:

1. Read the first value of the region:

   ```python
   first = grid[row][col]
   ```

2. Scan the square region.

3. If every cell equals `first`, return a leaf node:

   ```python
   Node(bool(first), True, None, None, None, None)
   ```

4. Otherwise the region contains both values, so it cannot be compressed into one leaf.

5. Split it into four quadrants:

   ```python
   half = size // 2
   ```

6. Recursively build children:

   ```python
   top_left = build(row, col, half)
   top_right = build(row, col + half, half)
   bottom_left = build(row + half, col, half)
   bottom_right = build(row + half, col + half, half)
   ```

7. Return an internal node:

   ```python
   Node(True, False, top_left, top_right, bottom_left, bottom_right)
   ```

For internal nodes, `val` does not affect correctness on LeetCode. Many solutions use `True` by convention.

### 7. Pseudocode

```python
class Solution:
    def construct(self, grid: List[List[int]]) -> 'Node':
        n = len(grid)

        def build(row: int, col: int, size: int) -> 'Node':
            first = grid[row][col]
            uniform = True

            for r in range(row, row + size):
                for c in range(col, col + size):
                    if grid[r][c] != first:
                        uniform = False
                        break
                if not uniform:
                    break

            if uniform:
                return Node(bool(first), True, None, None, None, None)

            half = size // 2
            top_left = build(row, col, half)
            top_right = build(row, col + half, half)
            bottom_left = build(row + half, col, half)
            bottom_right = build(row + half, col + half, half)

            return Node(
                True,
                False,
                top_left,
                top_right,
                bottom_left,
                bottom_right,
            )

        return build(0, 0, n)
```

This is the clearest version. It favors direct reasoning over micro-optimization.

### 8. Walkthrough: Example 1

Input:

```text
grid = [
  [0, 1],
  [1, 0]
]
```

The whole grid is a `2 x 2` square.

Start with:

```text
build(0, 0, 2)
```

The first cell is:

```text
grid[0][0] = 0
```

Scan the full square:

```text
0 1
1 0
```

The region is not uniform because it contains both `0` and `1`.

So we split it into four `1 x 1` quadrants:

```text
top-left:     build(0, 0, 1) -> cell 0
top-right:    build(0, 1, 1) -> cell 1
bottom-left:  build(1, 0, 1) -> cell 1
bottom-right: build(1, 1, 1) -> cell 0
```

Every `1 x 1` region is automatically uniform, so each one becomes a leaf:

```text
top-left leaf:     val = 0
top-right leaf:    val = 1
bottom-left leaf:  val = 1
bottom-right leaf: val = 0
```

The root is an internal node because the original `2 x 2` region was mixed.

Tree shape:

```text
internal
├── topLeft:     leaf 0
├── topRight:    leaf 1
├── bottomLeft:  leaf 1
└── bottomRight: leaf 0
```

That matches the serialized output style shown in the example:

```text
[[0, 1], [1, 0], [1, 1], [1, 1], [1, 0]]
```

In that serialization, each node is represented as:

```text
[isLeaf, val]
```

So:

```text
[0, 1] = internal node, val shown as 1 by convention
[1, 0] = leaf with value 0
[1, 1] = leaf with value 1
```

### 9. Walkthrough: Example 2

Input:

```text
1 1 1 1 0 0 0 0
1 1 1 1 0 0 0 0
1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1
1 1 1 1 0 0 0 0
1 1 1 1 0 0 0 0
1 1 1 1 0 0 0 0
1 1 1 1 0 0 0 0
```

Start with the full `8 x 8` region:

```text
build(0, 0, 8)
```

It contains both `1` and `0`, so the root is internal.

Split into four `4 x 4` quadrants.

#### Top-left `4 x 4`

Rows `0..3`, columns `0..3`:

```text
1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 1
```

Uniform all `1`, so this whole quadrant becomes one leaf.

#### Top-right `4 x 4`

Rows `0..3`, columns `4..7`:

```text
0 0 0 0
0 0 0 0
1 1 1 1
1 1 1 1
```

Mixed, so this quadrant becomes an internal node.

It splits into four `2 x 2` quadrants:

```text
top-left:     all 0 -> leaf 0
top-right:    all 0 -> leaf 0
bottom-left:  all 1 -> leaf 1
bottom-right: all 1 -> leaf 1
```

#### Bottom-left `4 x 4`

Rows `4..7`, columns `0..3`:

```text
1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 1
```

Uniform all `1`, so this becomes one leaf.

#### Bottom-right `4 x 4`

Rows `4..7`, columns `4..7`:

```text
0 0 0 0
0 0 0 0
0 0 0 0
0 0 0 0
```

Uniform all `0`, so this becomes one leaf.

The important point is that the algorithm does not keep splitting uniform regions. Once it proves a region is all one value, it compresses the entire square into a single leaf.

### 10. Correctness Proof

We prove that `build(row, col, size)` returns a correct quad tree for the square region it represents.

#### Base Case

If the region is uniform, every cell in it has the same value `first`.

The algorithm returns a leaf node with:

```text
isLeaf = True
val = first
```

A leaf node is exactly the quad tree representation of a uniform region. Therefore the returned node is correct.

This also covers every `1 x 1` region, because a single cell is always uniform.

#### Recursive Case

If the region is not uniform, it contains both `0` and `1`. Therefore it cannot be represented by one leaf.

The quad tree definition requires this mixed square to be represented as an internal node with four children corresponding to the four equal quadrants.

The algorithm computes:

```text
half = size // 2
```

and recursively builds exactly those four quadrants:

```text
top-left     square
top-right    square
bottom-left  square
bottom-right square
```

Each quadrant is smaller than the original region. By the induction hypothesis, each recursive call returns a correct quad tree for its quadrant.

The algorithm attaches those four correct child trees to an internal node in the required order. Therefore the returned node correctly represents the whole mixed region.

#### Conclusion

The initial call is:

```text
build(0, 0, n)
```

That region is the entire input grid. Since the helper is correct for every region it is called on, the returned root is a correct quad tree for the whole grid.

### 11. Complexity

Let `n` be the side length of the grid, so the grid contains `n^2` cells.

In the straightforward implementation, each recursive node scans its whole region to check whether it is uniform.

At one recursion level, the regions are disjoint and together cover the whole grid, so a full level can scan at most:

```text
n^2 cells
```

There are:

```text
log2(n) + 1
```

levels, because the side length halves each time.

So the worst-case time complexity is:

```text
O(n^2 log n)
```

This worst case happens when regions keep being mixed, forcing many levels of recursive splitting.

The output tree itself can contain `O(n^2)` nodes in the worst case, because a highly alternating grid may need many leaves.

Auxiliary recursion space is:

```text
O(log n)
```

because the recursion depth is the number of times the side length can be halved.

If counting the returned tree as space, total space is:

```text
O(n^2)
```

#### Prefix Sum Optimization

There is also an optimization using a two-dimensional prefix sum.

For any square region, compute the number of `1` cells inside it in `O(1)` time. If the sum is:

```text
0
```

then the region is all `0`.

If the sum is:

```text
size * size
```

then the region is all `1`.

Otherwise it is mixed.

With that optimization:

```text
Time:  O(n^2)
Space: O(n^2) for the prefix sum, plus output tree space
```

The simpler scanning version is usually accepted and is easier to explain. The prefix sum version is useful when you want the uniformity check to be constant time.

### 12. Common Pitfalls

- **Copying subgrids:** Passing sliced grids into recursion is unnecessary and can add large hidden memory and time costs. Pass `(row, col, size)` instead.
- **Wrong child order:** The problem expects `topLeft`, `topRight`, `bottomLeft`, `bottomRight`. Swapping bottom-left and top-right creates a structurally wrong tree.
- **Forgetting that internal `val` is arbitrary:** For non-leaf nodes, LeetCode accepts either `True` or `False`. Do not use internal `val` to encode extra meaning.
- **Stopping only at `size == 1`:** That works but misses compression. You must stop early for any uniform region, not only single cells.
- **Using rectangles instead of squares:** Every node represents a square. The side length halves in both row and column directions.
- **Off-by-one scan bounds:** The region is half-open: rows `row` through `row + size - 1`, columns `col` through `col + size - 1`. In Python, that is `range(row, row + size)` and `range(col, col + size)`.
- **Returning primitive values instead of nodes:** The answer is the root `Node`, not the serialized list shown in examples.

### 13. First-Principles Summary

This problem follows from a small set of ideas:

```text
1. A quad tree node represents one square region of the grid.
2. A uniform square can be compressed into one leaf node.
3. A mixed square must be split into four equal square quadrants.
4. Each recursive call is fully described by row, col, and size.
5. The required invariant is: build(row, col, size) returns the correct tree for exactly that square.
6. The four children must preserve the physical quadrant order: top-left, top-right, bottom-left, bottom-right.
```

So the whole algorithm is:

> Look at the current square. If all cells match, return one leaf. Otherwise split the square into four quadrants, recursively build their trees, and return an internal node connecting them in quadrant order.

## Implementation
See `solutions/divide_conquer/p427_construct_quad_tree.py`.

## Tests
See `tests/divide_conquer/test_p427_construct_quad_tree.py`.

## Examples

### Example 1
- Input: `{'grid': [[0, 1], [1, 0]]}`
- Output: `[[0, 1], [1, 0], [1, 1], [1, 1], [1, 0]]`

### Example 2
- Input: `{'grid': [[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]]}`
- Output: `[[0, 1], [1, 1], [0, 1], [1, 1], [1, 0], None, None, None, None, [1, 0], [1, 0], [1, 1], [1, 1]]`

## Follow-up Practice
- Trace the four coordinate ranges for a `4 x 4` grid by hand.
- Implement the direct scanning version first.
- Then implement the prefix-sum uniformity check and compare the complexity.
- Test an all-zero grid, an all-one grid, a checkerboard grid, and a grid where only one quadrant is mixed.
