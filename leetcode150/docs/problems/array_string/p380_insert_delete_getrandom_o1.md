# 380. Insert Delete GetRandom O(1)

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/insert-delete-getrandom-o1/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: design, hash-map, dynamic-array

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Design a data structure called `RandomizedSet` that stores **unique integers** and supports three operations:

```text
insert(val)    add val if it is not already present
remove(val)    remove val if it is present
getRandom()    return one currently stored value uniformly at random
```

Each operation must run in average `O(1)` time.

The word **set** matters because each value can appear at most once.

The word **random** also matters. `getRandom()` is not allowed to return any arbitrary value. If the set currently contains `k` values, then every stored value must have probability:

```text
1 / k
```

For example, if the set contains:

```text
{10, 20, 30, 40}
```

then `getRandom()` must return each of `10`, `20`, `30`, and `40` with equal probability.

So the real problem is:

> Maintain a changing set of unique values while supporting membership checks, deletion, insertion, and uniform random selection in constant average time.

---

### 2. Start From Simple Data Structures

Before combining structures, ask what each operation naturally wants.

#### A Hash Set Handles Insert and Remove

A normal hash set gives:

```text
insert(val): O(1) average
remove(val): O(1) average
contains(val): O(1) average
```

That sounds close, but `getRandom()` is the problem.

A hash set has no direct constant-time way to choose the element at random index `i`. You could convert it to a list each time:

```python
random.choice(list(values))
```

But building that list costs `O(n)` per call.

So a plain set fails because random access is missing.

#### An Array Handles Random Selection

A dynamic array/list gives:

```text
getRandom(): choose random index from 0 to len(array) - 1
```

That is perfect for uniform random selection because indices are equally likely.

If:

```text
array = [10, 20, 30, 40]
```

then choosing a random integer from `0..3` gives every stored value probability `1/4`.

But an array alone has two problems:

1. To check whether `val` already exists, we may need to scan the array: `O(n)`.
2. To remove a value from the middle while preserving compact storage, shifting elements costs `O(n)`.

So a plain array fails because membership lookup and arbitrary deletion are too slow.

---

### 3. The Key Observation

The operations want two different abilities:

```text
Hash map: find a value immediately.
Array: choose a random position immediately.
```

The first-principles idea is to store the same set of values in two synchronized representations:

```text
values:       dynamic array of all current values
index_by_val: hash map from value -> its index in values
```

For example:

```text
values       = [10, 40, 30]
index_by_val = {
  10: 0,
  40: 1,
  30: 2
}
```

Now:

- `insert(20)` can check `20 in index_by_val` in `O(1)` average time.
- `getRandom()` can choose a random index in `values` in `O(1)` time.
- `remove(40)` can find where `40` lives by reading `index_by_val[40]`.

The remaining challenge is deletion.

---

### 4. Why Deletion Is Tricky

Suppose:

```text
values = [10, 40, 30, 20]
```

and we want to remove `40`, which is at index `1`.

If we simply delete index `1` from the array, the later elements shift left:

```text
[10, 30, 20]
```

That shifting costs `O(n)` in the worst case.

But the problem does not require the array to preserve order.

That is the crucial freedom.

A set has no meaningful order. The internal array can store values in any order as long as:

1. every stored value appears exactly once, and
2. the map records the correct current index for each value.

So instead of shifting all later elements, we can fill the removed slot with the last array element.

For removing `40` from:

```text
values = [10, 40, 30, 20]
```

use the last value `20` to fill index `1`:

```text
values = [10, 20, 30, 20]
```

Then pop the duplicate last slot:

```text
values = [10, 20, 30]
```

Finally, update the moved value's index:

```text
index_by_val[20] = 1
```

This turns arbitrary deletion into a constant-time overwrite plus a constant-time pop.

---

### 5. State and Invariant

The data structure maintains two pieces of state:

```text
values: list[int]
index_by_val: dict[int, int]
```

The invariant is:

```text
For every value x currently in the set,
index_by_val[x] is the exact index where x appears in values.

For every index i in values,
index_by_val[values[i]] == i.

The values list contains each current set value exactly once.
```

This invariant is the whole solution.

If it holds, then:

- membership is answered by the map,
- insertion appends to the array and records the new index,
- removal jumps directly to the target index,
- random selection chooses a uniform random index from the array.

Each operation is just a small update that preserves this invariant.

---

### 6. Detailed Algorithm

#### `insert(val)`

If `val` is already in `index_by_val`, insertion should fail because the set cannot contain duplicates.

Otherwise:

1. Append `val` to the end of `values`.
2. Record its index as `len(values) - 1`.
3. Return `True`.

```text
Before inserting 7:
values       = [4, 9]
index_by_val = {4: 0, 9: 1}

After appending 7:
values       = [4, 9, 7]
index_by_val = {4: 0, 9: 1, 7: 2}
```

#### `remove(val)`

If `val` is not in `index_by_val`, removal should fail.

Otherwise:

1. Find the index of `val`.
2. Read the last value in the array.
3. Move the last value into `val`'s index.
4. Update the moved value's index in the map.
5. Pop the last array slot.
6. Delete `val` from the map.
7. Return `True`.

The swap-with-last idea also works when `val` is already the last value. In that case, the value is moved onto itself, then popped. Some implementations special-case this, but it is not necessary if the map deletion happens after the move/update.

#### `getRandom()`

1. Choose a random integer `i` between `0` and `len(values) - 1`.
2. Return `values[i]`.

Because the array contains each set value exactly once, a uniform random index is the same as a uniform random value.

---

### 7. Pseudocode

```python
class RandomizedSet:
    def __init__(self):
        self.values = []
        self.index_by_val = {}

    def insert(self, val: int) -> bool:
        if val in self.index_by_val:
            return False

        self.index_by_val[val] = len(self.values)
        self.values.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.index_by_val:
            return False

        remove_index = self.index_by_val[val]
        last_value = self.values[-1]

        self.values[remove_index] = last_value
        self.index_by_val[last_value] = remove_index

        self.values.pop()
        del self.index_by_val[val]
        return True

    def getRandom(self) -> int:
        random_index = random.randint(0, len(self.values) - 1)
        return self.values[random_index]
```

In Python, `random.choice(self.values)` is also appropriate for `getRandom()` because it chooses uniformly from the list.

---

### 8. Walk Through the Official Example

Operations:

```text
["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[],              [1],      [2],      [2],      [],          [1],      [2],      []]
```

Start with an empty structure:

```text
values       = []
index_by_val = {}
```

#### Operation 1: `insert(1)`

`1` is not present.

Append it:

```text
values       = [1]
index_by_val = {1: 0}
```

Return:

```text
True
```

#### Operation 2: `remove(2)`

`2` is not present in the map.

No state changes:

```text
values       = [1]
index_by_val = {1: 0}
```

Return:

```text
False
```

#### Operation 3: `insert(2)`

`2` is not present.

Append it at index `1`:

```text
values       = [1, 2]
index_by_val = {1: 0, 2: 1}
```

Return:

```text
True
```

#### Operation 4: `getRandom()`

There are two values:

```text
values = [1, 2]
```

Choose index `0` or `1` uniformly.

So `getRandom()` may return either:

```text
1 or 2
```

Each has probability `1/2`.

#### Operation 5: `remove(1)`

`1` is present at index `0`.

The last value is `2`.

Move `2` into index `0`:

```text
values       = [2, 2]
index_by_val = {1: 0, 2: 0}
```

Pop the last slot and delete `1` from the map:

```text
values       = [2]
index_by_val = {2: 0}
```

Return:

```text
True
```

#### Operation 6: `insert(2)`

`2` is already present.

No state changes:

```text
values       = [2]
index_by_val = {2: 0}
```

Return:

```text
False
```

#### Operation 7: `getRandom()`

Only one value remains:

```text
values = [2]
```

So `getRandom()` must return:

```text
2
```

---

### 9. Why `getRandom()` Is Uniform

Uniformity comes from the array representation.

At any time, the invariant says:

```text
values contains every current set value exactly once
```

If there are `k` values, the valid indices are:

```text
0, 1, 2, ..., k - 1
```

A random index generator chooses each index with probability `1 / k`.

Since each value occupies exactly one index, each value is returned with probability `1 / k`.

The hash map is not used for randomness. It is only used to keep the array updateable in constant time.

---

### 10. Correctness Argument

We prove that the data structure implements the required behavior by showing that every operation preserves the invariant and returns the correct result.

#### Initialization

After construction:

```text
values = []
index_by_val = {}
```

There are no stored values, so the invariant holds vacuously.

#### Insert

If `val` is already in `index_by_val`, then by the invariant `val` is already in the set. Returning `False` is correct, and the state is unchanged, so the invariant remains true.

If `val` is not in `index_by_val`, then by the invariant `val` is not currently stored. Appending it to the end of `values` adds it exactly once. Recording its index as the old length of `values` makes the map point to its actual position. No existing value moves, so all previous map entries remain correct. Therefore the invariant holds after insertion, and returning `True` is correct.

#### Remove

If `val` is not in `index_by_val`, then by the invariant `val` is not in the set. Returning `False` is correct, and the state is unchanged.

If `val` is present, the map gives its exact index. The algorithm takes the last array value and writes it into the removed value's slot. Then it updates that last value's map entry to the new slot. Popping the final array position removes the duplicate copy of the moved value. Finally, deleting `val` from the map removes the deleted value from the set representation.

All remaining values still appear exactly once in `values`, and every map entry points to the correct current index. Therefore the invariant holds after removal, and returning `True` is correct.

#### GetRandom

By the invariant, `values` contains exactly the current set values, each once. `getRandom()` chooses one valid array index uniformly at random and returns the value at that index. Since each value occupies exactly one index, each current set value is returned with equal probability. Therefore `getRandom()` satisfies the problem requirement.

Because every operation is correct while preserving the invariant, the data structure is correct.

---

### 11. Complexity

Let `n` be the number of values currently stored.

#### `insert(val)`

- Hash map lookup: average `O(1)`
- Append to dynamic array: amortized `O(1)`
- Hash map write: average `O(1)`

Overall:

```text
Average time: O(1)
```

#### `remove(val)`

- Hash map lookup: average `O(1)`
- Constant number of array reads/writes: `O(1)`
- Array pop from the end: `O(1)`
- Hash map updates/deletion: average `O(1)`

Overall:

```text
Average time: O(1)
```

#### `getRandom()`

- Choose a random index: `O(1)`
- Read one array element: `O(1)`

Overall:

```text
Time: O(1)
```

#### Space

The structure stores each value once in the array and once in the map:

```text
Space: O(n)
```

---

### 12. Common Pitfalls

#### Pitfall 1: Using Only a Set

A set makes insert and remove easy, but random selection becomes expensive if you rebuild a list on every `getRandom()` call.

#### Pitfall 2: Removing From the Middle of a List

Calling an operation that deletes index `i` and shifts later elements breaks the `O(1)` deletion requirement.

The correct deletion is:

```text
swap/move with last, then pop
```

#### Pitfall 3: Forgetting to Update the Moved Value's Index

After moving the last value into the removed slot, its map entry must change.

Wrong state example:

```text
values       = [10, 20, 30]
index_by_val = {10: 0, 20: 1, 30: 2}

remove(20), move 30 to index 1
values       = [10, 30]
```

If the map still says:

```text
30 -> 2
```

then future operations will access the wrong index.

#### Pitfall 4: Deleting the Map Entry Too Early

During removal, you still need enough information to update the moved value correctly. A safe order is:

```text
find remove_index
find last_value
move last_value
update last_value index
pop
remove val from map
```

#### Pitfall 5: Mishandling Removal of the Last Element

When the removed value is already the last element, the same algorithm can still work:

```text
values[remove_index] = last_value
index_by_val[last_value] = remove_index
pop
remove val from map
```

Because `last_value == val`, the final map deletion removes the value completely.

#### Pitfall 6: Breaking Uniform Randomness

Do not choose randomly from map keys in a way that first converts keys to a list every time if the operation must be constant time. The persistent array is what makes uniform random access efficient.

---

### 13. First-Principles Summary

The problem is difficult only if we expect one data structure to do everything.

A hash map gives constant-time answers to:

```text
Where is this value?
Is this value present?
```

An array gives constant-time answers to:

```text
What value is at this random index?
```

The bridge between them is the invariant:

```text
index_by_val[value] always equals value's current index in values
```

Deletion becomes constant time because the set has no required order. When removing an interior value, we are free to replace it with the last array value and pop the end. That avoids shifting while keeping the array compact for random selection.

So the whole design is:

```text
array for random access
map for direct location
swap-with-last for constant-time deletion
```

## Implementation

See `solutions/array_string/p380_insert_delete_getrandom_o1.py`.

## Tests

See `tests/array_string/test_p380_insert_delete_getrandom_o1.py`.

## Examples

### Example 1
- Input: `{'raw': '["RandomizedSet","insert","remove","insert","getRandom","remove","insert","getRandom"]\n[[],[1],[2],[2],[],[1],[2],[]]'}`
- Output: `'See official examples'`

## Follow-up Practice
- Trace `values` and `index_by_val` after every operation in a sequence.
- Test removing the only element, removing the last element, and removing a middle element.
- Explain why choosing a random array index gives each stored value equal probability.
