# 114. Flatten Binary Tree to Linked List

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/flatten-binary-tree-to-linked-list/
- Official Group: Binary Tree General
- Pattern Group: Binary Tree DFS
- Patterns: binary-tree-dfs, tree-traversal, linked-list

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given the root of a binary tree, modify the tree **in place** so that it becomes a linked list using the tree's existing nodes.

The required linked list order is the tree's **preorder traversal**:

```text
node, then left subtree, then right subtree
```

The linked list is represented using the tree pointers:

- every `left` pointer must become `None`
- every `right` pointer points to the next node in preorder
- no new nodes should be created

For example, consider:

```text
        1
       / \
      2   5
     / \   \
    3   4   6
```

Its preorder traversal is:

```text
1, 2, 3, 4, 5, 6
```

After flattening, the same nodes should be rewired as:

```text
1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6
```

Equivalently, every node's `left` pointer is empty, and following `right` pointers produces:

```text
1 -> 2 -> 3 -> 4 -> 5 -> 6
```

The important part is that the problem is not asking us to return a list of values. It is asking us to **mutate the pointer structure** of the original tree.

---

### 2. Start From the Baseline Idea

The simplest mental solution has two phases:

1. Traverse the tree in preorder and store the nodes in an array.
2. Walk through the array and connect each node to the next one using `right` pointers.

Conceptually:

```python
nodes = []

preorder(root):
    if root is None:
        return
    nodes.append(root)
    preorder(root.left)
    preorder(root.right)

for i in range(len(nodes)):
    nodes[i].left = None
    nodes[i].right = nodes[i + 1] if i + 1 < len(nodes) else None
```

This is easy to reason about because the array explicitly stores the desired final order.

But it uses `O(n)` extra space for `nodes`, where `n` is the number of tree nodes. The problem's in-place requirement encourages a deeper question:

> Can each subtree flatten itself and tell its parent where the flattened chain ends?

That question leads directly to the recursive pointer invariant.

---

### 3. The Key Observation

For one node `root`, the final preorder order of its subtree must be:

```text
root -> flattened left subtree -> flattened right subtree
```

So if the original subtree is:

```text
        root
       /    \
   left     right
```

then after flattening it should become:

```text
root -> left preorder chain -> right preorder chain
```

The current tree already gives us the two pieces we need:

- `root.left` is the start of the left subtree
- `root.right` is the start of the right subtree

The only difficulty is pointer order. If we immediately overwrite `root.right`, we may lose the original right subtree. So the local transformation must do three careful things:

1. Save the original left and right children.
2. Flatten both subtrees.
3. Splice the flattened left chain between `root` and the flattened right chain.

The splice looks like this:

```text
before:

root
├── left_chain
└── right_chain

after:

root -> left_chain -> right_chain
```

To attach `right_chain` after `left_chain`, we need to know the **tail** of the flattened left chain.

That is the reason the recursive function should not only flatten a subtree. It should also return the last node of the flattened subtree.

---

### 4. Recursive Contract and Pointer Invariant

Define a helper like this:

```text
flatten_subtree(node) returns the tail of the flattened chain rooted at node
```

After `flatten_subtree(node)` finishes, this invariant must be true:

```text
The subtree that used to be rooted at node is now a right-only chain
in preorder order, and the function returns the last node in that chain.
```

More explicitly, for every node inside that subtree:

- `left` is `None`
- `right` points to the next preorder node, or `None` at the tail
- no node is duplicated
- no original subtree is discarded

This return value is exactly the piece of information the parent needs. The parent does not need the entire flattened list. The parent already knows where the chain starts: it starts at the child pointer. It only needs the chain's tail so it can connect the next piece.

The invariant is local but powerful:

```text
If the left subtree can become a preorder chain,
and the right subtree can become a preorder chain,
then the current node can become a preorder chain by joining:

current node -> left chain -> right chain
```

---

### 5. Detailed Algorithm

For a node `node`:

1. If `node` is `None`, there is no chain, so return `None`.
2. Save references to the original children:
   - `left = node.left`
   - `right = node.right`
3. Recursively flatten the left subtree and get `left_tail`.
4. Recursively flatten the right subtree and get `right_tail`.
5. If there was a left subtree:
   - move the flattened left chain to `node.right`
   - set `node.left = None`
   - connect `left_tail.right` to the original right subtree chain
6. If there was no left subtree:
   - leave the flattened right chain after `node`
   - still ensure `node.left = None`
7. Return the tail of the whole flattened chain:
   - if `right_tail` exists, it is the tail
   - otherwise if `left_tail` exists, it is the tail
   - otherwise `node` itself is the tail

The return rule follows from preorder order:

```text
node -> left chain -> right chain
```

The final node is therefore the tail of the right chain if a right chain exists. If not, it is the tail of the left chain. If neither child exists, the current node is a one-node chain.

---

### 6. Pseudocode

```python
def flatten(root):
    def dfs(node):
        if node is None:
            return None

        original_left = node.left
        original_right = node.right

        left_tail = dfs(original_left)
        right_tail = dfs(original_right)

        if original_left is not None:
            node.right = original_left
            node.left = None
            left_tail.right = original_right
        else:
            node.left = None

        if right_tail is not None:
            return right_tail
        if left_tail is not None:
            return left_tail
        return node

    dfs(root)
```

This version is postorder in its mechanics: it first flattens children, then rewires the current node. But the resulting chain is preorder because the rewiring order is:

```text
current node, then left result, then right result
```

That distinction is a common source of confusion. The traversal used to perform the work does not have to be the same as the order of the final chain, as long as the pointer invariant creates the final preorder structure.

---

### 7. Walk Through the Main Example

Start with:

```text
        1
       / \
      2   5
     / \   \
    3   4   6
```

We want:

```text
1 -> 2 -> 3 -> 4 -> 5 -> 6
```

#### Flatten subtree rooted at `2`

The subtree is:

```text
    2
   / \
  3   4
```

Flatten `3`:

```text
3
```

Tail is `3`.

Flatten `4`:

```text
4
```

Tail is `4`.

Now splice at `2`:

```text
2 -> 3 -> 4
```

Pointers become:

```text
2.left = None
2.right = 3
3.right = 4
```

The flattened subtree rooted at `2` has tail `4`.

#### Flatten subtree rooted at `5`

The subtree is:

```text
5
 \
  6
```

There is no left subtree. The right subtree `6` flattens to itself.

The result remains:

```text
5 -> 6
```

The flattened subtree rooted at `5` has tail `6`.

#### Splice at root `1`

Before rewiring `1`, the two child chains are:

```text
left chain:  2 -> 3 -> 4
right chain: 5 -> 6
```

Preorder requires:

```text
1 -> left chain -> right chain
```

So we set:

```text
1.left = None
1.right = 2
4.right = 5
```

The final result is:

```text
1 -> 2 -> 3 -> 4 -> 5 -> 6
```

Written as a binary-tree level-style serialization with missing left children, that is:

```text
[1, None, 2, None, 3, None, 4, None, 5, None, 6]
```

---

### 8. Why Saving the Original Right Child Matters

Suppose we have:

```text
    1
   / \
  2   5
```

If we do this too early:

```python
node.right = node.left
```

then the original `node.right` might no longer be reachable unless it was saved somewhere first.

The correct local mindset is:

```python
original_left = node.left
original_right = node.right
```

Only after those references are saved is it safe to mutate `node.left` and `node.right`.

This is the main pointer-safety issue in the problem. The algorithm is simple once the two original child roots are protected.

---

### 9. Correctness Argument

We prove that `dfs(node)` flattens the subtree rooted at `node` into preorder order and returns the tail of that flattened chain.

#### Base case

If `node` is `None`, there is no subtree to flatten and no tail to return. Returning `None` is correct.

If `node` is a leaf, both recursive calls return `None`. The algorithm sets `node.left = None` and returns `node`. A one-node chain is exactly the preorder traversal of a leaf subtree, and its tail is the node itself.

#### Inductive step

Assume the recursive contract is correct for `node.left` and `node.right`.

After the recursive calls:

- the original left subtree, if it exists, is a right-only chain in preorder order
- `left_tail` is the last node of that left chain
- the original right subtree, if it exists, is a right-only chain in preorder order
- `right_tail` is the last node of that right chain

There are two cases.

If `node` has no original left subtree, the preorder order for this subtree is simply:

```text
node -> right preorder chain
```

The algorithm leaves that chain after `node`, clears `node.left`, and returns the correct tail: `right_tail` if it exists, otherwise `node`.

If `node` has an original left subtree, the preorder order must be:

```text
node -> left preorder chain -> right preorder chain
```

The algorithm sets `node.right` to the left chain, clears `node.left`, and connects `left_tail.right` to the original right chain. Because the left and right chains are already correct by the inductive hypothesis, this produces exactly the preorder chain for `node`'s whole subtree.

The returned tail is also correct: if the right chain exists, its tail is the final node; otherwise the left chain's tail is final.

Therefore the contract holds for `node`. By induction, it holds for the original root, so the entire tree is flattened correctly.

---

### 10. Complexity

Let `n` be the number of nodes and `h` be the height of the tree.

- Time: `O(n)` because each node is visited once, and each node performs only constant pointer work.
- Space: `O(h)` for the recursion stack.

The recursion stack is `O(log n)` for a balanced tree and `O(n)` for a completely skewed tree.

No extra array of nodes is needed in the recursive tail-return approach.

---

### 11. Common Pitfalls

#### Losing the original right subtree

If you overwrite `node.right` before saving it, the original right subtree may become disconnected.

Always preserve it before rewiring:

```python
original_right = node.right
```

#### Forgetting to clear `left`

The output is not merely a preorder path through `right` pointers. It must also have every `left` pointer set to `None`.

A tree that has the correct right chain but still has old left links is not fully flattened.

#### Returning the wrong tail

After joining:

```text
node -> left chain -> right chain
```

The tail is not always `left_tail`. If the right chain exists, `right_tail` is the final node.

The correct priority is:

```text
right_tail, then left_tail, then node
```

#### Confusing work order with output order

The recursive implementation may flatten children before rewiring the parent. That feels like postorder work, but the resulting structure is still preorder because of how the chains are connected.

The final chain order is determined by the splice, not by the chronological order of recursive calls.

#### Creating new nodes or returning a separate list

The problem asks for in-place mutation. Building a new linked list with new nodes does not satisfy the pointer requirement.

An auxiliary list of existing nodes is a useful baseline, but the final optimized approach should rewire the original nodes.

---

### 12. First-Principles Summary

The preorder flattening rule is:

```text
root subtree = root -> flattened left subtree -> flattened right subtree
```

So the fundamental operation is not searching, sorting, or balancing. It is **splicing two already-flattened child chains behind the current node**.

The smallest useful recursive return value is the tail of the flattened subtree. Once a child gives its parent the tail, the parent can attach the next chain without scanning.

The whole algorithm follows from one invariant:

```text
After dfs(node), node starts a right-only preorder chain for its original subtree,
and dfs(node) returns the last node of that chain.
```

With that invariant, each node performs the same local repair:

```text
save children -> flatten children -> move left chain to the right -> append old right chain -> return tail
```

That is the first-principles core of Flatten Binary Tree to Linked List.

## Implementation
See `solutions/binary_tree_dfs/p114_flatten_binary_tree_to_linked_list.py`.

## Tests
See `tests/binary_tree_dfs/test_p114_flatten_binary_tree_to_linked_list.py`.

## Examples

### Example 1
- Input: `{'root': [1, 2, 5, 3, 4, None, 6]}`
- Output: `[1, None, 2, None, 3, None, 4, None, 5, None, 6]`

### Example 2
- Input: `{'root': []}`
- Output: `[]`

### Example 3
- Input: `{'root': [0]}`
- Output: `[0]`
