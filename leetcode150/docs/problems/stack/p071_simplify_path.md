# 71. Simplify Path

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/simplify-path/
- Official Group: Stack
- Pattern Group: Stack
- Patterns: stack

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

The input is an absolute Unix-style file path such as:

```text
/home/user/Documents/../Pictures
```

The task is to return the **canonical path**: the shortest normalized form that points to the same location.

A canonical absolute Unix path must obey these rules:

```text
It starts with exactly one leading slash.
Directory names are separated by exactly one slash.
It does not end with a slash unless the path is the root path "/".
It contains no "." components.
It contains no ".." components that can move upward.
```

The important detail is that the path is not just a string formatting problem. Some pieces of the string have meaning:

```text
"."   means stay in the current directory
".."  means go up to the parent directory
"abc" means enter the directory named abc
```

Repeated slashes do not create directory names:

```text
/home//foo/
```

has the same path components as:

```text
/home/foo
```

So the real problem is:

> Read the path from left to right and determine which directory names remain after applying every current-directory and parent-directory instruction.

---

### 2. Start From the Brute Force Idea

A first attempt might repeatedly rewrite the string until it looks clean:

1. Replace every repeated slash with one slash.
2. Remove every `/./`.
3. Find every `name/..` pair and remove it.
4. Repeat until no more changes are possible.
5. Fix the leading and trailing slashes.

For example:

```text
/home/user/Documents/../Pictures
```

could be rewritten as:

```text
/home/user/Pictures
```

This idea can work if implemented very carefully, but it is awkward and inefficient.

The difficult part is step 3. When we see `..`, the directory it cancels is not necessarily adjacent in the raw string after repeated slashes and `.` components are considered. Also, each rewrite changes the string, so the algorithm may scan the same characters many times.

The brute-force rewrite approach misses the structure of the problem:

> A path is a sequence of components, and `..` always affects the most recent real directory that has not already been canceled.

That sentence is exactly the reason a stack is natural here.

---

### 3. Split the Path Into Components

Slashes are separators. If we split the path on `/`, each token describes one path component:

```text
path = "/home/user/Documents/../Pictures"
```

splits into:

```text
"", "home", "user", "Documents", "..", "Pictures"
```

The empty string appears because the path starts with `/`. Empty strings can also appear when there are repeated slashes:

```text
"/home//foo/"
```

splits into:

```text
"", "home", "", "foo", ""
```

So after splitting, every token falls into one of four cases:

```text
""     ignore it; it came from extra slashes
"."    ignore it; it means stay here
".."   move up one directory if possible
name   move down into this directory
```

The problem is now no longer about manipulating characters. It is about simulating movement through directories.

---

### 4. The Key Observation

When a normal directory name appears, it becomes part of the current path:

```text
/home/user
```

If the next meaningful component is another directory name, we go deeper:

```text
/home/user/Documents
```

If the next meaningful component is `..`, we go back to the parent:

```text
/home/user/Documents/..
```

which returns to:

```text
/home/user
```

Which directory did `..` remove?

It removed the most recent directory that was still part of the path:

```text
Documents
```

That is last-in, first-out behavior:

```text
Last directory entered = first directory removed by ".."
```

A stack is the data structure that directly represents this rule.

---

### 5. Stack / Path Invariant

Maintain a stack of directory names.

The invariant is:

```text
After processing some prefix of the input path,
the stack contains exactly the canonical directory components
needed to describe the current location from the root.
```

For example, if the processed prefix leads to:

```text
/home/user/Pictures
```

then the stack is:

```text
["home", "user", "Pictures"]
```

This invariant is precise enough to decide every token locally:

- If the token is empty, it contributes no directory.
- If the token is `.`, the current location does not change.
- If the token is a normal directory name, append it to the path by pushing it.
- If the token is `..`, remove the last directory by popping, but only if the stack is not empty.

The empty-stack case matters because the input path is absolute. From the root directory:

```text
/
```

moving to the parent directory still leaves you at root:

```text
/..
```

canonicalizes to:

```text
/
```

So `..` with an empty stack does nothing.

---

### 6. Detailed Algorithm

1. Create an empty stack:

```text
stack = []
```

2. Split the input path by `/`.

3. Process each component in order.

4. If the component is empty or `.`:

```text
ignore it
```

It does not change the location.

5. If the component is `..`:

```text
if stack is not empty:
    pop the most recent directory
otherwise:
    stay at root
```

6. Otherwise, the component is a real directory name:

```text
push it onto the stack
```

This includes names like:

```text
"..."
"...."
"a.b"
```

Only the exact strings `.` and `..` are special.

7. Rebuild the canonical path:

```text
"/" + "/".join(stack)
```

If the stack is empty, `"/" + ""` is just:

```text
/
```

which is the correct canonical root path.

---

### 7. Pseudocode

```python
def simplifyPath(path: str) -> str:
    stack = []

    for part in path.split("/"):
        if part == "" or part == ".":
            continue

        if part == "..":
            if stack:
                stack.pop()
        else:
            stack.append(part)

    return "/" + "/".join(stack)
```

The stack stores only directory names, never slashes. Slashes are formatting added at the end.

---

### 8. Walkthrough: `/home/user/Documents/../Pictures`

Start with an empty stack:

```text
stack = []
```

Split the path:

```text
"", "home", "user", "Documents", "..", "Pictures"
```

Process each component:

#### Component: `""`

This came from the leading slash.

```text
ignore
stack = []
```

#### Component: `"home"`

This is a real directory name.

```text
push "home"
stack = ["home"]
```

#### Component: `"user"`

This is another real directory name.

```text
push "user"
stack = ["home", "user"]
```

#### Component: `"Documents"`

Move deeper into `Documents`.

```text
push "Documents"
stack = ["home", "user", "Documents"]
```

#### Component: `".."`

Move to the parent directory. The most recent real directory is `Documents`, so remove it.

```text
pop "Documents"
stack = ["home", "user"]
```

#### Component: `"Pictures"`

Move into `Pictures`.

```text
push "Pictures"
stack = ["home", "user", "Pictures"]
```

Rebuild the answer:

```text
"/" + "home/user/Pictures"
```

Result:

```text
/home/user/Pictures
```

---

### 9. Walkthrough: `/.../a/../b/c/../d/./`

This example is useful because it shows that `...` is a normal directory name.

Split into components:

```text
"", "...", "a", "..", "b", "c", "..", "d", ".", ""
```

Process them:

```text
start: stack = []

""    -> ignore
"..." -> push, stack = ["..."]
"a"   -> push, stack = ["...", "a"]
".."  -> pop,  stack = ["..."]
"b"   -> push, stack = ["...", "b"]
"c"   -> push, stack = ["...", "b", "c"]
".."  -> pop,  stack = ["...", "b"]
"d"   -> push, stack = ["...", "b", "d"]
"."   -> ignore
""    -> ignore
```

Join the stack:

```text
/.../b/d
```

The token `...` remains because the problem gives special meaning only to exactly `.` and exactly `..`.

---

### 10. Correctness

We prove that the algorithm returns the canonical path for the input absolute path.

#### Invariant

After processing the first `k` components of `path.split("/")`, the stack contains exactly the canonical directory names for the location reached by those `k` components, relative to the root.

#### Initialization

Before processing any components, no directory has been entered.

```text
stack = []
```

This represents the root directory `/`, so the invariant is true initially.

#### Maintenance

Consider the next component.

If the component is empty, it came from a leading slash, trailing slash, or repeated slash. It does not represent a directory movement, so ignoring it preserves the current canonical location.

If the component is `.`, it means stay in the current directory. Ignoring it preserves the current canonical location.

If the component is `..`, it means move to the parent directory. If the stack is non-empty, the current location has a most recent directory component, and popping that component moves exactly one level up. If the stack is empty, the current location is already root, and the parent of root is still treated as root for this problem. In both cases, the stack after the operation represents the correct canonical location.

If the component is any other string, it is a real directory name. Moving into that directory appends it to the end of the current path, so pushing it preserves the invariant.

Thus every possible component update preserves the invariant.

#### Termination

After all components have been processed, the invariant says the stack contains exactly the canonical directory names for the whole input path.

Joining those names with one slash and adding the leading slash creates an absolute path that:

```text
starts with one slash,
uses one slash between directory names,
has no trailing slash unless it is root,
contains no empty, ".", or resolvable ".." components.
```

Therefore the returned string is the required canonical path.

---

### 11. Complexity

Let `n` be the length of the input string.

Splitting and scanning the path touches each character a constant number of times:

```text
Time: O(n)
```

The stack can store at most all directory-name components from the path:

```text
Space: O(n)
```

The output string also takes `O(n)` space in the worst case.

---

### 12. Common Pitfalls

#### Treating `...` as special

Only these exact tokens are special:

```text
"."
".."
```

A token such as `"..."` is a normal directory name.

#### Popping from an empty stack

For paths like:

```text
/../
```

there is no directory to remove. The result should be:

```text
/
```

Always check whether the stack is non-empty before popping.

#### Keeping empty components

Repeated slashes produce empty split components:

```text
/home//foo
```

Those empty components must be ignored, otherwise the rebuilt path may contain extra slashes.

#### Adding slashes to the stack

The stack should store directory names only:

```text
["home", "foo"]
```

not:

```text
["/home", "/foo"]
```

Slashes are separators, not directory names.

#### Forgetting root formatting

If the stack is empty, the answer is `/`, not an empty string.

A simple rebuild expression handles both cases:

```python
return "/" + "/".join(stack)
```

---

### 13. First-Principles Summary

The path is a history of directory movements.

A normal directory name moves one level deeper, so we record it.

A `..` instruction cancels the most recent recorded directory, so we remove the last recorded name.

That is the entire problem:

```text
canonical path = surviving directory names after all cancellations
```

The stack is not an arbitrary pattern choice. It is the exact data structure implied by the parent-directory rule:

```text
The directory removed by ".." is always the last real directory that remains in the path.
```

Once that invariant is maintained, the final answer is only a formatting step: join the surviving names with single slashes and prepend the root slash.

## Implementation
See `solutions/stack/p071_simplify_path.py`.

## Tests
See `tests/stack/test_p071_simplify_path.py`.

## Examples

### Example 1
- Input: `{'raw': '"/home/"\n"/home//foo/"\n"/home/user/Documents/../Pictures"\n"/../"\n"/.../a/../b/c/../d/./"'}`
- Output: `'See official examples'`

## Follow-up Practice
- Trace the stack for a path with repeated slashes.
- Explain why `..` maps to `pop`.
- Test root-only paths such as `/`, `/./`, and `/../`.
- Test names that look special but are not, such as `...` and `a..`.
