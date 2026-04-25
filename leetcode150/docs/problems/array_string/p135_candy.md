# 135. Candy

- Difficulty: Hard
- LeetCode: https://leetcode.com/problems/candy/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: array-string

## Problem

We are given an array `ratings`, where `ratings[i]` is the rating of the child standing at position `i` in a line. We must give every child at least one candy.

The only extra rule is local:

- If child `i` has a higher rating than the child immediately to the left, then child `i` must receive more candies than that left neighbor.
- If child `i` has a higher rating than the child immediately to the right, then child `i` must receive more candies than that right neighbor.

Children with equal ratings do not impose any ordering requirement on candies. The goal is to minimize the total number of candies.

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

The input is a line of children. Each child has a rating, and we need to assign a positive integer number of candies to every child.

The rule is not based on absolute rating size. A child with rating `100` does not automatically deserve `100` candies, and a child with rating `2` does not automatically deserve fewer candies than every child with a larger rating somewhere else in the array.

The rule only compares neighbors:

```text
ratings[i] > ratings[i - 1]  means  candies[i] > candies[i - 1]
ratings[i] > ratings[i + 1]  means  candies[i] > candies[i + 1]
```

So the problem is really asking:

> What is the smallest integer array `candies` such that every child gets at least `1`, and every adjacent higher-rated child gets strictly more candies than the lower-rated neighbor?

This local nature is the key. A child only needs to beat the neighbors with lower ratings, not every lower-rated child in the whole line.

For example:

```text
ratings:  1  2  100  2  1
candies:  1  2   3   2  1
```

The `100` sits on top of a local mountain, so it needs more than both adjacent `2`s. It does not need one hundred candies.

### 2. Start With the Minimum Possible Assignment

The smallest amount any child can receive is one candy. If there were no rating constraints, the best answer would always be:

```text
candies: 1  1  1  ...  1
```

That gives a total of `n` candies.

Rating constraints force us to add candies only where a local comparison demands it. This means every useful solution should start from the same mental baseline:

```text
Every child gets 1 candy first.
Then increase only the children that must be larger than a neighbor.
```

The word "minimize" matters here. If a child needs to be larger than a neighbor with `3` candies, the cheapest valid amount is `4`, not `5` or `10`.

### 3. Why Looking at Each Pair Once Is Not Enough Naively

Consider this rating array:

```text
ratings: 1  2  3  4
```

Every child is higher-rated than the child on the left, so the candies must strictly increase:

```text
candies: 1  2  3  4
```

The last child does not need `2` candies merely because it is higher than the previous child in one adjacent pair. It needs `4` candies because there is a chain of increasing comparisons:

```text
4 > 3 > 2 > 1
```

Each step in the chain adds one more required candy. The requirement propagates from left to right.

Now consider the opposite direction:

```text
ratings: 4  3  2  1
candies: 4  3  2  1
```

Here the requirement propagates from right to left. The first child must exceed the second, the second must exceed the third, and so on.

This tells us a single left-to-right idea cannot handle all cases by itself. Increasing slopes and decreasing slopes push information in opposite directions.

### 4. The Brute-Force Fixing Idea

A direct way to solve the problem is to repeatedly repair violations.

Start with:

```text
candies = [1, 1, 1, ..., 1]
```

Then keep scanning the array:

- If `ratings[i] > ratings[i - 1]` but `candies[i] <= candies[i - 1]`, increase `candies[i]` to `candies[i - 1] + 1`.
- If `ratings[i] > ratings[i + 1]` but `candies[i] <= candies[i + 1]`, increase `candies[i]` to `candies[i + 1] + 1`.

Eventually this becomes valid because candies only increase and every increase moves a violation closer to being fixed.

But this is wasteful. In a long slope, information may move one position per scan. For example:

```text
ratings: 1  2  3  4  5  6
```

The final answer is obvious:

```text
candies: 1  2  3  4  5  6
```

A repeated repair process can spend many passes rediscovering that each child must be one more than the previous child. We want to propagate each directional constraint exactly once.

### 5. Split the Rules by Direction

There are only two kinds of neighbor requirements.

The first kind points left-to-right:

```text
ratings[i] > ratings[i - 1]
```

If the current child has a higher rating than the left child, then the current child must get one more candy than the left child, at minimum:

```text
candies[i] = candies[i - 1] + 1
```

This kind of rule can be enforced by scanning from left to right, because by the time we are at `i`, the best value for `i - 1` is already known for all increasing chains ending there.

The second kind points right-to-left:

```text
ratings[i] > ratings[i + 1]
```

If the current child has a higher rating than the right child, then the current child must get one more candy than the right child, at minimum:

```text
candies[i] = candies[i + 1] + 1
```

This kind of rule can be enforced by scanning from right to left, because by the time we are at `i`, the best value for `i + 1` is already known for all decreasing chains starting there.

The problem becomes much simpler after this split:

1. Enforce everything that depends on the left neighbor.
2. Enforce everything that depends on the right neighbor.
3. Keep the larger requirement when both sides matter.

### 6. The Left-to-Right Pass

Begin with all ones:

```text
ratings:  1  2  3  2
candies:  1  1  1  1
```

Scan from left to right.

At index `1`, rating `2` is greater than rating `1`, so child `1` needs more candies than child `0`:

```text
candies:  1  2  1  1
```

At index `2`, rating `3` is greater than rating `2`, so child `2` needs more candies than child `1`:

```text
candies:  1  2  3  1
```

At index `3`, rating `2` is not greater than rating `3`, so the left-to-right rule says nothing.

After this pass, every rising edge from left to right is valid:

```text
ratings[i] > ratings[i - 1]  =>  candies[i] > candies[i - 1]
```

But the assignment may still violate right-side rules. In the example above, rating `3` is greater than rating `2` on its right, and `3` candies is already more than `1`, so it happens to be fine. Other arrays need the second pass to repair descending slopes.

### 7. Why the Left Pass Alone Fails

Look at a purely decreasing array:

```text
ratings:  4  3  2  1
candies:  1  1  1  1   after left-to-right pass
```

There are no places where `ratings[i] > ratings[i - 1]`, so the left-to-right pass makes no changes.

But the assignment is invalid:

```text
rating 4 > rating 3, but candy 1 is not > candy 1
rating 3 > rating 2, but candy 1 is not > candy 1
rating 2 > rating 1, but candy 1 is not > candy 1
```

This is not a small bug in the pass. It is a direction problem. The information for a decreasing slope lives on the right side, so it must be propagated from right to left.

### 8. The Right-to-Left Pass

Now scan from right to left. Whenever the current rating is greater than the rating to the right, the current child must receive at least one more candy than that right neighbor.

For the decreasing example:

```text
ratings:  4  3  2  1
candies:  1  1  1  1
```

Move from right to left:

```text
index 2: rating 2 > rating 1, so candies[2] = 2
candies: 1  1  2  1

index 1: rating 3 > rating 2, so candies[1] = 3
candies: 1  3  2  1

index 0: rating 4 > rating 3, so candies[0] = 4
candies: 4  3  2  1
```

Now every descending slope is valid.

### 9. Why the Second Pass Uses `max`

A child can be constrained by both neighbors.

Consider:

```text
ratings: 1  2  3  2  1
```

The peak rating `3` must be larger than the `2` on the left and larger than the `2` on the right.

After the left-to-right pass:

```text
ratings:  1  2  3  2  1
candies:  1  2  3  1  1
```

The increasing chain on the left has already forced the peak to have `3` candies.

During the right-to-left pass, the descending chain on the right says:

```text
ratings 3 > 2 > 1
```

So the peak must be at least one more than the right neighbor. That also requires `3` candies.

In this case both sides agree. But sometimes one side requires more than the other:

```text
ratings: 1  2  3  4  2  1
```

After the left-to-right pass:

```text
candies: 1  2  3  4  1  1
```

The peak already needs `4` candies because of the long rising slope from the left. The right side only requires the peak to be greater than the child with rating `2`, which will become `2` candies. That right-side requirement says the peak needs at least `3`, but it already has `4`.

If the second pass overwrote `4` with `3`, it would break the left-side chain:

```text
ratings: 1  2  3  4
candies: 1  2  3  3   invalid at the last step
```

Therefore the right-to-left pass must preserve the larger requirement:

```text
candies[i] = max(candies[i], candies[i + 1] + 1)
```

This is the central safety rule of the algorithm. The second pass is allowed to increase a candy count, but never decrease one.

### 10. Equal Ratings Reset the Constraint

Equal neighboring ratings do not require different candy counts.

For example:

```text
ratings: 1  2  2
```

The child with rating `2` at index `1` must get more than the child with rating `1`, so it gets `2` candies. The child with rating `2` at index `2` has the same rating as index `1`, so there is no rule requiring index `2` to match or exceed index `1`.

A minimum valid assignment is:

```text
candies: 1  2  1
```

This is why the algorithm uses strict `>` comparisons, not `>=` comparisons. Equal ratings break slopes. They do not continue them.

Another example:

```text
ratings: 2  2  2
candies: 1  1  1
```

All children can receive one candy because no child is higher-rated than an adjacent child.

### 11. The Invariant After Each Pass

The candy array has a precise meaning throughout the algorithm:

```text
candies[i] = the smallest amount currently known to be necessary for child i
```

At initialization:

```text
candies[i] = 1
```

This satisfies the universal rule that every child gets at least one candy.

After the left-to-right pass, this invariant holds:

```text
For every i > 0:
if ratings[i] > ratings[i - 1], then candies[i] > candies[i - 1].
```

So every increasing edge from left to right is valid.

After the right-to-left pass, this invariant also holds:

```text
For every i < n - 1:
if ratings[i] > ratings[i + 1], then candies[i] > candies[i + 1].
```

So every increasing edge from right to left is valid.

The second pass uses `max`, so it cannot invalidate the first invariant. It may increase `candies[i]`, but increasing a child that was already greater than its left neighbor does not make it smaller than that left neighbor.

At the end, both directional conditions hold. Those two conditions are exactly the full problem statement.

### 12. Why This Produces the Minimum Total

The algorithm is not only valid; it is minimal.

For any child `i`, there are two possible sources of required candies:

1. A rising chain ending at `i` from the left.
2. A rising chain ending at `i` from the right.

The left-to-right pass computes the minimum candies needed because of the left chain. If the child is part of a rising slope, it gets one more than the previous child. If not, the left side imposes no extra requirement and it stays at one.

The right-to-left pass computes the minimum candies needed because of the right chain. If the child is higher than the right neighbor, it must be one more than the right neighbor.

A child that is constrained by both sides must satisfy both lower bounds. The cheapest way to satisfy two lower bounds is to take the maximum of them.

So each final value is:

```text
candies[i] = max(required_by_left_side, required_by_right_side)
```

Any smaller value would violate at least one adjacent chain. Any larger value would be unnecessary. Therefore summing these values gives the minimum possible total.

### 13. Walk Through Example 1

Input:

```text
ratings = [1, 0, 2]
```

Start with one candy each:

```text
ratings:  1  0  2
candies:  1  1  1
```

Left-to-right pass:

- `0` is not greater than `1`, so index `1` stays `1`.
- `2` is greater than `0`, so index `2` becomes `2`.

```text
candies:  1  1  2
```

Right-to-left pass:

- `0` is not greater than `2`, so index `1` stays `1`.
- `1` is greater than `0`, so index `0` must become `2`.

```text
candies:  2  1  2
```

Total:

```text
2 + 1 + 2 = 5
```

### 14. Walk Through Example 2

Input:

```text
ratings = [1, 2, 2]
```

Start:

```text
ratings:  1  2  2
candies:  1  1  1
```

Left-to-right pass:

- `2 > 1`, so index `1` becomes `2`.
- The last `2` is not greater than the previous `2`, so index `2` stays `1`.

```text
candies:  1  2  1
```

Right-to-left pass:

- The middle `2` is not greater than the last `2`, so no change.
- `1` is not greater than `2`, so no change.

Final:

```text
candies:  1  2  1
```

Total:

```text
1 + 2 + 1 = 4
```

The equal ratings at the end are the important detail. Equal ratings do not require equal candies.

### 15. Algorithm

1. Create a `candies` array of length `n`, filled with `1`.
2. Scan `ratings` from left to right.
3. If `ratings[i] > ratings[i - 1]`, set `candies[i] = candies[i - 1] + 1`.
4. Scan `ratings` from right to left.
5. If `ratings[i] > ratings[i + 1]`, set `candies[i] = max(candies[i], candies[i + 1] + 1)`.
6. Return the sum of `candies`.

Pseudocode:

```text
function candy(ratings):
    candies = array of length ratings.length filled with 1

    for i from 1 to ratings.length - 1:
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1

    for i from ratings.length - 2 down to 0:
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)

    return sum(candies)
```

### 16. Complexity

Let `n` be the number of children.

The algorithm performs two linear scans and one sum:

```text
Time: O(n)
```

It stores one candy count per child:

```text
Space: O(n)
```

There are more advanced constant-space formulations that count rising and falling slopes directly, but the two-pass array version is usually the clearest first-principles solution. It exposes exactly why the constraints split into left and right requirements, and why taking the maximum preserves both.

### 17. First-Principles Summary

The candy count for a child is determined by local rating inequalities, not by the child's absolute rating.

Start with one candy for everyone. A rising slope from left to right forces candies to rise from left to right. A rising slope from right to left forces candies to rise from right to left. Since a child can be constrained by both sides, the final candy count must keep the larger of the two requirements.

The two-pass algorithm works because each pass enforces one complete direction of the adjacent rules. The `max` in the second pass is what lets the algorithm combine both directions without breaking work already done.

## Implementation

- [Python solution](../../../solutions/array_string/p135_candy.py)

## Tests

- [Pytest tests](../../../tests/array_string/test_p135_candy.py)

## Examples

### Example 1

```text
Input: ratings = [1,0,2]
Output: 5
Explanation: Give candies [2,1,2].
```

The middle child has the lowest rating, so one candy is enough there. Both neighbors have higher ratings than the middle child, so each needs more than one candy.

### Example 2

```text
Input: ratings = [1,2,2]
Output: 4
Explanation: Give candies [1,2,1].
```

The second child has a higher rating than the first child, so it must get more candy. The last child has the same rating as the second child, so there is no requirement between them.
