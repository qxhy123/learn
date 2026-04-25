# 68. Text Justification

- Difficulty: Hard
- LeetCode: https://leetcode.com/problems/text-justification/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: greedy, string-formatting

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given:

```text
words:    a list of words, already in the order they must appear
maxWidth: the exact width of every output line
```

You must return a list of strings where:

```text
1. Every returned string has length exactly maxWidth.
2. Words appear in the original order.
3. Each line contains as many words as possible.
4. Every non-last line is fully justified.
5. The last line is left-justified.
```

The important constraint is not only "fit words into lines." It is:

> For each line, choose the longest prefix of the remaining words that fits, then decide how many spaces must be inserted between those words so the line has exactly `maxWidth` characters.

For example:

```text
words = ["This", "is", "an", "example", "of", "text", "justification."]
maxWidth = 16
```

The first line can contain:

```text
"This" + "is" + "an"
```

because the minimum required width is:

```text
len("This") + 1 + len("is") + 1 + len("an")
= 4 + 1 + 2 + 1 + 2
= 10
```

Adding `"example"` would require:

```text
4 + 1 + 2 + 1 + 2 + 1 + 7 = 18
```

which is too wide.

So the first line must use exactly these three words. The remaining question is how to expand the spaces so the line length becomes exactly `16`.

### 2. Start From the Brute Force Idea

A baseline way to think about the problem is:

```text
For every possible way to split words into lines:
    Reject the split if any line is wider than maxWidth.
    Reject the split if any line does not use as many words as possible.
    Format the remaining split.
```

This is not a useful implementation, but it clarifies the two decisions involved:

```text
Decision 1: Which words belong on the current line?
Decision 2: Where do the spaces go inside that line?
```

If we tried all line breaks, the number of possibilities would grow quickly because each gap between adjacent words might or might not become a line break.

But the problem removes the need to search:

```text
Each line must contain as many words as possible.
```

That phrase makes the line break deterministic. Once we know the first unprocessed word, there is exactly one correct set of words for the next line: keep adding words while the minimum possible line width does not exceed `maxWidth`.

So the brute-force search collapses into a greedy scan.

### 3. The Key Observation

When deciding whether a set of words can fit on one line, we should count the minimum number of spaces first.

If a candidate line contains words:

```text
words[i], words[i + 1], ..., words[j]
```

then the minimum possible line uses one space between each adjacent pair:

```text
minimum_width = sum(word lengths) + number_of_gaps
```

where:

```text
number_of_gaps = j - i
```

This minimum width is the only thing needed while choosing the line's words.

Why?

Because any fully justified version of the line can only add more spaces between the same words. If the minimum version is already too wide, no spacing arrangement can save it. If the minimum version fits, then the line can always be padded to exactly `maxWidth`.

This gives the greedy rule:

> Starting at the next unused word, add the next word if and only if the current words plus one space between each adjacent pair still fit.

Once the next word does not fit, the current line is final.

### 4. Why the Greedy Line Choice Is Safe

Suppose the next unused word is at index `i`.

The problem explicitly says the current line should contain as many words as possible. Therefore, if words `i...j` fit but words `i...j+1` do not fit, any valid answer must end this line at `j`.

There is no advantage to putting fewer words on the line:

```text
It would violate the problem's rule.
```

There is also no way to put more words on the line:

```text
Even the minimum spacing would exceed maxWidth.
```

So the greedy line break is not merely an optimization. It is forced by the statement.

### 5. State and Invariant

We process words from left to right.

The main state is:

```text
i: index of the first word not yet placed into the output
result: all completed lines before i
```

While building the next line, we also track:

```text
j: one past the last word currently chosen for this line
line_words_len: total length of the words chosen for this line, excluding spaces
```

The invariant before formatting each line is:

```text
All words before index i have already been placed into valid output lines.
No word at or after index i has been placed yet.
The next output line must start with words[i].
```

The invariant while extending the line is:

```text
words[i:j] fit on one line using one space between adjacent words.
line_words_len equals the sum of lengths of words[i:j].
```

The extension stops when either:

```text
j == len(words)
```

or:

```text
words[i:j + 1] would not fit with one space between adjacent words.
```

At that point, `words[i:j]` is exactly the maximal set of words for the next line.

### 6. Formatting a Chosen Line

After choosing the words for one line, there are two cases.

#### Case A: Last line or single-word line

The line should be left-justified.

That means:

```text
Join the words using one space.
Append spaces on the right until length is maxWidth.
```

This applies to:

```text
1. The final line of the paragraph.
2. Any line that has only one word.
```

The single-word case is special because there are no gaps where spaces can be distributed. The only possible place for extra spaces is the right side.

#### Case B: Normal fully justified line

Suppose the line contains:

```text
k words
```

Then it has:

```text
gaps = k - 1
```

The total number of spaces needed is:

```text
spaces = maxWidth - line_words_len
```

These spaces must be distributed across the gaps.

The problem says:

```text
Spaces should be distributed as evenly as possible.
If the spaces do not divide evenly, the left gaps get the extra spaces.
```

So:

```text
base_spaces = spaces // gaps
extra_spaces = spaces % gaps
```

For each gap from left to right:

```text
gap_width = base_spaces
if this is one of the first extra_spaces gaps:
    gap_width += 1
```

This guarantees:

```text
1. Every gap has either base_spaces or base_spaces + 1 spaces.
2. Earlier gaps get the larger width.
3. The total number of inserted spaces is exactly spaces.
```

### 7. Detailed Algorithm

Use an index `i` to mark the first unprocessed word.

For each line:

1. Start with `j = i` and `line_words_len = 0`.
2. Try to include words while the minimum width still fits.
3. After the loop, `words[i:j]` is the maximal group for the next line.
4. If `j == len(words)`, this is the last line, so left-justify it.
5. Else if `j - i == 1`, this line has one word, so left-justify it.
6. Otherwise, distribute all required spaces across the internal gaps.
7. Append the formatted line to the answer.
8. Set `i = j` and repeat.

The fit test is the most common place for off-by-one mistakes.

If we are considering adding `words[j]` to the line `words[i:j]`, the new candidate would contain:

```text
j - i + 1 words
```

and therefore:

```text
j - i gaps
```

So the minimum width would be:

```text
line_words_len + len(words[j]) + (j - i)
```

The term `(j - i)` is the number of single spaces required between the candidate words.

### 8. Pseudocode

```python
def fullJustify(words, maxWidth):
    result = []
    i = 0

    while i < len(words):
        j = i
        line_words_len = 0

        # Choose the maximal group of words for this line.
        while j < len(words):
            candidate_words_len = line_words_len + len(words[j])
            candidate_gaps = j - i
            if candidate_words_len + candidate_gaps > maxWidth:
                break

            line_words_len = candidate_words_len
            j += 1

        line_words = words[i:j]
        word_count = j - i

        if j == len(words) or word_count == 1:
            line = " ".join(line_words)
            line += " " * (maxWidth - len(line))
        else:
            gaps = word_count - 1
            total_spaces = maxWidth - line_words_len
            base_spaces = total_spaces // gaps
            extra_spaces = total_spaces % gaps

            pieces = []
            for gap_index in range(gaps):
                pieces.append(line_words[gap_index])

                width = base_spaces
                if gap_index < extra_spaces:
                    width += 1

                pieces.append(" " * width)

            pieces.append(line_words[-1])
            line = "".join(pieces)

        result.append(line)
        i = j

    return result
```

### 9. Detailed Example Walkthrough

Use:

```text
words = ["This", "is", "an", "example", "of", "text", "justification."]
maxWidth = 16
```

#### Line 1

Start at:

```text
i = 0
```

Try words from left to right:

```text
"This"
minimum width = 4
fits
```

```text
"This is"
minimum width = 4 + 2 + 1 = 7
fits
```

```text
"This is an"
minimum width = 4 + 2 + 2 + 2 gaps = 10
fits
```

```text
"This is an example"
minimum width = 4 + 2 + 2 + 7 + 3 gaps = 18
does not fit
```

So the first line uses:

```text
["This", "is", "an"]
```

The total word length is:

```text
4 + 2 + 2 = 8
```

The total spaces needed are:

```text
16 - 8 = 8
```

There are:

```text
2 gaps
```

So:

```text
base_spaces = 8 // 2 = 4
extra_spaces = 8 % 2 = 0
```

Both gaps get four spaces:

```text
"This    is    an"
```

#### Line 2

Now:

```text
i = 3
```

The remaining words start at `"example"`.

Try:

```text
"example"
minimum width = 7
fits
```

```text
"example of"
minimum width = 7 + 2 + 1 = 10
fits
```

```text
"example of text"
minimum width = 7 + 2 + 4 + 2 gaps = 15
fits
```

```text
"example of text justification."
minimum width = 7 + 2 + 4 + 14 + 3 gaps = 30
does not fit
```

So the second line uses:

```text
["example", "of", "text"]
```

The total word length is:

```text
7 + 2 + 4 = 13
```

The total spaces needed are:

```text
16 - 13 = 3
```

There are:

```text
2 gaps
```

So:

```text
base_spaces = 3 // 2 = 1
extra_spaces = 3 % 2 = 1
```

The first gap gets two spaces, and the second gap gets one space:

```text
"example  of text"
```

#### Line 3

Now the only remaining word is:

```text
["justification."]
```

This is the last line, so it is left-justified:

```text
"justification.  "
```

The final answer is:

```text
[
  "This    is    an",
  "example  of text",
  "justification.  "
]
```

### 10. Correctness

We prove that the algorithm returns exactly the required text justification.

#### Lemma 1: Each chosen line contains the maximum possible number of words.

For a line starting at index `i`, the algorithm keeps adding the next word while the candidate line fits with one space between adjacent words.

If the loop stops because the next word would exceed `maxWidth`, then even the minimum spacing version of that larger line is too wide. Since every valid line needs at least one space between adjacent words, no valid formatting can include that next word.

If the loop stops because all words have been consumed, then the line already contains all remaining words.

Therefore, each chosen line contains as many words as possible.

#### Lemma 2: Every output line has length exactly `maxWidth`.

For a left-justified line, the algorithm first joins words with one space, producing a string whose length is at most `maxWidth` because the line was chosen by the fit test. It then appends exactly `maxWidth - len(line)` spaces. The final length is exactly `maxWidth`.

For a fully justified line, the algorithm computes:

```text
total_spaces = maxWidth - line_words_len
```

It inserts exactly `total_spaces` spaces across the gaps and includes each word exactly once. Therefore the final line length is:

```text
line_words_len + total_spaces = maxWidth
```

#### Lemma 3: Every non-last multi-word line has spaces distributed correctly.

For a normal line with `gaps` gaps, the algorithm computes:

```text
base_spaces = total_spaces // gaps
extra_spaces = total_spaces % gaps
```

It gives `base_spaces + 1` spaces to the first `extra_spaces` gaps and `base_spaces` spaces to the remaining gaps.

By division with remainder:

```text
total_spaces = base_spaces * gaps + extra_spaces
```

So all spaces are used. Also, no two gap widths differ by more than one, and any larger gaps appear before smaller gaps. This is exactly the required distribution.

#### Lemma 4: Words appear in the original order and none are skipped or duplicated.

Each iteration formats exactly `words[i:j]`, appends one output line, and then sets:

```text
i = j
```

The next iteration begins at the first word not yet used. Since `i` only moves forward to `j`, the algorithm processes disjoint consecutive ranges whose union is the entire input list.

Therefore, every word appears exactly once and in the original order.

#### Theorem: The algorithm returns the required result.

By Lemma 1, each line contains the required maximum number of words. By Lemma 2, each returned string has length exactly `maxWidth`. By Lemma 3, every non-last multi-word line is fully justified with left-biased extra spaces. By the left-justification branch, the last line and single-word lines are formatted correctly. By Lemma 4, the word order is preserved and all words are used exactly once.

Therefore, the returned list is exactly the text justification required by the problem.

### 11. Complexity

Let:

```text
n = number of words
L = total number of characters in all output lines
```

Each word is considered once when choosing line breaks.

Each output line is built once, and the total number of characters written across all output strings is `L`.

So:

```text
Time:  O(n + L)
Space: O(L) for the returned answer
```

If output space is excluded, the extra working space is:

```text
O(maxWidth)
```

for constructing one line at a time.

In LeetCode discussions this is often summarized as:

```text
Time:  O(total output size)
Space: O(total output size)
```

because returning the fully formatted text dominates the storage.

### 12. Common Pitfalls

#### Pitfall 1: Counting spaces incorrectly while choosing a line

When considering `words[i:j + 1]`, the number of minimum gaps is:

```text
j - i
```

not:

```text
j - i + 1
```

There is no space before the first word or after the last word in the minimum-width test.

#### Pitfall 2: Forgetting that the last line is different

The last line is not fully justified. It is left-justified:

```text
words separated by one space, then trailing spaces
```

For example, the last line:

```text
["shall", "be"]
```

with `maxWidth = 16` becomes:

```text
"shall be        "
```

not a line with a huge internal gap.

#### Pitfall 3: Dividing by zero for a one-word line

A one-word line has:

```text
gaps = 0
```

So the normal space distribution formula cannot be used. Left-justify it instead.

#### Pitfall 4: Treating all lines with one word as last lines only

A single-word line can occur in the middle of the output when a word is long enough that no other word fits beside it.

For example:

```text
["acknowledgment"]
```

with `maxWidth = 16` becomes:

```text
"acknowledgment  "
```

even though more words remain.

#### Pitfall 5: Putting extra spaces on the rightmost gaps

When spaces do not divide evenly, the earlier gaps must be wider.

For:

```text
["example", "of", "text"]
maxWidth = 16
```

the total word length is `13`, so there are `3` spaces across `2` gaps.

Correct:

```text
"example  of text"
```

Incorrect:

```text
"example of  text"
```

#### Pitfall 6: Building lines with repeated string concatenation in a tight loop

Repeatedly doing:

```python
line += next_piece
```

can create many intermediate strings in Python. Building a `pieces` list and using `"".join(pieces)` is usually cleaner and avoids unnecessary copying.

### 13. First-Principles Summary

This problem follows from a small set of basic facts:

```text
1. Word order is fixed.
2. Every line must start at the first unused word.
3. "As many words as possible" makes the next line break forced.
4. A candidate line can fit if its word lengths plus one space per internal gap fit.
5. After the words are chosen, the only remaining freedom is where to put spaces.
6. Last lines and one-word lines are left-justified.
7. Other lines distribute spaces evenly, with leftover spaces assigned left to right.
```

In one sentence:

> Greedily choose the longest prefix of remaining words that fits, then either left-pad the end for last/single-word lines or divide the required spaces evenly across internal gaps with the remainder going to the left.

## Implementation

See `solutions/array_string/p068_text_justification.py`.

## Tests

See `tests/array_string/test_p068_text_justification.py`.

## Examples

### Example 1
- Input: `{'words': ['This', 'is', 'an', 'example', 'of', 'text', 'justification.'], 'maxWidth': 16}`
- Output: `'[\n\xa0 \xa0"This \xa0 \xa0is \xa0 \xa0an",\n\xa0 \xa0"example \xa0of text",\n\xa0 \xa0"justification. \xa0"\n]'`

### Example 2
- Input: `{'words': ['What', 'must', 'be', 'acknowledgment', 'shall', 'be'], 'maxWidth': 16}`
- Output: `'[\n\xa0 "What \xa0 must \xa0 be",\n\xa0 "acknowledgment \xa0",\n\xa0 "shall be \xa0 \xa0 \xa0 \xa0"\n]'`

### Example 3
- Input: `{'words': ['Science', 'is', 'what', 'we', 'understand', 'well', 'enough', 'to', 'explain', 'to', 'a', 'computer.', 'Art', 'is', 'everything', 'else', 'we', 'do'], 'maxWidth': 20}`
- Output: `'[\n\xa0 "Science \xa0is \xa0what we",\n  "understand \xa0 \xa0 \xa0well",\n\xa0 "enough to explain to",\n\xa0 "a \xa0computer. \xa0Art is",\n\xa0 "everything \xa0else \xa0we",\n\xa0 "do \xa0 \xa0 \xa0 \xa0 \xa0 \xa0 \xa0 \xa0 \xa0"\n]'`
