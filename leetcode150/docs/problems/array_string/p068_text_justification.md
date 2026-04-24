# 68. Text Justification

- Difficulty: Hard
- LeetCode: https://leetcode.com/problems/text-justification/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: array-string

## Core Pattern
TODO

## When To Use It
TODO

## Approach
TODO

## Correctness Sketch
TODO

## Complexity
TODO

## Common Pitfalls
TODO

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

## Follow-up Practice
TODO
