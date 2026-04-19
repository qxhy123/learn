# Chapter 06: Optionals and Basic Error Handling

## What You Will Build

Safely parse missing or invalid command input without crashing the program.

## Core Concepts

- optionals
- `if let`
- `guard let`
- simple error reporting

## Code Walkthrough

Parse user input that may be missing, malformed, or empty and choose a safe
response for each case.

## Common Mistakes

- force-unwrapping input instead of handling absence explicitly
- merging missing-input and invalid-input cases into one unclear branch

## Drills

- concept check: explain the difference between `nil` and an invalid value
- code reading: identify where optional binding keeps a parser safe
- hands-on change: replace one unsafe unwrap with `if let` or `guard let`

## Checkpoint

Explain why optionals are central to Swift safety.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| optional | 可选值 | A value that can either contain data or `nil`. |
| optional binding | 可选值绑定 | A safe way to unwrap an optional value. |

## Further Reading

Connect this chapter to the shared glossary and notice how optional handling
supports clearer CLI error reporting.
