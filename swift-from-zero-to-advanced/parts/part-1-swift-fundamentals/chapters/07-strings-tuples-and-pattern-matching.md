# Chapter 07: Strings, Tuples, and Pattern Matching

## What You Will Build

Human-readable task output and lightweight grouped values for the CLI.

## Core Concepts

- string interpolation
- tuples
- basic pattern matching

## Code Walkthrough

Format task output for humans, group related values in tuples, and match simple
cases without introducing a full custom type too early.

## Common Mistakes

- overusing tuples when named types would communicate intent better
- building unreadable output strings instead of composing them step by step

## Drills

- concept check: explain when string interpolation improves readability
- code reading: inspect a tuple-returning helper and its matched cases
- hands-on change: improve one output path for clarity and consistency

## Checkpoint

Explain when a tuple is enough and when a custom type is better.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| string interpolation | 字符串插值 | Embedding values directly inside a string literal. |
| tuple | 元组 | A lightweight grouped value with multiple elements. |

## Further Reading

Compare this chapter's output choices with the Part 1 project finish line and
keep the CLI readable without premature abstraction.
