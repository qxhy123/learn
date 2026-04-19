# Chapter 03: Control Flow

## What You Will Build

Branching and looping logic for task-list command handling.

## Core Concepts

- `if`
- `switch`
- `for`
- `while`

## Code Walkthrough

Follow a tiny command parser that branches on subcommands and loops through a
small list of tasks.

## Common Mistakes

- reaching for long `if` chains when `switch` expresses intent better
- mutating loop state in ways that make command handling hard to trace

## Drills

- concept check: choose between `if`, `switch`, `for`, and `while`
- code reading: trace loop execution by hand for a sample command list
- hands-on change: rewrite one branch-heavy parser as a `switch`

## Checkpoint

Choose the right control-flow construct for a command parser.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| branch | 分支 | A path selected based on a condition or pattern. |
| loop | 循环 | Repeated execution while iterating or checking a condition. |

## Further Reading

Link this chapter back to the Part 1 project and notice how command routing
depends on readable control flow.
