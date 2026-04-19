# Part 1: Swift Fundamentals

## Why Part 1 Exists

Part 1 is where the reader stops treating Swift as "just another syntax" and
starts seeing what the language is trying to optimize for: explicitness,
safety, readable modeling, and small pieces that compose into a real program.

The whole part is organized around one question:

How do you grow from "I can print a line in Swift" to "I can build and extend a
small command-line tool without losing control of the code"?

## What Part 1 Does Not Try To Do

Part 1 does not teach everything in Swift.
It also does not front-load architecture, protocols, generics, testing, or
concurrency.

Those belong later.
Part 1 is about getting the language and the mental model right.

## What You Will Learn

By the end of Part 1, you should be able to:

- run Swift code as a script and as a compiled executable
- reason about values, mutability, and type information
- read input from the command line and format useful output
- shape a small program with control flow and functions
- model domain data with structs
- store and update task state with collections
- parse unsafe or missing input with optionals
- model commands with enums and pattern matching
- integrate all of that into `TaskCLI Lite v1`

## Chapter Sequence

1. Running Swift
2. Values and Types
3. Strings and Program I/O
4. Control Flow for Commands
5. Functions and Program Shape
6. Structs and Data Modeling
7. Collections and Task State
8. Optionals and Safe Parsing
9. Enums and Pattern Matching
10. Build TaskCLI Lite v1

## Project Spine

The Part 1 project is `TaskCLI Lite v1`.
At the start, it is barely a program.
By the end, it is still intentionally small, but it is real enough to carry the
semantics from the whole part.

## How To Study Part 1

- run the examples, even when they look trivial
- compare the chapter code before and after the code-evolution sections
- do the drills in order: concept, reading, extension
- do not skip the English recap if you want stronger technical vocabulary

## What Changes In Part 2

Part 2 does not replace Part 1.
It assumes Part 1 is already stable and then asks how the same program should be
structured, tested, and extended when the codebase grows.
