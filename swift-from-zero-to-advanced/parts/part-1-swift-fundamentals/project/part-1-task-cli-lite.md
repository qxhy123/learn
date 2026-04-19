# Part 1 Project: TaskCLI Lite

## Project Goal

Build a small command-line task manager that is simple enough to fit inside
Part 1 and real enough to force the language fundamentals to work together in
one coherent program.

The goal is not industrial software.
The goal is a trustworthy learning surface.

## Minimum Capabilities

The Part 1 version must support these commands:

- `list`
- `add <title>`
- `done <title>`

Those three capabilities are enough to require meaningful use of values, CLI
I/O, control flow, functions, structs, arrays, optionals, and enums without
dragging the reader into larger architecture too early.

## Why It Fits Part 1

`TaskCLI Lite` fits Part 1 because it is the smallest project shape that still
forces real integration.
It has:

- a visible input boundary
- a tiny but genuine domain model
- mutable task state
- command routing
- missing-input cases that require safe handling

That makes it a better Part 1 endpoint than either disconnected toy snippets or
an oversized project brief.

## Finish Line

The finish line is clear:
the reader can run the program, use `list`, `add <title>`, and `done <title>`,
read the code without getting lost, and explain why each Swift concept appears
where it does.

When that is true, Part 1 is complete and the tutorial can move into Part 2
without pretending the fundamentals still need to be guessed.
