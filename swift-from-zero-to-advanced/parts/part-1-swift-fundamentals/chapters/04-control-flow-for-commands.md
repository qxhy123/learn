# Chapter 04: Control Flow for Commands

## Problem

Once a CLI can read text, it needs a rule for deciding what to do with that
text.
That means control flow is no longer an abstract syntax lesson.
It is the mechanism that turns one input surface into multiple program
behaviors.

## Running Example

~~~swift
let arguments = Array(CommandLine.arguments.dropFirst())

if arguments.isEmpty {
    print("Usage: task-cli-lite <command>")
} else {
    let command = arguments[0]

    switch command {
    case "list":
        print("Listing tasks")
    case "add":
        print("Adding a task")
    default:
        print("Unknown command: \(command)")
    }
}
~~~

## Semantic Deep Dive

Swift gives you several control-flow tools, but they are not interchangeable in
meaning.

- `if` is good for binary or narrow conditional checks
- `switch` is stronger when the input space has named branches
- loops are for repeated work over data or repeated checks over state

In a command router, `switch` often communicates intent better than a stack of
`if` statements because the reader can see the command space in one place.

## Code Evolution

A weaker version looks like this:

~~~swift
if command == "list" {
    print("Listing tasks")
}

if command == "add" {
    print("Adding a task")
}
~~~

This version is weaker because it:

- spreads related cases apart
- makes it easier to forget the fallback path
- hides that all branches are answering the same question

Using `switch` makes the control surface visible.

## Common Mistakes

- writing several unrelated `if` blocks when the program is really choosing
  between known cases
- forgetting the unknown-command path
- mixing command routing and data mutation in the same unreadable block

## Drills

- concept check: when is `switch` clearer than `if` in a CLI?
- code reading: trace the behavior for `task-cli-lite list`
- hands-on extension: add a `help` command and decide whether it belongs in the
  same `switch`

## Checkpoint

You should now be able to explain why command routing is a control-flow design
problem and why `switch` is often the best fit for a small CLI command surface.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Branch | 分支 | One path selected by a condition or case. |
| Fallback | 兜底路径 | The path used when no intended command matches. |

## English Recap

- CLI control flow is about mapping one input space to multiple behaviors.
- `switch` often expresses command routing better than repeated `if` blocks.
- Good control flow makes the command space readable in one place.

## Project Bridge

`TaskCLI Lite` now has real command routing.
The next problem is structure: once those branches start growing, how do we keep
the program from collapsing into one giant file?
