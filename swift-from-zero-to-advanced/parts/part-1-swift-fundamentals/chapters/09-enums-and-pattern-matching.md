# Chapter 09: Enums and Pattern Matching

## Problem

Strings are useful at the boundary because the command line gives us text.
But strings are weak once the program already knows the command space.

If `list`, `add`, and `done` are the only commands Part 1 cares about, then
keeping them as unrelated string literals makes the code harder to trust:

- typos stay easy to write
- valid commands are not visible in one type
- routing logic keeps re-explaining the same knowledge

The question becomes:

How do we move from raw command text to typed commands that the rest of the
program can reason about safely?

## Running Example

~~~swift
enum Command: String {
    case list
    case add
    case done
}

let arguments = Array(CommandLine.arguments.dropFirst())
let command = arguments.first.flatMap(Command.init(rawValue:))

if let command {
    switch command {
    case .list:
        print("Listing tasks")
    case .add:
        print("Adding a task")
    case .done:
        print("Marking a task as done")
    }
} else {
    print("Unknown or missing command.")
}
~~~

## Semantic Deep Dive

An enum（枚举） lets Swift express "one value chosen from a known set of
cases."
For command routing, that is exactly the model we want.

`enum Command: String` says two things at once:

- `Command` is a real domain type
- each case has a raw string value that can be used for parsing

That creates a clean transition from unsafe input to safe internal state.
The CLI still receives strings, but it does not have to keep reasoning in
strings after parsing.

`Command(rawValue: someText)` returns an optional because parsing may fail.
That is the correct design.
Not every string is a valid command.

Once parsing succeeds, `switch` becomes stronger too.
This is where pattern matching（模式匹配） matters.
Pattern matching is Swift's way of comparing a value against structured cases
and selecting behavior from that structure.

With an enum, `switch` is not only comparing raw text.
It is matching over the cases of a known type.
That is safer, clearer, and easier to extend.

## Code Evolution

A weak version keeps routing in raw strings:

~~~swift
let arguments = Array(CommandLine.arguments.dropFirst())

if let command = arguments.first {
    switch command {
    case "list":
        print("Listing tasks")
    case "add":
        print("Adding a task")
    case "done":
        print("Marking a task as done")
    default:
        print("Unknown command: \(command)")
    }
}
~~~

This works, but it is weak because the command space is still scattered across
string literals.
Nothing stops a typo like `"lst"` from quietly becoming just another string
mistake.

A stronger version gives the command space a type:

~~~swift
enum Command: String {
    case list
    case add
    case done
}

let arguments = Array(CommandLine.arguments.dropFirst())

if let rawCommand = arguments.first,
   let command = Command(rawValue: rawCommand) {
    switch command {
    case .list:
        print("Listing tasks")
    case .add:
        print("Adding a task")
    case .done:
        print("Marking a task as done")
    }
} else {
    print("Usage: task-cli-lite <list|add|done>")
}
~~~

This stronger version improves the program in several ways:

- the known command space is visible in one declaration
- parsing is explicit through raw-value initialization
- successful parsing gives later code a typed `Command`
- `switch` now matches over domain cases instead of unrelated strings

This is a small example of a larger engineering idea:
once the program knows a concept is closed and named, a dedicated type is
usually stronger than repeating string conventions forever.

## Common Mistakes

- keeping strings everywhere because the input started as text
  Boundary format should not dictate the whole internal model.
- using an enum without actually parsing into it
  If the rest of the code still routes on raw strings, the type is decorative.
- forgetting that `Command(rawValue:)` returns an optional
  Unknown text is a real case and must be handled.
- writing a `default` case too early when matching a well-defined enum
  For closed command spaces, explicit cases often communicate intent better.

## Drills

- concept check: why is an enum a better internal model than raw strings for a
  known command space?
- code reading: trace what happens when the user runs `task-cli-lite remove`
- hands-on extension: add a `help` case to `enum Command` and update the
  parser and `switch` accordingly

## Checkpoint

You should now be able to explain why strings are weak for a known command
space, how raw-value parsing turns text into a typed command, and how `switch`
pattern matching makes the command router more explicit and reliable.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Enum | 枚举 | A type whose values come from a known set of named cases. |
| Raw value | 原始值 | A built-in value attached to an enum case, often used for parsing or serialization. |
| Pattern matching | 模式匹配 | Comparing a value against structured cases and selecting behavior from that structure. |
| Case | 枚举成员 | One named possibility inside an enum. |

## English Recap

- Raw CLI input starts as strings, but internal models do not need to stay as strings.
- `enum Command` gives the known command space one typed home.
- `Command(rawValue:)` is a safe parser because it returns an optional.
- `switch` over enum cases is stronger than routing on repeated string literals.

## Project Bridge

`TaskCLI Lite` now has the pieces it needs to stop pretending the CLI is just a
collection of text comparisons.
The final Part 1 step is integration: values, input/output, control flow,
functions, structs, arrays, optionals, and enums all need to work together in a
single small program that the reader can actually hold in their head.
