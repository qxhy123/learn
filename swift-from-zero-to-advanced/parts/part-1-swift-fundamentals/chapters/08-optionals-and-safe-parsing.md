# Chapter 08: Optionals and Safe Parsing

## Problem

Command-line input is hostile by default.
That does not mean users are malicious.
It means the program receives text that may be missing, incomplete, malformed,
or simply different from what the code hoped for.

Up to this point, `TaskCLI Lite` has been able to route commands and hold task
state, but it still faces a core engineering problem:

How should Swift represent "there may be a value here, or there may be no
usable value at all"?

If that question stays vague, the CLI turns brittle immediately.

## Running Example

~~~swift
let arguments = Array(CommandLine.arguments.dropFirst())

let commandText = arguments.first
let maybePriority = Int(arguments.dropFirst().first ?? "")

if let command = commandText {
    print("Command:", command)
} else {
    print("No command provided.")
}

if let priority = maybePriority {
    print("Parsed priority:", priority)
} else {
    print("Priority was missing or invalid.")
}
~~~

## Semantic Deep Dive

Swift uses `Optional` to model absence explicitly.
Optional（可选值） means "this value may exist, or it may be `nil`."

That sounds small, but it changes program design in an important way:

- missing data is not hidden
- failed parsing is not faked as a valid value
- the code has to acknowledge uncertainty before using the result

For a CLI, that is exactly what we want.
Arguments come in as strings, and many parsing operations are not guaranteed to
succeed.
`arguments.first` may be missing.
`Int("abc")` does not produce a number; it produces `nil`.

This is where Optional binding（可选值绑定） matters.
With `if let`, Swift lets you say:

"If the optional contains a real value, bind it to a stable local name and use
it inside this branch."

That is much stronger than hoping the value exists.
The branch itself documents the safety boundary.

Safe parsing is therefore not a separate topic from optionals.
It is the practical expression of optionals at a program boundary.

## Code Evolution

A weak version assumes too much:

~~~swift
let arguments = Array(CommandLine.arguments.dropFirst())
let command = arguments[0]
let priority = Int(arguments[1])!

print("Command:", command)
print("Priority:", priority)
~~~

This version is weak because:

- `arguments[0]` crashes if no command was supplied
- `arguments[1]` crashes if the second argument is missing
- `Int(arguments[1])!` crashes if the text is not a valid integer
- the code treats hostile input as if it were trusted structure

A stronger version makes absence and failure visible:

~~~swift
let arguments = Array(CommandLine.arguments.dropFirst())

if let command = arguments.first {
    print("Command:", command)
} else {
    print("Usage: task-cli-lite <command> [priority]")
}

if let rawPriority = arguments.dropFirst().first {
    if let priority = Int(rawPriority) {
        print("Priority:", priority)
    } else {
        print("Invalid priority: \(rawPriority)")
    }
} else {
    print("No priority supplied.")
}
~~~

This stronger version is still small, but it does real engineering work:

- it acknowledges that a command may be missing
- it distinguishes "missing" from "invalid"
- it uses `if let` to create clear safe-use boundaries
- it turns parsing failure into readable program behavior rather than a crash

For Part 1, that distinction is enough.
We do not need a full parser framework yet.
We need the habit of treating outside input as uncertain until the code proves
otherwise.

## Common Mistakes

- force-unwrapping with `!` because the example input "should" be valid
  That turns a teaching CLI into a crash demonstration.
- collapsing all failure modes into one blurry message
  Missing input and invalid input are related, but they are not identical.
- unwrapping too early and too far away from use
  The safest code keeps the optional boundary close to the operation that needs
  the value.
- treating `nil` as if it were an error message
  `nil` means absence, not explanation.
  The program still has to decide what message or fallback behavior to provide.

## Drills

- concept check: why is `Optional` a better model for missing CLI input than
  inventing a fake default string such as `"unknown"`?
- code reading: explain the difference between `arguments.first` returning
  `nil` and `Int(rawPriority)` returning `nil`
- hands-on extension: add support for an optional numeric limit argument and
  print separate messages for missing, invalid, and valid values

## Checkpoint

You should now be able to explain why command-line parsing is unsafe by
default, what `Optional` models in Swift, and how `if let` creates a readable
safe path for using values only after they have been proven to exist.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Optional | 可选值 | A value that may either contain a real value or `nil`. |
| Optional binding | 可选值绑定 | Safely extracting the wrapped value from an optional into a local name. |
| Parsing | 解析 | Turning raw input text into structured program values. |
| `nil` | 空值 | The absence of a value for an optional. |

## English Recap

- CLI input is hostile because it may be missing or malformed.
- `Optional` makes absence explicit in the type system.
- `if let` is the standard Part 1 tool for safe parsing boundaries.
- Stronger code distinguishes between missing input and invalid input.

## Project Bridge

`TaskCLI Lite` can now admit a critical truth: raw input is not yet trusted
program state.
The next step is making the command space itself stronger.
Instead of passing around loose strings forever, Part 1 now turns to enums so
the CLI can represent known commands as real Swift values.
