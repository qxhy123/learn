# Chapter 03: Strings and Program I/O

## Problem

A command-line program lives at a boundary:

- the user gives it text
- the program turns that text into behavior
- the program sends text back

So even before we talk about complex parsing, we need to understand strings and
program I/O as a design surface, not just as syntax.

## Running Example

~~~swift
let arguments = Array(CommandLine.arguments.dropFirst())

if arguments.isEmpty {
    print("Usage: task-cli-lite <command>")
} else {
    let command = arguments[0]
    print("Received command: \(command)")
}
~~~

## Semantic Deep Dive

String interpolation is not just a convenience feature.
It is one of the simplest ways to keep output readable while still composing
dynamic values.

Program I/O also creates a discipline:

- input arrives as strings
- your program has to decide what those strings mean
- output should be designed for the human on the other side

That is why `CommandLine.arguments` matters so early.
It is the narrow entrance through which the whole CLI will later grow.

## Code Evolution

A weaker version hard-codes output:

~~~swift
print("Received something")
~~~

A stronger version makes the CLI behavior visible:

~~~swift
let arguments = Array(CommandLine.arguments.dropFirst())

if arguments.isEmpty {
    print("Usage: task-cli-lite <command>")
} else {
    let command = arguments[0]
    print("Received command: \(command)")
}
~~~

This is still simple, but now the program is expressing a real interface.

## Common Mistakes

- treating user input as if it is already valid structure
  At this stage it is just text.
- building unreadable output strings with too much inline noise
  String interpolation helps, but clarity still depends on how you compose the
  message.
- skipping usage text because the program is "only for learning"
  Clear output is part of learning to design programs well.

## Drills

- concept check: why is command-line input best thought of as a boundary rather
  than a convenience?
- code reading: explain what happens when no arguments are supplied
- hands-on extension: add support for printing the raw argument count alongside
  the command

## Checkpoint

You should now be able to read command-line input, explain how string
interpolation improves output clarity, and describe why CLI text is a design
surface instead of a side detail.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Command-line argument | 命令行参数 | A value provided to the program when it starts. |
| String interpolation | 字符串插值 | Embedding values inside a string literal. |

## English Recap

- `CommandLine.arguments` is the raw input boundary for the CLI.
- String interpolation helps make dynamic output readable.
- Good CLI output is part of program design, not an afterthought.

## Project Bridge

At this point, `TaskCLI Lite` can accept a command-shaped string and respond to
it in a human-readable way.
The next step is deciding how the program should branch on those commands.
