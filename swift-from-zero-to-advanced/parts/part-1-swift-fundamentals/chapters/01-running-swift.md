# Chapter 01: Running Swift

## Problem

Before Swift can feel like a language, it has to feel like a real toolchain.
Many readers coming from Python, JavaScript, Go, or Java have an immediate
question:

What exactly is the difference between running Swift code with `swift` and
building a program with `swiftc`?

If that question stays fuzzy, later chapters also stay fuzzy.

## Running Example

Create a file called `hello.swift`:

~~~swift
print("Hello from Swift")
print("Arguments:", CommandLine.arguments.dropFirst())
~~~

Run it as a script:

~~~bash
swift hello.swift one two
~~~

Then compile it:

~~~bash
swiftc hello.swift -o hello
./hello one two
~~~

## Semantic Deep Dive

`swift` is the command you use when you want the Swift toolchain to interpret
and execute the source directly as a script-like workflow.
`swiftc` is the compiler driver that turns Swift source into an executable.

For Part 1, the important point is not the internal implementation detail.
The important point is the development model:

- `swift` is quick for experimentation
- `swiftc` makes the "this is a program" boundary explicit

That distinction matters because `TaskCLI Lite` will eventually live in a real
Swift package, not as a permanent throwaway script.

## Code Evolution

The first version prints raw output.
The second version makes the command-line boundary explicit:

~~~swift
let arguments = Array(CommandLine.arguments.dropFirst())

if arguments.isEmpty {
    print("No command provided.")
} else {
    print("Command:", arguments[0])
}
~~~

This is still tiny, but it already does two useful things:

- it treats the outside world as input, not magic
- it makes later command handling possible

## Common Mistakes

- assuming `CommandLine.arguments` contains only user arguments
  It also contains the program path as the first element.
- assuming `swift` and `swiftc` are interchangeable
  They can run the same source, but they express different development intents.
- delaying the executable model too long
  If you stay in "just a script" mode forever, later package structure feels
  unnatural.

## Drills

- concept check: explain the difference between `swift` and `swiftc` in one
  sentence each
- code reading: say what `CommandLine.arguments.dropFirst()` removes and why
- hands-on extension: print the number of user-supplied arguments before
  printing the first command

## Checkpoint

You should now be able to explain how a Swift source file becomes running
program behavior and why command-line arguments are already a useful boundary.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Toolchain | 工具链 | The commands used to build and run Swift code. |
| Executable | 可执行程序 | A compiled program that can be launched directly. |

## English Recap

- `swift` is useful for quick execution and exploration.
- `swiftc` makes the compilation step explicit.
- `CommandLine.arguments` is the first bridge between the program and the user.

## Project Bridge

`TaskCLI Lite` starts as "a file that can run."
That sounds small, but it is the first step toward a command-driven program
instead of a disconnected code snippet.
