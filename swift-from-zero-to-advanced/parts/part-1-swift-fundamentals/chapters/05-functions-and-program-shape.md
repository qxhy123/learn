# Chapter 05: Functions and Program Shape

## Problem

As soon as a CLI has more than one branch, the file starts to feel heavier.
The question becomes:

How do we keep the program readable without pretending we need "architecture"
before we have even learned the language well?

## Running Example

~~~swift
func printUsage() {
    print("Usage: task-cli-lite <list|add>")
}

func handleList() {
    print("Listing tasks")
}

func handleAdd() {
    print("Adding a task")
}
~~~

## Semantic Deep Dive

Functions are not only about reuse.
They are also about shape.

When a program has a visible set of responsibilities, named functions let the
reader see those responsibilities directly:

- print usage
- handle list
- handle add

That is much easier to reason about than one long body that mixes output,
control flow, and data mutation.

## Code Evolution

A weaker version keeps everything inline.
A stronger version extracts named operations:

~~~swift
func handle(command: String) {
    switch command {
    case "list":
        handleList()
    case "add":
        handleAdd()
    default:
        printUsage()
    }
}
~~~

This still is not over-engineered.
It is simply a clearer program shape.

## Common Mistakes

- extracting functions without a clear responsibility
- naming functions after mechanics instead of intent
- pushing too many unrelated values through a function signature

## Drills

- concept check: why are functions about program shape, not only reuse?
- code reading: identify the responsibility boundary of `handle(command:)`
- hands-on extension: split one inline fallback path into a named helper

## Checkpoint

You should now be able to explain why a small CLI benefits from named
responsibilities before it benefits from larger abstractions.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Function signature | 函数签名 | The name, parameters, and return information of a function. |
| Responsibility | 职责 | The single job a function or unit should own. |

## English Recap

- Functions improve the shape of a program, not only its reuse story.
- Good function names expose intent.
- Small, clear helpers are enough for Part 1.

## Project Bridge

`TaskCLI Lite` now has a more readable command path.
The next step is data modeling: the program still talks mostly in loose values,
which is where Swift's `struct` story begins to matter.
