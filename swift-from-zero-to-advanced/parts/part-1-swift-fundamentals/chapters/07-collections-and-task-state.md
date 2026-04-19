# Chapter 07: Collections and Task State

## Problem

One `Task` is a modeling exercise.
Multiple `Task` values are a program.

As soon as the CLI needs to list, add, or update more than one task, the
question becomes:

What collection shape gives us readable, predictable task state?

## Running Example

~~~swift
struct Task {
    let title: String
    var isDone: Bool
}

var tasks: [Task] = [
    Task(title: "Buy milk", isDone: false),
    Task(title: "Read Swift docs", isDone: true),
]
~~~

## Semantic Deep Dive

For Part 1, an array is the right first collection because it preserves order
and keeps the mental model simple.

Swift also gives you dictionaries and sets, but they solve different problems:

- arrays preserve sequence
- dictionaries optimize lookup by key
- sets optimize uniqueness

The best collection is not the one you personally like most.
It is the one that matches the access pattern of the program.

## Code Evolution

A weaker version hard-codes one task.
A stronger version moves to `[Task]`:

~~~swift
for task in tasks {
    print("- \(task.title) [done: \(task.isDone)]")
}
~~~

Now the CLI can iterate over real state instead of one-off examples.

## Common Mistakes

- choosing a set just because uniqueness sounds attractive
- choosing a dictionary before the program really has stable lookup keys
- mutating collection state inline everywhere without a clear update path

## Drills

- concept check: why is `[Task]` the right first collection for Part 1?
- code reading: explain what the loop prints for the sample state
- hands-on extension: append one extra task and print the updated list

## Checkpoint

You should now be able to compare arrays, dictionaries, and sets in terms of
program needs and explain why task state begins with an array in Part 1.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Array | 数组 | An ordered collection of values. |
| State | 状态 | The current data the program is holding and changing. |

## English Recap

- `[Task]` is the right first storage model for Part 1.
- Collection choice should follow access patterns.
- Arrays make the CLI's task state visible and iterable.

## Project Bridge

`TaskCLI Lite` can now hold real task state.
The next problem is safety: user input is still messy, and the program still
needs a better way to represent missing or invalid command data.
