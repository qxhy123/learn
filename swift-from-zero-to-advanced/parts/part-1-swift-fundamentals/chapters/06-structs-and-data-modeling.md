# Chapter 06: Structs and Data Modeling

## Problem

Up to this point, the CLI can still fake its way forward with separate values:

- a title string
- a done flag
- maybe a priority integer

But once those values belong to one conceptual thing, leaving them separate
stops being clarity and starts being friction.

## Running Example

~~~swift
struct Task {
    let title: String
    var isDone: Bool
}

let firstTask = Task(title: "Buy milk", isDone: false)
print(firstTask.title)
~~~

## Semantic Deep Dive

`struct` is one of the most important parts of early Swift thinking.
It gives you a way to model related data as one named value.

Struct（结构体） is not just a prettier dictionary.
It does several things at once:

- gives the data a real name
- makes the fields explicit
- supports value-oriented design

This is where Value semantics（值语义） starts to become more than vocabulary.
If you copy a value type, Swift wants you to reason in terms of independent
values rather than shared mutable identity.

## Code Evolution

A weaker version spreads task data across independent values:

~~~swift
let title = "Buy milk"
var isDone = false
~~~

A stronger version models the domain explicitly:

~~~swift
struct Task {
    let title: String
    var isDone: Bool
}
~~~

This is better because the code is finally speaking in domain units instead of
just in raw fields.

## Common Mistakes

- delaying `struct` because separate values still "work"
- confusing "value type" with "immutable forever"
- using a dictionary when the domain shape is already known

## Drills

- concept check: why is `Task` a better unit than separate `title` and `isDone`
  values?
- code reading: explain what information the struct declaration makes visible
- hands-on extension: add a `priority` field and decide whether it should use a
  default value

## Checkpoint

You should now be able to explain why a struct is a modeling tool, not merely a
syntax feature, and why it moves the CLI closer to a real program.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Struct | 结构体 | A value type used to model grouped data. |
| Value semantics | 值语义 | Treating copied values as independent values rather than shared mutable identity. |

## English Recap

- A `struct` gives the domain a named data model.
- Swift encourages value-oriented modeling early.
- `Task` is the first real domain object in the CLI.

## Project Bridge

`TaskCLI Lite` no longer has to pretend task data is just loose local state.
It can now store and move around real task values, which makes the next topic
natural: collections.
