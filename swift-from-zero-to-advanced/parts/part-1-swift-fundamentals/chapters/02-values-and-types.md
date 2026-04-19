# Chapter 02: Values and Types

## Problem

Small programs feel easy until their values stop being obvious.
Swift makes an early promise: values should be readable, type information should
stay visible, and mutability should be explicit.

If you ignore that promise, your Part 1 code will still run, but it will stop
feeling like Swift.

## Running Example

Start with a tiny slice of task data:

~~~swift
let title = "Buy milk"
var isDone = false
let priority: Int = 2

print(title, isDone, priority)
~~~

## Semantic Deep Dive

Swift uses `let` for constants and `var` for values that may change.
That looks simple, but it changes how you think about state:

- values should be immutable unless mutation is part of the model
- types can often be inferred, but type inference is not a reason to hide
  useful intent

Type inference（类型推断） works well when the expression is already clear.
Explicit annotations are better when the annotation adds information the reader
would otherwise have to reconstruct.

## Code Evolution

Here is a noisier first version:

~~~swift
var taskTitle: String = "Buy milk"
var taskDone: Bool = false
var taskPriority: Int = 2
~~~

Now tighten the model:

~~~swift
let title = "Buy milk"
var isDone = false
let priority: Int = 2
~~~

This version is better for Part 1 because:

- names are shorter but still domain-specific
- only the value that may actually change uses `var`
- the one explicit type annotation shows where type clarity matters

## Common Mistakes

- using `var` everywhere "just in case"
  That weakens the meaning of mutation in the program.
- adding explicit types to every line
  That often makes the code louder without making it clearer.
- treating type inference as if types disappear
  They do not disappear; Swift still has a concrete type system underneath the
  syntax.

## Drills

- concept check: when is `let` better than `var` even if the code would still
  compile with `var`?
- code reading: name the inferred types in the running example
- hands-on extension: add a `notes` value and decide whether it should use type
  inference or an explicit annotation

## Checkpoint

You should now be able to explain why Swift makes mutability explicit and how
type inference can help without replacing deliberate type design.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Type inference | 类型推断 | Swift infers a type from the expression and context. |
| Mutability | 可变性 | Whether a value is allowed to change after declaration. |

## English Recap

- `let` communicates stable intent.
- `var` should mean real mutation, not convenience.
- Type inference is helpful when the expression is already clear.

## Project Bridge

The task CLI cannot stay as raw print statements forever.
It needs stable values, meaningful names, and intentional mutation before the
command logic starts to grow.
