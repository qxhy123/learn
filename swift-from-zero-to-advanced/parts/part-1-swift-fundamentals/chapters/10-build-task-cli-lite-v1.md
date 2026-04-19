# Chapter 10: Build TaskCLI Lite v1

## Problem

Part 1 has covered many ideas that are easy to understand in isolation:

- values and types
- CLI input and output
- control flow
- functions
- structs
- arrays
- optionals
- enums

The risk at the end of a fundamentals section is obvious:
the reader may know each topic separately, but still not know how those pieces
cohere inside one real program.

That is why Part 1 ends with `TaskCLI Lite v1`.
The goal is not larger architecture.
The goal is integration.

## Running Example

~~~swift
struct Task {
    let title: String
    var isDone: Bool
}

enum Command: String {
    case list
    case add
    case done
}

func printUsage() {
    print("Usage: task-cli-lite <list|add <title>|done <title>>")
}

func listTasks(_ tasks: [Task]) {
    for task in tasks {
        print("- \(task.title) [done: \(task.isDone)]")
    }
}

func addTask(title: String, to tasks: inout [Task]) {
    tasks.append(Task(title: title, isDone: false))
}

func markDone(title: String, in tasks: inout [Task]) {
    for index in tasks.indices {
        if tasks[index].title == title {
            tasks[index].isDone = true
        }
    }
}

var tasks: [Task] = [
    Task(title: "Buy milk", isDone: false),
    Task(title: "Read Swift docs", isDone: false),
]

let arguments = Array(CommandLine.arguments.dropFirst())

if let rawCommand = arguments.first,
   let command = Command(rawValue: rawCommand) {
    switch command {
    case .list:
        listTasks(tasks)
    case .add:
        if let title = arguments.dropFirst().first {
            addTask(title: title, to: &tasks)
            listTasks(tasks)
        } else {
            print("Missing title for add.")
        }
    case .done:
        if let title = arguments.dropFirst().first {
            markDone(title: title, in: &tasks)
            listTasks(tasks)
        } else {
            print("Missing title for done.")
        }
    }
} else {
    printUsage()
}
~~~

## Semantic Deep Dive

This small program works because each Part 1 concept is doing a specific job:

- values hold stable local information such as titles and booleans
- `CommandLine.arguments` provides raw CLI input
- control flow decides what behavior follows from the parsed command
- functions give the program named responsibilities
- `Task` is a struct-based domain model
- `[Task]` stores ordered task state
- optionals protect parsing and missing-argument cases
- `enum Command` models the closed command space

The important lesson is not that this program is "impressive."
It is that the parts finally compose.

That is also why Part 1 ends with a coherent small program rather than bigger
architecture.
If we introduced packages, protocols, persistence layers, testing strategy, or
multi-file design before this integration felt solid, the reader could hide a
weak semantic foundation under more structure.

Part 1 refuses that escape route.
It insists that the language basics must already support a real, readable
program before later parts complicate the system.

## Code Evolution

A weak final version would only gesture at integration:

~~~swift
print("Imagine a task CLI here.")
print("It would probably list, add, and complete tasks.")
~~~

That kind of chapter ending is weak because it does not force the concepts to
meet each other.
It leaves the reader with fragments instead of a program.

A stronger version is still intentionally small, but it is complete enough to
exercise the full Part 1 mental model:

~~~swift
struct Task {
    let title: String
    var isDone: Bool
}

enum Command: String {
    case list
    case add
    case done
}

func handle(command: Command, arguments: ArraySlice<String>, tasks: inout [Task]) {
    switch command {
    case .list:
        listTasks(tasks)
    case .add:
        if let title = arguments.first {
            addTask(title: title, to: &tasks)
        } else {
            print("Missing title for add.")
        }
    case .done:
        if let title = arguments.first {
            markDone(title: title, in: &tasks)
        } else {
            print("Missing title for done.")
        }
    }
}
~~~

This stronger version proves several things:

- the CLI has a real domain type
- command parsing and command execution are connected
- optional handling protects the boundary with the user
- small functions keep the whole program readable
- integration can be real without pretending we already need a larger system

That is the right finish line for Part 1.
Not "production ready."
Not "architecturally complete."
Semantically coherent.

## Common Mistakes

- ending fundamentals with a project brief instead of a working integrated example
  Readers need to see composition, not only intentions.
- adding bigger abstractions just to make the program look advanced
  More layers can hide weak understanding.
- forgetting that optionals and enums are part of the integrated design, not
  isolated chapter topics
- making the final program so large that the reader can no longer inspect it as
  one mental unit

## Drills

- concept check: why does Part 1 end with a coherent small program instead of a
  larger architecture exercise?
- code reading: identify where each of these concepts appears in the running
  example: structs, arrays, optionals, enums, functions
- hands-on extension: add a `help` command that prints usage without expanding
  the architecture beyond Part 1 scope

## Checkpoint

You should now be able to explain how the whole Part 1 toolset composes into
`TaskCLI Lite v1`, why each concept has a distinct job in the program, and why
the correct endpoint for fundamentals is a coherent small integration rather
than premature system design.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Integration | 集成 | Combining multiple language concepts into one coherent working program. |
| Command router | 命令路由器 | The logic that maps a parsed command to program behavior. |
| Program shape | 程序结构 | The readable organization of responsibilities inside a program. |
| Scope | 范围 | The intentional limit of what a chapter or system tries to do. |

## English Recap

- `TaskCLI Lite v1` integrates the entire Part 1 mental model into one program.
- A strong fundamentals ending proves composition, not just isolated syntax knowledge.
- Part 1 stays intentionally small so the reader can still see the semantics clearly.
- Larger architecture belongs later, after this foundation is stable.

## Project Bridge

Part 1 is now complete.
The reader has a small but coherent CLI and a working model of Swift
fundamentals.
Part 2 can now ask a better question:
once this program starts growing, how should its architecture, boundaries, and
testing strategy evolve without losing the semantic clarity established here?
