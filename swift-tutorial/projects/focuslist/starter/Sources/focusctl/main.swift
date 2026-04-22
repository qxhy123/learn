import FocusCore

let store = FocusStore.sample()
print("FocusList inbox")
for task in store.inboxTasks {
    let mark = task.isDone ? "[x]" : "[ ]"
    print("\(mark) \(task.title)")
}
