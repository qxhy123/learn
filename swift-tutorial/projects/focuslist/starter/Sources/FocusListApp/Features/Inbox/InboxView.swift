import SwiftUI
import FocusCore

struct InboxView: View {
    @Bindable var store: FocusStore
    @State private var draftTitle = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                TextField("Add a task", text: $draftTitle)
                    .textFieldStyle(.roundedBorder)
                Button("Add") {
                    store.addTask(title: draftTitle)
                    draftTitle = ""
                }
                .disabled(draftTitle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }

            List(store.inboxTasks) { task in
                Button {
                    store.toggleCompletion(task.id)
                } label: {
                    HStack {
                        Image(systemName: task.isDone ? "checkmark.circle.fill" : "circle")
                        Text(task.title)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.plain)
            }
        }
        .padding()
        .navigationTitle("Inbox")
    }
}
