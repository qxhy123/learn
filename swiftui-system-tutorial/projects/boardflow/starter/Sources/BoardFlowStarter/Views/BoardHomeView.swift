import SwiftUI

struct BoardHomeView: View {
    let document: BoardDocument

    var body: some View {
        NavigationSplitView {
            List(document.boards) { board in
                Label(board.title, systemImage: "square.on.square")
            }
            .navigationTitle("Boards")
        } detail: {
            VStack(alignment: .leading, spacing: 16) {
                Text(document.title)
                    .font(.largeTitle.bold())
                Text("BoardFlow starter for Part 1 and Part 2")
                    .foregroundStyle(.secondary)
                Text("Recent boards")
                    .font(.headline)
                ForEach(document.boards) { board in
                    HStack {
                        Text(board.title)
                        Spacer()
                        Text("\(board.cardCount) cards")
                            .foregroundStyle(.secondary)
                    }
                }
                Spacer()
            }
            .padding(24)
        }
    }
}
