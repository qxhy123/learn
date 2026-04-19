// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "TaskCLILite",
    products: [
        .executable(name: "task-cli-lite", targets: ["TaskCLILite"]),
    ],
    targets: [
        .executableTarget(
            name: "TaskCLILite"
        ),
        .testTarget(
            name: "TaskCLILiteTests",
            dependencies: ["TaskCLILite"]
        ),
    ]
)
