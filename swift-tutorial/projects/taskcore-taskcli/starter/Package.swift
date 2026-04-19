// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "TaskCoreTaskCLI",
    products: [
        .library(
            name: "TaskCore",
            targets: ["TaskCore"]
        ),
        .executable(
            name: "TaskCLI",
            targets: ["TaskCLI"]
        ),
    ],
    targets: [
        .target(
            name: "TaskCore"
        ),
        .executableTarget(
            name: "TaskCLI",
            dependencies: ["TaskCore"]
        ),
        .testTarget(
            name: "TaskCoreTests",
            dependencies: ["TaskCore"]
        ),
    ]
)
