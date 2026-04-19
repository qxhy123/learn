// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "TaskCLILite",
    products: [
        .executable(
            name: "TaskCLILite",
            targets: ["TaskCLILite"]
        )
    ],
    targets: [
        .executableTarget(
            name: "TaskCLILite"
        ),
        .testTarget(
            name: "TaskCLILiteTests",
            dependencies: ["TaskCLILite"]
        )
    ]
)
