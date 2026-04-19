// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "BoardFlowStarter",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "BoardFlowStarter", targets: ["BoardFlowStarter"])
    ],
    targets: [
        .executableTarget(name: "BoardFlowStarter"),
        .testTarget(name: "BoardFlowStarterTests", dependencies: ["BoardFlowStarter"])
    ]
)
