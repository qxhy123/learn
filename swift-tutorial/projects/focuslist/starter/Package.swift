// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "FocusListStarter",
    platforms: [
        .macOS(.v15),
        .iOS(.v18)
    ],
    products: [
        .library(name: "FocusCore", targets: ["FocusCore"]),
        .executable(name: "FocusListApp", targets: ["FocusListApp"]),
        .executable(name: "focusctl", targets: ["focusctl"])
    ],
    targets: [
        .target(name: "FocusCore"),
        .executableTarget(
            name: "FocusListApp",
            dependencies: ["FocusCore"]
        ),
        .executableTarget(
            name: "focusctl",
            dependencies: ["FocusCore"]
        ),
        .testTarget(
            name: "FocusCoreTests",
            dependencies: ["FocusCore"]
        )
    ]
)
