// swift-tools-version: 6.2
import PackageDescription

// The MLX engine for local inference on Apple Silicon, built as our own
// sidecar rather than driving Apple's Python `mlx-lm`.
//
// Three reasons this exists instead of shelling out to mlx_lm.server:
//   - no Python runtime to provision on the user's machine;
//   - mlx_lm.server has no authentication at all, while this speaks the same
//     random per-process api-key the llama.cpp sidecar does;
//   - tool-call parsing comes from ml-explore's own per-family parsers
//     (Qwen35, Llama3, Mistral, GLM4, Harmony…), which is the part that fails
//     silently if reimplemented.
let package = Package(
    name: "claudinio-mlx",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "claudinio-mlx", targets: ["claudinio-mlx"])
    ],
    dependencies: [
        // Pinned exactly, like every other artifact the app fetches.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", exact: "3.31.4"),
        .package(url: "https://github.com/apple/swift-nio", from: "2.83.0"),
        // mlx-swift-lm deliberately does not depend on this: its tokenizer
        // loader is a macro that expands into the consumer's `Tokenizers`, so
        // the library never forces a tokenizer choice. We make that choice.
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3"),
    ],
    targets: [
        .executableTarget(
            name: "claudinio-mlx",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                // Multimodal checkpoints keep their language weights under
                // `language_model.*`; loading one with the text-only factory
                // finds nothing there and generates from uninitialized weights,
                // which comes out as mojibake rather than as an error.
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                // Adapts swift-transformers' AutoTokenizer to MLX's protocol.
                // Only the tokenizer half is used: weights are fetched by the
                // Rust side, which already verifies them against the Hub's own
                // sha256, so the Swift downloader is never involved.
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ]
        ),
        .testTarget(
            name: "claudinio-mlxTests",
            dependencies: ["claudinio-mlx"]
        ),
    ]
)
