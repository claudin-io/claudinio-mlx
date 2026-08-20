import Foundation
import MLXLLM
import MLXLMCommon

/// `claudinio-mlx` — the MLX engine for local inference on Apple Silicon.
///
/// Two subcommands on purpose: `serve` is what the app launches, `bench` is how
/// you tell "the engine is broken" apart from "the server wiring is broken"
/// when a model produces nothing.
func usage() -> Never {
    print(
        """
        usage:
          claudinio-mlx serve --model <dir> [--host 127.0.0.1] [--port N]
                              [--alias NAME] [--api-key KEY] [--ctx-size N]
          claudinio-mlx bench --model <dir> [--prompt TEXT]
        """)
    exit(2)
}

func flag(_ name: String, in args: [String]) -> String? {
    guard let i = args.firstIndex(of: "--\(name)"), i + 1 < args.count else { return nil }
    return args[i + 1]
}

let args = CommandLine.arguments
guard args.count > 1 else { usage() }
guard let modelPath = flag("model", in: args) else { usage() }
let directory = URL(fileURLWithPath: modelPath)
let alias = flag("alias", in: args) ?? directory.lastPathComponent
let ctxSize = Int(flag("ctx-size", in: args) ?? "") ?? 0

switch args[1] {
case "serve":
    let address = flag("host", in: args) ?? "127.0.0.1"
    let port = Int(flag("port", in: args) ?? "") ?? 8080
    guard let apiKey = flag("api-key", in: args), !apiKey.isEmpty else {
        // Refused rather than defaulted: an unauthenticated inference endpoint
        // on loopback is reachable by every process on the machine, and by any
        // page the agent's own browser visits.
        FileHandle.standardError.write(Data("--api-key is required\n".utf8))
        exit(2)
    }
    // Load before binding, so a successful connection means a ready model and
    // /health never lies.
    let host = try await ModelHost.load(
        directory: directory, alias: alias, ctxSize: ctxSize)
    try await runServer(host: host, address: address, port: port, apiKey: apiKey)

case "bench":
    let prompt = flag("prompt", in: args) ?? "Count to 20."
    let host = try await ModelHost.load(directory: directory, alias: alias, ctxSize: ctxSize)
    try await host.generate(
        messages: [.user(prompt)], tools: nil, maxTokens: 128,
        temperature: nil, topP: nil
    ) { event in
        switch event {
        case .chunk(let text):
            FileHandle.standardOutput.write(Data(text.utf8))
        case .toolCall(let name, let arguments):
            print("\n[tool call] \(name)(\(arguments))")
        case .done(let promptTokens, let completionTokens, let reason):
            print("\n---\nprompt \(promptTokens) tok, generated \(completionTokens) tok, \(reason)")
        }
    }
    let stats = await host.stats()
    print(
        String(
            format: "generation: %.2f tok/s (prompt %.2f tok/s)",
            stats.tokensPerSecond, stats.promptTokensPerSecond))

default:
    usage()
}
