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

/// Report a failure the way the supervisor can show it.
///
/// Swift's top level turns an uncaught error into a fatal error: the process
/// dies with SIGTRAP and a `Swift/ErrorType.swift:254` stack trace, which
/// reaches the user as "exited during startup (signal: 5)" and says nothing
/// about what went wrong. Every failure path here exits with a message on
/// stderr instead, because stderr is what the supervisor tails.
func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(1)
}

/// Turn a load failure into something the user can act on.
func describeLoadFailure(_ error: any Error, directory: URL) -> String {
    let detail = String(describing: error)
    let entries = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
    let hasSafetensors = entries.contains { $0.hasSuffix(".safetensors") }
    let hasGguf = entries.contains { $0.lowercased().hasSuffix(".gguf") }

    // The likeliest cause is a directory that is not an MLX model at all —
    // most often a GGUF, which belongs to the other engine.
    if hasGguf && !hasSafetensors {
        let name = directory.lastPathComponent
        return "\(name) is a GGUF model, which MLX cannot load; it needs an MLX (safetensors) model. Original error: \(detail)"
    }
    if !hasSafetensors {
        return "\(directory.path) does not look like an MLX model: no .safetensors file. Original error: \(detail)"
    }
    return "could not load the model at \(directory.path): \(detail)"
}

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
    let host: ModelHost
    do {
        host = try await ModelHost.load(directory: directory, alias: alias, ctxSize: ctxSize)
    } catch {
        fail(describeLoadFailure(error, directory: directory))
    }
    do {
        try await runServer(host: host, address: address, port: port, apiKey: apiKey)
    } catch {
        fail("the server stopped: \(error)")
    }

case "bench":
    let prompt = flag("prompt", in: args) ?? "Count to 20."
    let host: ModelHost
    do {
        host = try await ModelHost.load(directory: directory, alias: alias, ctxSize: ctxSize)
    } catch {
        fail(describeLoadFailure(error, directory: directory))
    }
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
