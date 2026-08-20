import Foundation
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

/// Lifetime counters, mirroring what llama-server's `/metrics` reports so the
/// status bar reads the same whichever engine is running.
actor Counters {
    private var tokensGenerated = 0
    private var generationSeconds = 0.0
    private var promptTokens = 0
    private var promptSeconds = 0.0
    private var lastContextTokens = 0
    private var inFlight = 0

    func begin() { inFlight += 1 }
    func end() { inFlight = max(0, inFlight - 1) }

    func record(
        promptTokens: Int, promptTime: TimeInterval,
        generationTokens: Int, generateTime: TimeInterval
    ) {
        self.promptTokens += promptTokens
        self.promptSeconds += promptTime
        self.tokensGenerated += generationTokens
        self.generationSeconds += generateTime
        self.lastContextTokens = promptTokens + generationTokens
    }

    func snapshot(alias: String, ctxSize: Int) -> Stats {
        Stats(
            modelKey: alias,
            ctxSize: ctxSize,
            ctxUsed: lastContextTokens,
            tokensPerSecond: generationSeconds > 0
                ? Double(tokensGenerated) / generationSeconds : 0,
            promptTokensPerSecond: promptSeconds > 0
                ? Double(promptTokens) / promptSeconds : 0,
            tokensGenerated: tokensGenerated,
            busy: inFlight > 0
        )
    }
}

/// Owns the loaded model.
///
/// Exclusive access to the weights is `ModelContainer`'s job: MLX evaluation is
/// not reentrant, and two concurrent requests against one context corrupt the
/// KV cache rather than merely contending. This type deliberately does *not*
/// add an actor of its own around generation — doing so makes the (non-Sendable)
/// `UserInput` actor-isolated, which the compiler correctly refuses to let
/// cross back out.
final class ModelHost: Sendable {
    private let container: ModelContainer
    private let counters = Counters()
    let alias: String
    let ctxSize: Int

    private init(container: ModelContainer, alias: String, ctxSize: Int) {
        self.container = container
        self.alias = alias
        self.ctxSize = ctxSize
    }

    /// Load the weights up front, so `/health` can answer truthfully instead of
    /// the first request silently paying for a multi-second load.
    ///
    /// Weights come off disk: the Rust side downloads and verifies them against
    /// the Hub's own sha256, so this never reaches the network.
    static func load(directory: URL, alias: String, ctxSize: Int) async throws -> ModelHost {
        let container = try await LLMModelFactory.shared.loadContainer(
            from: directory,
            using: #huggingFaceTokenizerLoader()
        )
        return ModelHost(container: container, alias: alias, ctxSize: ctxSize)
    }

    func stats() async -> Stats {
        await counters.snapshot(alias: alias, ctxSize: ctxSize)
    }

    /// What one generation produced, in the order the model produced it.
    enum Event: Sendable {
        case chunk(String)
        case toolCall(name: String, arguments: String)
        case done(promptTokens: Int, completionTokens: Int, finishReason: String)
    }

    /// Generate, handing each event to `emit` as it happens.
    /// `messages` is `sending`: `Chat.Message` is not Sendable, and the
    /// `UserInput` built from it has to cross into the container's isolation.
    /// Transferring ownership is what lets the compiler prove nobody else can
    /// still be holding it.
    func generate(
        messages: sending [Chat.Message],
        tools: [ToolSpec]?,
        maxTokens: Int,
        temperature: Float?,
        topP: Float?,
        emit: @Sendable (Event) async -> Void
    ) async throws {
        await counters.begin()
        defer { Task { await counters.end() } }

        var parameters = GenerateParameters(maxTokens: maxTokens)
        if let temperature { parameters.temperature = temperature }
        if let topP { parameters.topP = topP }

        let input = try await container.prepare(
            input: UserInput(chat: messages, tools: tools))
        let promptTokenCount = input.text.tokens.size

        var completionTokens = 0
        var finishReason = "stop"

        let stream = try await container.generate(input: input, parameters: parameters)
        for await event in stream {
            switch event {
            case .chunk(let text):
                await emit(.chunk(text))
            case .toolCall(let call):
                // A call wins over the stop reason: the turn ended in a tool
                // call even though generation itself stopped normally, and the
                // agent loop keys off exactly this field.
                finishReason = "tool_calls"
                await emit(
                    .toolCall(
                        name: call.function.name,
                        arguments: Self.encodeArguments(call.function.arguments)))
            case .info(let info):
                completionTokens = info.generationTokenCount
                await counters.record(
                    promptTokens: info.promptTokenCount,
                    promptTime: info.promptTime,
                    generationTokens: info.generationTokenCount,
                    generateTime: info.generateTime)
                if finishReason != "tool_calls" {
                    switch info.stopReason {
                    case .length: finishReason = "length"
                    case .cancelled: finishReason = "cancelled"
                    case .stop: finishReason = "stop"
                    @unknown default: finishReason = "stop"
                    }
                }
            @unknown default:
                break
            }
        }

        await emit(
            .done(
                promptTokens: promptTokenCount,
                completionTokens: completionTokens,
                finishReason: finishReason))
    }

    /// Tool arguments go out as a JSON string: that is what the OpenAI shape
    /// specifies and what `openai.rs` parses on the other side.
    ///
    /// Encoded through `JSONEncoder` rather than `JSONSerialization`: the
    /// arguments are `[String: MLXLMCommon.JSONValue]`, which is not a
    /// Foundation object graph, so `isValidJSONObject` rejects it and every
    /// call would ship `{}` — a tool call with no arguments, which reads as the
    /// model's fault rather than ours.
    private static func encodeArguments(_ arguments: [String: JSONValue]) -> String {
        let encoder = JSONEncoder()
        // Otherwise a path argument ships as "src\/main.rs": valid JSON, but
        // noise in every log and diff that quotes a tool call.
        encoder.outputFormatting = .withoutEscapingSlashes
        guard let data = try? encoder.encode(arguments),
            let text = String(data: data, encoding: .utf8)
        else {
            return "{}"
        }
        return text
    }
}

enum ServerError: Error, CustomStringConvertible {
    case notLoaded
    case badRequest(String)

    var description: String {
        switch self {
        case .notLoaded: return "the model is still loading"
        case .badRequest(let m): return m
        }
    }
}
