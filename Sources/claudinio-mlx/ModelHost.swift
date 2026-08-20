import Foundation
import MLXHuggingFace
import MLXLLM
import MLXVLM
import MLX
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

    /// `promptTokens` is what this request actually prefilled, which after a
    /// cache hit is only the new tail — that is the number the prompt-rate
    /// stat wants. `contextTokens` is the whole window the model saw, which is
    /// what the context gauge wants. They stopped being the same number when
    /// the prefix cache landed.
    func record(
        promptTokens: Int, promptTime: TimeInterval,
        generationTokens: Int, generateTime: TimeInterval,
        contextTokens: Int
    ) {
        self.promptTokens += promptTokens
        self.promptSeconds += promptTime
        self.tokensGenerated += generationTokens
        self.generationSeconds += generateTime
        self.lastContextTokens = contextTokens
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
    private let gate = GenerationGate()
    private let prefix = PrefixCache()
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
        if let reason = ModelCompatibility.reasonToRefuse(directory: directory) {
            throw ServerError.unsupportedModel(reason)
        }
        // A multimodal checkpoint keeps its language weights under
        // `language_model.*`. The text-only factory looks for `model.*`, finds
        // nothing, and generates from uninitialized weights — which reads as a
        // broken app, not an unsupported model.
        let container: ModelContainer
        if ModelCompatibility.isMultimodal(directory: directory) {
            container = try await VLMModelFactory.shared.loadContainer(
                from: directory,
                using: #huggingFaceTokenizerLoader()
            )
        } else {
            container = try await LLMModelFactory.shared.loadContainer(
                from: directory,
                using: #huggingFaceTokenizerLoader()
            )
        }
        return ModelHost(container: container, alias: alias, ctxSize: ctxSize)
    }

    func stats() async -> Stats {
        var stats = await counters.snapshot(alias: alias, ctxSize: ctxSize)
        // Weights live in Metal buffers, which do not show up in the process's
        // resident set: RSS reported ~80 MB for a model the OS was accounting
        // 49 GB for. MLX knows what it allocated, so it says so.
        stats.memoryBytes = GPU.snapshot().activeMemory
        return stats
    }

    /// What one generation produced, in the order the model produced it.
    enum Event: Sendable {
        case chunk(String)
        case toolCall(name: String, arguments: String)
        case done(promptTokens: Int, completionTokens: Int, finishReason: String)
    }

    /// What one generation cost, carried back out of the model's isolation.
    /// Every field is a plain number: nothing here may reference `MLXArray`.
    private struct Summary: Sendable {
        var promptTokens: Int
        var processedTokens: Int
        var completionTokens: Int
        var finishReason: String
        var promptTime: TimeInterval
        var generateTime: TimeInterval
        var reused: Int
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
        // Held for the whole generation, not just the prefill: the KV cache
        // outlives the request now, so two overlapping requests would write
        // into the same recurrence.
        await gate.acquire()
        defer { Task { [gate] in await gate.release() } }

        await counters.begin()
        defer { Task { await counters.end() } }

        // Immutable so it can cross into the model's isolation below.
        let parameters: GenerateParameters = {
            var p = GenerateParameters(maxTokens: maxTokens)
            if let temperature { p.temperature = temperature }
            if let topP { p.topP = topP }
            return p
        }()

        let input = try await container.prepare(
            input: UserInput(chat: messages, tools: tools))
        let promptTokenCount = input.text.tokens.size

        // Reusing a cache means driving the token loop here, which means
        // owning the stop conditions too. Where this loop would only be a
        // lookalike of upstream's, the request takes upstream's instead:
        //   - stop strings, whose partial-match buffering is a private type;
        //   - a quantized cache, whose entries upstream swaps out from under
        //     the array we kept, leaving our reference stale;
        //   - anything but plain text, where the prompt is not a flat token
        //     run that a suffix can be appended to.
        //
        // The mask is the subtle one. A multimodal processor emits an
        // attention mask even for text-only input, and for a single unpadded
        // sequence it is all ones — which carries nothing `sequenceLengths`
        // would not derive from the token count anyway. That case is fine.
        // A mask with a zero in it means padding or a batch, and neither
        // survives having a suffix appended.
        let singleUnpaddedSequence =
            input.text.tokens.ndim == 2 && input.text.tokens.dim(0) == 1
            && {
                guard let mask = input.text.mask else { return true }
                return mask.size == mask.asType(.int32).sum().item(Int.self)
            }()

        let cacheable =
            await container.configuration.effectiveStopStrings.isEmpty
            && parameters.kvBits == nil
            && input.image == nil && input.video == nil && input.audio == nil
            && singleUnpaddedSequence

        guard cacheable else {
            try await generateUncached(
                input: input, parameters: parameters,
                promptTokenCount: promptTokenCount, emit: emit)
            return
        }

        let promptTokens = input.text.tokens.asArray(Int32.self).map(Int.init)

        let summary = try await container.perform(nonSendable: input) {
            context, input -> Summary in
            /// One slice of the prompt, as the model wants it. The mask is
            /// dropped on purpose: `cacheable` already established it was all
            /// ones, and `sequenceLengths` derives the same thing from the
            /// token count.
            func slice(_ range: Range<Int>) -> LMInput {
                let tokens = promptTokens[range].map(Int32.init)
                return LMInput(
                    tokens: MLXArray(tokens).reshaped([1, tokens.count]))
            }

            let cache: [KVCache]
            var base: Int
            switch self.prefix.plan(for: promptTokens) {
            case .cold:
                cache = context.model.newCache(parameters: parameters)
                base = 0
            case .fromPin(let copy, let prefix):
                cache = copy
                base = prefix
            }
            let reused = base

            // Snapshot the cache partway through the prefill, at the boundary
            // this prompt and the previous one still agree on. Splitting the
            // prefill in two costs one extra sampling step at the seam and
            // buys a pin the *next* request can start from — which is the
            // whole point, since by the time generation is done the cache has
            // run past any boundary a later prompt will still match.
            if let pinAt = self.prefix.pinLength(for: promptTokens, startingAt: base) {
                _ = try TokenIterator(
                    input: slice(base ..< pinAt), model: context.model,
                    cache: cache, parameters: parameters)
                self.prefix.pin(
                    tokens: Array(promptTokens[0 ..< pinAt]), cache: cache)
                base = pinAt
            }

            var iterator = try TokenIterator(
                input: slice(base ..< promptTokens.count), model: context.model,
                cache: cache, parameters: parameters)

            let stopTokenIds = Self.stopTokenIds(
                configuration: context.configuration,
                tokenizer: context.tokenizer)
            var detokenizer = NaiveStreamingDetokenizer(
                tokenizer: context.tokenizer)
            // Upstream's own per-family parser. Picking the format is the part
            // that fails silently if reimplemented, so it stays theirs.
            let toolProcessor = ToolCallProcessor(
                format: context.configuration.toolCallFormat ?? .json)

            var completionTokens = 0
            var emittedToolCalls = 0
            var transcript = ""
            var finishReason = "stop"
            var stopped = false

            let started = Date.timeIntervalSinceReferenceDate
            var promptTime: TimeInterval = 0
            var generationStarted = started

            func drainToolCalls() async {
                while emittedToolCalls < toolProcessor.toolCalls.count {
                    let call = toolProcessor.toolCalls[emittedToolCalls]
                    emittedToolCalls += 1
                    // A call wins over the stop reason: the turn ended in a
                    // tool call even though generation itself stopped
                    // normally, and the agent loop keys off exactly this.
                    finishReason = "tool_calls"
                    await emit(
                        .toolCall(
                            name: call.function.name,
                            arguments: Self.encodeArguments(
                                call.function.arguments)))
                }
            }

            while let token = iterator.next() {
                // The first token to come back is the one the prefill
                // produced, so everything up to here was prompt time.
                if promptTime == 0 {
                    generationStarted = Date.timeIntervalSinceReferenceDate
                    promptTime = generationStarted - started
                }
                if Task.isCancelled {
                    finishReason = "cancelled"
                    stopped = true
                    break
                }
                if token == context.tokenizer.unknownTokenId
                    || stopTokenIds.contains(token)
                {
                    stopped = true
                    break
                }

                completionTokens += 1
                detokenizer.append(token: token)
                guard let chunk = detokenizer.next() else { continue }
                transcript += chunk
                if let text = toolProcessor.processChunk(chunk), !text.isEmpty {
                    await emit(.chunk(text))
                }
                await drainToolCalls()
            }

            if !stopped, finishReason != "tool_calls",
                let limit = iterator.maxTokens, iterator.tokenCount >= limit
            {
                finishReason = "length"
            }

            if let buffered = toolProcessor.processEOS(returnBufferedText: true),
                !buffered.isEmpty
            {
                transcript += buffered
                await emit(.chunk(buffered))
            }
            await drainToolCalls()

            // A tool call the model clearly made, that the inferred parser did
            // not recognise, is worse than no tool calling at all: the agent
            // waits for a call that arrived as prose. Upstream picks the parser
            // by scanning the chat template, and a template that merely
            // *mentions* another dialect's marker wins the wrong match —
            // Qwen3-VL's template names `[TOOL_CALLS]` 44 times while the model
            // emits `<tool_call>`.
            if emittedToolCalls == 0, !transcript.isEmpty {
                for call in Self.recoverToolCalls(from: transcript) {
                    finishReason = "tool_calls"
                    await emit(
                        .toolCall(name: call.name, arguments: call.arguments))
                }
            }

            self.prefix.notePrompt(promptTokens)

            return Summary(
                promptTokens: promptTokens.count,
                processedTokens: promptTokens.count - reused,
                completionTokens: completionTokens,
                finishReason: finishReason,
                promptTime: promptTime,
                generateTime: Date.timeIntervalSinceReferenceDate
                    - generationStarted,
                reused: reused)
        }

        await counters.record(
            promptTokens: summary.processedTokens,
            promptTime: summary.promptTime,
            generationTokens: summary.completionTokens,
            generateTime: summary.generateTime,
            contextTokens: summary.promptTokens + summary.completionTokens)

        await emit(
            .done(
                promptTokens: summary.promptTokens,
                completionTokens: summary.completionTokens,
                finishReason: summary.finishReason))
    }

    /// The stop set upstream builds for its own loop, rebuilt here because the
    /// function that does it is private. Same three sources, same order.
    private static func stopTokenIds(
        configuration: ModelConfiguration, tokenizer: MLXLMCommon.Tokenizer
    ) -> Set<Int> {
        var ids = configuration.eosTokenIds
        if let eos = tokenizer.eosTokenId {
            ids.insert(eos)
        }
        for token in configuration.extraEOSTokens {
            if let id = tokenizer.convertTokenToId(token) {
                ids.insert(id)
            }
        }
        return ids
    }

    /// The original path, for the requests the prefix cache has to sit out.
    /// Upstream owns the loop here, so nothing about stop handling is ours.
    private func generateUncached(
        input: sending LMInput,
        parameters: GenerateParameters,
        promptTokenCount: Int,
        emit: @Sendable (Event) async -> Void
    ) async throws {
        var completionTokens = 0
        var finishReason = "stop"
        var emittedToolCalls = 0
        var transcript = ""

        let stream = try await container.generate(
            input: input, parameters: parameters)
        for await event in stream {
            switch event {
            case .chunk(let text):
                transcript += text
                await emit(.chunk(text))
            case .toolCall(let call):
                finishReason = "tool_calls"
                emittedToolCalls += 1
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
                    generateTime: info.generateTime,
                    contextTokens: info.promptTokenCount
                        + info.generationTokenCount)
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

        if emittedToolCalls == 0, !transcript.isEmpty {
            for call in Self.recoverToolCalls(from: transcript) {
                finishReason = "tool_calls"
                await emit(.toolCall(name: call.name, arguments: call.arguments))
            }
        }

        await emit(
            .done(
                promptTokens: promptTokenCount,
                completionTokens: completionTokens,
                finishReason: finishReason))
    }

    /// Pull `<tool_call>{"name":…,"arguments":{…}}</tool_call>` blocks out of
    /// generated text.
    ///
    /// Deliberately narrow: only the tagged JSON form, and only when the run
    /// produced no parsed calls at all. Anything looser would start inventing
    /// tool calls out of a model that was merely talking about one.
    static func recoverToolCalls(from text: String) -> [(name: String, arguments: String)] {
        var found: [(name: String, arguments: String)] = []
        var rest = Substring(text)
        while let close = rest.range(of: "</tool_call>") {
            // The *last* opening tag before this close: the template emits one
            // and the model emits another, so the observed output had it twice
            // and taking the first left an unparseable body.
            let head = rest[rest.startIndex..<close.lowerBound]
            guard let open = head.range(of: "<tool_call>", options: .backwards) else {
                rest = rest[close.upperBound...]
                continue
            }
            let body = rest[open.upperBound..<close.lowerBound]
            rest = rest[close.upperBound...]
            guard
                let data = body.trimmingCharacters(in: .whitespacesAndNewlines)
                    .data(using: .utf8),
                let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let name = object["name"] as? String
            else { continue }
            let arguments = object["arguments"] ?? [String: Any]()
            // `withoutEscapingSlashes` for the same reason as the main encoder:
            // a path argument otherwise ships as "src\/main.rs".
            let encoded =
                (try? JSONSerialization.data(
                    withJSONObject: arguments, options: [.withoutEscapingSlashes]))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
            found.append((name, encoded))
        }
        return found
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
    case unsupportedModel(String)

    var description: String {
        switch self {
        case .notLoaded: return "the model is still loading"
        case .badRequest(let m): return m
        case .unsupportedModel(let m): return m
        }
    }
}
