import Foundation
import MLXLMCommon
import NIOCore
import NIOHTTP1
import NIOPosix

/// The HTTP surface, deliberately the same subset of llama-server that
/// `agent/provider/openai.rs` already speaks, so switching engines needs no
/// change on the Rust side beyond which binary is launched.
///
/// - `GET  /health`              — 200 once the weights are in memory, 503 before
/// - `GET  /v1/models`           — the single served alias
/// - `POST /v1/chat/completions` — streaming (SSE) or not, with tool calls
/// - `GET  /stats`               — for the status bar
///
/// `/health` is open, like llama-server's: it answers a boolean and nothing
/// else. Everything under `/v1` and `/stats` requires the bearer token.
final class HTTPHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    private let host: ModelHost
    private let apiKey: String
    private var head: HTTPRequestHead?
    private var body: ByteBuffer?

    init(host: ModelHost, apiKey: String) {
        self.host = host
        self.apiKey = apiKey
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        switch unwrapInboundIn(data) {
        case .head(let head):
            self.head = head
            self.body = context.channel.allocator.buffer(capacity: 0)
        case .body(var chunk):
            body?.writeBuffer(&chunk)
        case .end:
            guard let head else { return }
            let bytes = body.map { Data($0.readableBytesView) } ?? Data()
            self.head = nil
            self.body = nil
            route(context: context, head: head, body: bytes)
        }
    }

    private func authorized(_ head: HTTPRequestHead) -> Bool {
        // The key is the only thing standing between a loopback port and every
        // other process on the machine — including any page the agent's own
        // browser visits, which can reach 127.0.0.1 from JavaScript.
        guard let header = head.headers.first(name: "Authorization") else { return false }
        return header == "Bearer \(apiKey)"
    }

    private func route(context: ChannelHandlerContext, head: HTTPRequestHead, body: Data) {
        let path = head.uri.split(separator: "?", maxSplits: 1).first.map(String.init) ?? head.uri
        let channel = context.channel
        let host = self.host

        switch (head.method, path) {
        case (.GET, "/health"):
            Task {
                let ready = true  // the process only listens after load() returns
                await Self.sendJSON(
                    channel, status: ready ? .ok : .serviceUnavailable,
                    payload: ["status": ready ? "ok" : "loading"])
            }

        case (.GET, "/v1/models"):
            guard authorized(head) else { return unauthorized(channel) }
            Task {
                await Self.sendEncodable(
                    channel, status: .ok,
                    value: ModelList(data: [.init(id: host.alias)]))
            }

        case (.GET, "/stats"):
            guard authorized(head) else { return unauthorized(channel) }
            Task {
                await Self.sendEncodable(channel, status: .ok, value: await host.stats())
            }

        case (.POST, "/v1/chat/completions"):
            guard authorized(head) else { return unauthorized(channel) }
            Task { await Self.completions(channel: channel, host: host, body: body) }

        default:
            Task {
                await Self.sendError(channel, status: .notFound, message: "no such endpoint")
            }
        }
    }

    private func unauthorized(_ channel: Channel) {
        Task {
            await Self.sendError(
                channel, status: .unauthorized, message: "Invalid API Key")
        }
    }

    // MARK: - Chat completions

    private static func completions(channel: Channel, host: ModelHost, body: Data) async {
        let request: ChatRequest
        do {
            request = try JSONDecoder().decode(ChatRequest.self, from: body)
        } catch {
            await sendError(channel, status: .badRequest, message: "malformed request: \(error)")
            return
        }

        let messages: [Chat.Message]
        do {
            messages = try convert(request.messages)
        } catch {
            await sendError(channel, status: .badRequest, message: String(describing: error))
            return
        }

        let tools: [ToolSpec]? = request.tools?.compactMap { value in
            guard case .object(let object) = value else { return nil }
            return object.mapValues(\.sendable)
        }
        let maxTokens = request.maxTokens ?? 2048
        let id = "chatcmpl-\(UUID().uuidString)"
        let created = Int(Date().timeIntervalSince1970)
        let model = request.model ?? host.alias

        if request.stream == true {
            await streamCompletion(
                channel: channel, host: host, messages: messages, tools: tools,
                maxTokens: maxTokens, request: request, id: id, created: created, model: model)
        } else {
            await bufferedCompletion(
                channel: channel, host: host, messages: messages, tools: tools,
                maxTokens: maxTokens, request: request, id: id, created: created, model: model)
        }
    }

    /// Accumulates a whole response. Used by the classify/one-shot paths, which
    /// do not stream.
    private static func bufferedCompletion(
        channel: Channel, host: ModelHost, messages: sending [Chat.Message],
        tools: [ToolSpec]?, maxTokens: Int, request: ChatRequest,
        id: String, created: Int, model: String
    ) async {
        let collected = Collected()
        do {
            try await host.generate(
                messages: messages, tools: tools, maxTokens: maxTokens,
                temperature: request.temperature, topP: request.topP
            ) { event in
                await collected.add(event)
            }
        } catch {
            await sendError(
                channel, status: .internalServerError, message: String(describing: error))
            return
        }

        let (text, calls, promptTokens, completionTokens, finishReason) = await collected.result()
        let response = ChatResponse(
            id: id, created: created, model: model,
            choices: [
                .init(
                    message: .init(
                        content: text.isEmpty && !calls.isEmpty ? nil : text,
                        toolCalls: calls.isEmpty
                            ? nil
                            : calls.enumerated().map { index, call in
                                .init(
                                    id: "call_\(index)",
                                    function: .init(name: call.name, arguments: call.arguments))
                            }),
                    finishReason: finishReason)
            ],
            usage: .init(
                promptTokens: promptTokens, completionTokens: completionTokens,
                totalTokens: promptTokens + completionTokens))
        await sendEncodable(channel, status: .ok, value: response)
    }

    /// Server-sent events, in the exact shape `openai.rs` already parses.
    private static func streamCompletion(
        channel: Channel, host: ModelHost, messages: sending [Chat.Message],
        tools: [ToolSpec]?, maxTokens: Int, request: ChatRequest,
        id: String, created: Int, model: String
    ) async {
        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "text/event-stream")
        headers.add(name: "Cache-Control", value: "no-cache")
        headers.add(name: "Connection", value: "keep-alive")
        let head = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)
        try? await channel.writeAndFlush(HTTPServerResponsePart.head(head)).get()

        @Sendable func send(_ chunk: ChatChunk) async {
            guard let data = try? JSONEncoder().encode(chunk),
                let json = String(data: data, encoding: .utf8)
            else { return }
            var buffer = channel.allocator.buffer(capacity: json.utf8.count + 8)
            buffer.writeString("data: \(json)\n\n")
            try? await channel.writeAndFlush(HTTPServerResponsePart.body(.byteBuffer(buffer))).get()
        }

        // The role frame first, which is what an OpenAI client expects to open
        // an assistant message.
        await send(
            ChatChunk(
                id: id, created: created, model: model,
                choices: [.init(delta: .init(role: "assistant", content: nil, toolCalls: nil), finishReason: nil)]))

        let toolIndex = Counter()
        do {
            try await host.generate(
                messages: messages, tools: tools, maxTokens: maxTokens,
                temperature: request.temperature, topP: request.topP
            ) { event in
                switch event {
                case .chunk(let text):
                    await send(
                        ChatChunk(
                            id: id, created: created, model: model,
                            choices: [
                                .init(
                                    delta: .init(role: nil, content: text, toolCalls: nil),
                                    finishReason: nil)
                            ]))
                case .toolCall(let name, let arguments):
                    let index = await toolIndex.next()
                    await send(
                        ChatChunk(
                            id: id, created: created, model: model,
                            choices: [
                                .init(
                                    delta: .init(
                                        role: nil, content: nil,
                                        toolCalls: [
                                            .init(
                                                index: index, id: "call_\(index)",
                                                function: .init(name: name, arguments: arguments))
                                        ]),
                                    finishReason: nil)
                            ]))
                case .done(_, _, let finishReason):
                    await send(
                        ChatChunk(
                            id: id, created: created, model: model,
                            choices: [
                                .init(
                                    delta: .init(role: nil, content: nil, toolCalls: nil),
                                    finishReason: finishReason)
                            ]))
                }
            }
        } catch {
            // Mid-stream the head is already sent, so the only honest signal
            // left is the terminator; the error goes to stderr where the
            // supervisor's tail picks it up.
            FileHandle.standardError.write(Data("generation failed: \(error)\n".utf8))
        }

        var done = channel.allocator.buffer(capacity: 16)
        done.writeString("data: [DONE]\n\n")
        try? await channel.writeAndFlush(HTTPServerResponsePart.body(.byteBuffer(done))).get()
        try? await channel.writeAndFlush(HTTPServerResponsePart.end(nil)).get()
        try? await channel.close()
    }

    /// Translate the OpenAI message list into MLX's chat representation.
    private static func convert(_ messages: [ChatRequest.Message]) throws -> [Chat.Message] {
        try messages.map { message in
            let text = message.content?.flattened ?? ""
            switch message.role {
            case "system":
                return .system(text)
            case "user":
                return .user(text)
            case "assistant":
                return .assistant(text)
            case "tool":
                // The result of a call the model asked for. MLX carries these
                // as tool messages so the template can render them in the
                // place the model was trained to look.
                return .tool(text)
            default:
                throw ServerError.badRequest("unknown message role '\(message.role)'")
            }
        }
    }

    // MARK: - Writing

    private static func sendEncodable<T: Encodable>(
        _ channel: Channel, status: HTTPResponseStatus, value: T
    ) async {
        let data = (try? JSONEncoder().encode(value)) ?? Data("{}".utf8)
        await write(channel, status: status, contentType: "application/json", body: data)
    }

    private static func sendJSON(
        _ channel: Channel, status: HTTPResponseStatus, payload: [String: String]
    ) async {
        let data = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
        await write(channel, status: status, contentType: "application/json", body: data)
    }

    private static func sendError(
        _ channel: Channel, status: HTTPResponseStatus, message: String
    ) async {
        let payload = ErrorResponse(
            error: .init(
                message: message, type: "invalid_request_error", code: Int(status.code)))
        await sendEncodable(channel, status: status, value: payload)
    }

    private static func write(
        _ channel: Channel, status: HTTPResponseStatus, contentType: String, body: Data
    ) async {
        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: contentType)
        headers.add(name: "Content-Length", value: String(body.count))
        let head = HTTPResponseHead(version: .http1_1, status: status, headers: headers)
        var buffer = channel.allocator.buffer(capacity: body.count)
        buffer.writeBytes(body)
        try? await channel.writeAndFlush(HTTPServerResponsePart.head(head)).get()
        try? await channel.writeAndFlush(HTTPServerResponsePart.body(.byteBuffer(buffer))).get()
        try? await channel.writeAndFlush(HTTPServerResponsePart.end(nil)).get()
    }
}

/// Accumulates a non-streamed response across the generation callback.
actor Collected {
    private var text = ""
    private var calls: [(name: String, arguments: String)] = []
    private var promptTokens = 0
    private var completionTokens = 0
    private var finishReason = "stop"

    func add(_ event: ModelHost.Event) {
        switch event {
        case .chunk(let chunk):
            text += chunk
        case .toolCall(let name, let arguments):
            calls.append((name, arguments))
        case .done(let prompt, let completion, let reason):
            promptTokens = prompt
            completionTokens = completion
            finishReason = reason
        }
    }

    func result() -> (String, [(name: String, arguments: String)], Int, Int, String) {
        (text, calls, promptTokens, completionTokens, finishReason)
    }
}

actor Counter {
    private var value = 0
    func next() -> Int {
        defer { value += 1 }
        return value
    }
}

/// Bind and serve until killed.
func runServer(host: ModelHost, address: String, port: Int, apiKey: String) async throws {
    let group = MultiThreadedEventLoopGroup(numberOfThreads: 2)
    let bootstrap = ServerBootstrap(group: group)
        .serverChannelOption(ChannelOptions.backlog, value: 64)
        .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
        .childChannelInitializer { channel in
            channel.pipeline.configureHTTPServerPipeline().flatMap {
                channel.pipeline.addHandler(HTTPHandler(host: host, apiKey: apiKey))
            }
        }
        .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)

    let channel = try await bootstrap.bind(host: address, port: port).get()
    FileHandle.standardError.write(
        Data("listening on http://\(address):\(port)\n".utf8))
    try await channel.closeFuture.get()
}
