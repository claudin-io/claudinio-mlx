import Foundation
import Testing

@testable import claudinio_mlx

/// The request shapes `agent/provider/openai.rs` actually sends. Each of these
/// is a wire contract with the Rust side, not an internal detail.
@Suite("Chat API decoding")
struct ChatRequestTests {

    @Test("content decodes from a plain string")
    func plainStringContent() throws {
        let json = """
            {"messages":[{"role":"user","content":"hello"}]}
            """
        let request = try JSONDecoder().decode(ChatRequest.self, from: Data(json.utf8))
        #expect(request.messages.first?.content?.flattened == "hello")
    }

    /// Multimodal-shaped content: the app sends this whenever a message
    /// carries an image alongside text.
    @Test("content decodes from an array of parts")
    func partsContent() throws {
        let json = """
            {"messages":[{"role":"user","content":[
              {"type":"text","text":"first"},
              {"type":"text","text":"second"}
            ]}]}
            """
        let request = try JSONDecoder().decode(ChatRequest.self, from: Data(json.utf8))
        #expect(request.messages.first?.content?.flattened == "first\nsecond")
    }

    @Test("snake_case fields are picked up")
    func snakeCase() throws {
        let json = """
            {"messages":[],"max_tokens":512,"top_p":0.9,"stream":true}
            """
        let request = try JSONDecoder().decode(ChatRequest.self, from: Data(json.utf8))
        #expect(request.maxTokens == 512)
        #expect(request.topP == 0.9)
        #expect(request.stream == true)
    }

    @Test("tool definitions survive the round trip without a schema")
    func toolsArePreserved() throws {
        let json = """
            {"messages":[],"tools":[{"type":"function","function":{
              "name":"read_file","parameters":{"type":"object",
              "properties":{"path":{"type":"string"}},"required":["path"]}}}]}
            """
        let request = try JSONDecoder().decode(ChatRequest.self, from: Data(json.utf8))
        guard case .object(let tool)? = request.tools?.first,
            case .object(let function)? = tool["function"],
            case .string(let name)? = function["name"]
        else {
            Issue.record("tool definition did not survive decoding")
            return
        }
        #expect(name == "read_file")

        // The parser is handed a Foundation object graph; anything else and it
        // silently authorizes nothing.
        let sendable = tool.mapValues(\.sendable)
        #expect(JSONSerialization.isValidJSONObject(sendable))
    }

    @Test("a streamed chunk serializes into the shape openai.rs parses")
    func chunkShape() throws {
        let chunk = ChatChunk(
            id: "chatcmpl-1", created: 0, model: "m",
            choices: [
                .init(
                    delta: .init(
                        role: nil, content: nil,
                        toolCalls: [
                            .init(
                                index: 0, id: "call_0",
                                function: .init(name: "read_file", arguments: #"{"path":"a.rs"}"#))
                        ]),
                    finishReason: nil)
            ])
        let data = try JSONEncoder().encode(chunk)
        let text = String(data: data, encoding: .utf8) ?? ""
        #expect(text.contains("\"object\":\"chat.completion.chunk\""))
        #expect(text.contains("\"tool_calls\""))
        // Arguments are a JSON *string*, not a nested object.
        #expect(text.contains("\"arguments\":\"{"))
    }

    @Test("finish_reason is emitted with the underscore the API specifies")
    func finishReasonKey() throws {
        let chunk = ChatChunk(
            id: "1", created: 0, model: "m",
            choices: [
                .init(delta: .init(role: nil, content: nil, toolCalls: nil), finishReason: "stop")
            ])
        let text = String(data: try JSONEncoder().encode(chunk), encoding: .utf8) ?? ""
        #expect(text.contains("\"finish_reason\":\"stop\""))
    }
}
