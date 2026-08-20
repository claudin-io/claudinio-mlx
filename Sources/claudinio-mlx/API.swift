import Foundation

/// The subset of the OpenAI chat API the agent actually sends, mirrored from
/// what `agent/provider/openai.rs` already produces for llama.cpp. Anything the
/// app never sends is deliberately absent rather than accepted and ignored.
struct ChatRequest: Decodable {
    struct Message: Decodable {
        let role: String
        let content: MessageContent?
        let toolCallId: String?
        let toolCalls: [RequestToolCall]?

        enum CodingKeys: String, CodingKey {
            case role, content
            case toolCallId = "tool_call_id"
            case toolCalls = "tool_calls"
        }
    }

    /// Content arrives either as a string or as an array of typed parts.
    enum MessageContent: Decodable {
        case text(String)
        case parts([Part])

        struct Part: Decodable {
            let type: String
            let text: String?
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let s = try? container.decode(String.self) {
                self = .text(s)
            } else {
                self = .parts(try container.decode([Part].self))
            }
        }

        var flattened: String {
            switch self {
            case .text(let s): return s
            case .parts(let parts): return parts.compactMap(\.text).joined(separator: "\n")
            }
        }
    }

    struct RequestToolCall: Decodable {
        struct Function: Decodable {
            let name: String
            let arguments: String
        }
        let id: String?
        let function: Function
    }

    let model: String?
    let messages: [Message]
    let stream: Bool?
    let maxTokens: Int?
    let temperature: Float?
    let topP: Float?
    /// Passed straight to the tool-call parser for authorization: a supplied
    /// list means only those names may be called.
    let tools: [AnyJSON]?

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature, tools
        case maxTokens = "max_tokens"
        case topP = "top_p"
    }
}

/// A JSON value that survives a round trip without a schema. Tool definitions
/// are passed through to the parser as-is, so they must not be lossy.
///
/// Named to stay out of the way of `MLXLMCommon.JSONValue`, which is a
/// different type with the same job: shadowing it made a `[String: JSONValue]`
/// silently fail `JSONSerialization.isValidJSONObject` and every tool call went
/// out with empty arguments.
enum AnyJSON: Codable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([AnyJSON])
    case object([String: AnyJSON])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
        } else if let v = try? c.decode(Bool.self) {
            self = .bool(v)
        } else if let v = try? c.decode(Double.self) {
            self = .number(v)
        } else if let v = try? c.decode(String.self) {
            self = .string(v)
        } else if let v = try? c.decode([AnyJSON].self) {
            self = .array(v)
        } else {
            self = .object(try c.decode([String: AnyJSON].self))
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let v): try c.encode(v)
        case .number(let v): try c.encode(v)
        case .string(let v): try c.encode(v)
        case .array(let v): try c.encode(v)
        case .object(let v): try c.encode(v)
        }
    }

    /// The `[String: any Sendable]` shape MLX's tool-call parser expects.
    var sendable: any Sendable {
        switch self {
        case .null: return NSNull()
        case .bool(let v): return v
        case .number(let v): return v
        case .string(let v): return v
        case .array(let v): return v.map(\.sendable)
        case .object(let v): return v.mapValues(\.sendable)
        }
    }
}

struct ChatResponse: Encodable {
    struct ToolCall: Encodable {
        struct Function: Encodable {
            let name: String
            let arguments: String
        }
        let id: String
        let type = "function"
        let function: Function
    }

    struct Message: Encodable {
        let role = "assistant"
        let content: String?
        let toolCalls: [ToolCall]?

        enum CodingKeys: String, CodingKey {
            case role, content
            case toolCalls = "tool_calls"
        }
    }

    struct Choice: Encodable {
        let index = 0
        let message: Message
        let finishReason: String

        enum CodingKeys: String, CodingKey {
            case index, message
            case finishReason = "finish_reason"
        }
    }

    struct Usage: Encodable {
        let promptTokens: Int
        let completionTokens: Int
        let totalTokens: Int

        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
        }
    }

    let id: String
    let object = "chat.completion"
    let created: Int
    let model: String
    let choices: [Choice]
    let usage: Usage
}

/// One SSE frame of a streamed completion.
struct ChatChunk: Encodable {
    struct Delta: Encodable {
        let role: String?
        let content: String?
        let toolCalls: [StreamToolCall]?

        enum CodingKeys: String, CodingKey {
            case role, content
            case toolCalls = "tool_calls"
        }
    }

    /// Streamed tool calls carry an index so a client can assemble them; we
    /// always emit a complete call in one frame, because MLX's parser only
    /// surfaces a call once it has fully parsed.
    struct StreamToolCall: Encodable {
        struct Function: Encodable {
            let name: String
            let arguments: String
        }
        let index: Int
        let id: String
        let type = "function"
        let function: Function
    }

    struct Choice: Encodable {
        let index = 0
        let delta: Delta
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index, delta
            case finishReason = "finish_reason"
        }
    }

    let id: String
    let object = "chat.completion.chunk"
    let created: Int
    let model: String
    let choices: [Choice]
}

struct ModelList: Encodable {
    struct Entry: Encodable {
        let id: String
        let object = "model"
        let ownedBy = "claudinio"

        enum CodingKeys: String, CodingKey {
            case id, object
            case ownedBy = "owned_by"
        }
    }
    let object = "list"
    let data: [Entry]
}

/// Fed to the status bar. Deliberately our own shape rather than Prometheus
/// text: there is one consumer, and it is ours.
struct Stats: Encodable {
    let modelKey: String
    /// Metal buffer bytes, not RSS — see `ModelHost.stats()`.
    var memoryBytes: Int = 0
    let ctxSize: Int
    let ctxUsed: Int
    let tokensPerSecond: Double
    let promptTokensPerSecond: Double
    let tokensGenerated: Int
    let busy: Bool
}

struct ErrorResponse: Encodable {
    struct Body: Encodable {
        let message: String
        let type: String
        let code: Int
    }
    let error: Body
}
