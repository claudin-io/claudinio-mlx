import Foundation
import Testing

@testable import claudinio_mlx

/// Upstream picks a tool-call parser by scanning the chat template, and a
/// template that merely mentions another dialect's marker wins the wrong
/// match. Qwen3-VL names `[TOOL_CALLS]` 44 times while the model emits
/// `<tool_call>` — the call then arrives as prose and the agent waits forever.
@Suite("Tool call recovery")
struct ToolCallRecoveryTests {

    @Test("a tagged JSON call is recovered")
    func taggedCall() {
        let text = """
            <tool_call>
            {"name": "read_file", "arguments": {"path": "src/main.rs"}}
            </tool_call>
            """
        let calls = ModelHost.recoverToolCalls(from: text)
        #expect(calls.count == 1)
        #expect(calls.first?.name == "read_file")
        #expect(calls.first?.arguments.contains("src/main.rs") == true)
    }

    /// The observed output had the opening tag twice: the template emits one
    /// and the model emits another.
    @Test("a duplicated opening tag does not break the scan")
    func duplicatedTag() {
        let text = """
            <tool_call>
            <tool_call>
            {"name": "read_file", "arguments": {"path": "a.rs"}}
            </tool_call>
            """
        let calls = ModelHost.recoverToolCalls(from: text)
        #expect(calls.count == 1)
        #expect(calls.first?.name == "read_file")
    }

    @Test("several calls in one response are all recovered")
    func multipleCalls() {
        let text = """
            <tool_call>{"name":"a","arguments":{}}</tool_call>
            some words
            <tool_call>{"name":"b","arguments":{"x":1}}</tool_call>
            """
        let calls = ModelHost.recoverToolCalls(from: text)
        #expect(calls.map(\.name) == ["a", "b"])
    }

    /// The recovery must not invent calls out of a model that is merely
    /// talking about one — that would be worse than missing them.
    @Test("prose about tools is not a tool call")
    func proseIsNotACall() {
        #expect(ModelHost.recoverToolCalls(from: "I would use read_file here.").isEmpty)
        #expect(ModelHost.recoverToolCalls(from: "<tool_call>not json</tool_call>").isEmpty)
        #expect(ModelHost.recoverToolCalls(from: "<tool_call>{\"no\":\"name\"}</tool_call>").isEmpty)
        // Unterminated: the model was cut off mid-call.
        #expect(ModelHost.recoverToolCalls(from: "<tool_call>{\"name\":\"a\"}").isEmpty)
    }

    @Test("a call with no arguments still parses")
    func noArguments() {
        let calls = ModelHost.recoverToolCalls(from: "<tool_call>{\"name\":\"list\"}</tool_call>")
        #expect(calls.first?.name == "list")
        #expect(calls.first?.arguments == "{}")
    }
}
