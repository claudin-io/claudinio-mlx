import Foundation
import Testing

@testable import claudinio_mlx

/// The failure this prevents: a checkpoint MLX partially understands loads
/// without error and then emits mojibake, which reads as a broken app rather
/// than an unsupported model.
@Suite("Model compatibility")
struct ModelCompatibilityTests {

    private func write(_ config: String) throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(config.utf8).write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    @Test("a plain causal LM is accepted")
    func causalLM() throws {
        let dir = try write(#"{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}"#)
        #expect(ModelCompatibility.reasonToRefuse(directory: dir) == nil)
    }

    /// The shape of Qwen3.8-27B-MTPLX-Optimized-Speed, which loaded through
    /// the text-only factory and emitted mojibake: its language weights live
    /// under `language_model.*`, so `model.*` found nothing.
    @Test("a checkpoint with a vision config takes the multimodal factory")
    func multimodalByVisionConfig() throws {
        let dir = try write(
            #"{"model_type":"qwen3_5","vision_config":{"depth":27},"architectures":["Qwen3_5ForConditionalGeneration"]}"#
        )
        #expect(ModelCompatibility.isMultimodal(directory: dir))
        #expect(ModelCompatibility.reasonToRefuse(directory: dir) == nil)
    }

    /// An empty vision config is how upstream marks a text-only variant of an
    /// otherwise multimodal family.
    @Test("an empty vision config stays on the text factory")
    func emptyVisionConfig() throws {
        let dir = try write(#"{"model_type":"qwen3_5","vision_config":{}}"#)
        #expect(!ModelCompatibility.isMultimodal(directory: dir))
    }

    @Test("a plain causal LM is not multimodal")
    func causalIsNotMultimodal() throws {
        let dir = try write(#"{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}"#)
        #expect(!ModelCompatibility.isMultimodal(directory: dir))
    }

    /// An unreadable config is the loader's to report — it has a better
    /// message than a guess made here.
    @Test("a missing config is left to the loader")
    func missingConfig() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        #expect(ModelCompatibility.reasonToRefuse(directory: dir) == nil)
    }
}
