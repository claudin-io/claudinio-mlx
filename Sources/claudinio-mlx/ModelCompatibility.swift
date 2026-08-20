import Foundation

/// Whether this build can load the model in `directory`, and why not.
///
/// MLX loads what it recognizes and ignores what it does not, so an
/// unsupported checkpoint does not fail — it produces a model whose weights
/// are misread, and the first thing the user sees is a wall of mojibake.
/// Failing loudly here is the whole point: a refusal names the problem, a
/// silent misload wastes an hour.
enum ModelCompatibility {
    /// Whether the checkpoint keeps its language weights under
    /// `language_model.*` and needs the multimodal factory.
    ///
    /// Decided by a non-empty `vision_config`, which is what upstream's own
    /// registration predicate looks at.
    static func isMultimodal(directory: URL) -> Bool {
        guard let config = readConfig(directory: directory) else { return false }
        if let vision = config["vision_config"] as? [String: Any], !vision.isEmpty {
            return true
        }
        if let architectures = config["architectures"] as? [String],
            let first = architectures.first
        {
            return first.hasSuffix("ForConditionalGeneration")
        }
        return false
    }

    private static func readConfig(directory: URL) -> [String: Any]? {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    static func reasonToRefuse(directory: URL) -> String? {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
            let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            // A missing or unreadable config is the loader's problem to report;
            // it has a better message for it than this does.
            return nil
        }

        // Multimodal checkpoints are loaded through the VLM factory instead —
        // see `isMultimodal`. Nothing to refuse here today; the check stays
        // because "loads and emits garbage" is this engine's failure mode and
        // it is worth a named place to add the next one.
        _ = config
        return nil
    }
}
