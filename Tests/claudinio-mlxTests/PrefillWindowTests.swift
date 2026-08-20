import Foundation
import Testing

@testable import claudinio_mlx

/// The failure these prevent: upstream's `prepare` for this family takes a
/// `windowSize` and discards it, so the prompt reaches the GPU as one command
/// buffer. Past ~25k tokens on an M2 Max macOS kills that buffer as
/// `Impacting Interactivity` and MLX raises it as an uncaught C++ exception,
/// which ends the process mid-answer. Splitting the prefill is what avoids it,
/// and a split that loses or repeats a span answers from the wrong prompt
/// instead of failing.
@Suite("Prefill windows")
struct PrefillWindowTests {

    private func windows(_ range: Range<Int>, _ step: Int) -> [Range<Int>] {
        ModelHost.prefillWindows(range, step: step)
    }

    /// The property that matters more than any individual boundary: the
    /// windows have to be the prompt, in order, once each.
    private func covers(_ range: Range<Int>, _ step: Int) -> Bool {
        let parts = windows(range, step)
        guard !parts.isEmpty else { return range.isEmpty }
        guard parts.allSatisfy({ !$0.isEmpty }) else { return false }
        guard parts.first?.lowerBound == range.lowerBound,
            parts.last?.upperBound == range.upperBound
        else { return false }
        return zip(parts, parts.dropFirst()).allSatisfy { $0.upperBound == $1.lowerBound }
    }

    @Test("a prompt shorter than one window stays whole")
    func shorterThanStep() {
        #expect(windows(0 ..< 300, 4096) == [0 ..< 300])
    }

    @Test("a prompt is split at the window size")
    func splitsAtStep() {
        #expect(windows(0 ..< 9000, 4096) == [0 ..< 4096, 4096 ..< 8192, 8192 ..< 9000])
    }

    /// An exact multiple must not produce a trailing empty window: the last
    /// one carries the token generation starts from.
    @Test("an exact multiple ends on a full window, not an empty one")
    func exactMultiple() {
        #expect(windows(0 ..< 8192, 4096) == [0 ..< 4096, 4096 ..< 8192])
    }

    @Test("a prefill resuming from a pin starts where the pin ended")
    func resumesFromOffset() {
        #expect(windows(6000 ..< 14000, 4096) == [6000 ..< 10096, 10096 ..< 14000])
    }

    @Test("one token is one window")
    func singleToken() {
        #expect(windows(30_000 ..< 30_001, 4096) == [30_000 ..< 30_001])
    }

    @Test("an empty range has nothing to prefill")
    func empty() {
        #expect(windows(0 ..< 0, 4096).isEmpty)
    }

    /// A step of zero would otherwise loop forever building empty windows.
    @Test("a nonsensical step still makes progress")
    func degenerateStep() {
        #expect(windows(0 ..< 3, 0) == [0 ..< 1, 1 ..< 2, 2 ..< 3])
    }

    @Test("the windows are the prompt, in order, once each")
    func coversEverything() {
        for length in [1, 2, 4095, 4096, 4097, 30_000, 46_000] {
            for step in [1, 512, 4096, 8192] {
                #expect(covers(0 ..< length, step), "length \(length), step \(step)")
            }
        }
    }

    /// The window is a measured value, not a taste: 512 cost 50% more time to
    /// first token at 30k tokens, and 8192 lost at 46k.
    @Test("the shipped window is the measured one")
    func shippedWindow() {
        #expect(ModelHost.prefillWindow == 4096)
    }
}
