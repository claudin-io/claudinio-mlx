import Foundation
import Testing

@testable import claudinio_mlx

/// The boundary logic, exercised without a model. Where the pin lands decides
/// whether the next request re-reads eight thousand tokens or eighty, and it
/// is the one part of the cache that is pure arithmetic over token ids.
@Suite("Prefix cache boundaries")
struct PrefixCacheTests {

    private func tokens(_ range: Range<Int>) -> [Int] { Array(range) }

    @Test("nothing is pinned until two prompts have been seen")
    func noPinWithoutHistory() {
        let cache = PrefixCache()
        #expect(cache.pinLength(for: tokens(0 ..< 4096), startingAt: 0) == nil)
    }

    @Test("the pin lands where two consecutive prompts stop agreeing")
    func pinAtDivergence() {
        let cache = PrefixCache()
        cache.notePrompt(tokens(0 ..< 4096))
        // Same head, different tail — the shape of a turn being re-rendered.
        var next = tokens(0 ..< 4000)
        next.append(contentsOf: (9000 ..< 9100))
        #expect(cache.pinLength(for: next, startingAt: 0) == 4000)
    }

    @Test("a prompt that merely extends the last one pins just short of it")
    func pinOnPureExtension() {
        let cache = PrefixCache()
        cache.notePrompt(tokens(0 ..< 4096))
        #expect(cache.pinLength(for: tokens(0 ..< 5000), startingAt: 0) == 4096)
    }

    @Test("an identical prompt leaves nothing to feed, so it is not pinned")
    func noPinWhenNothingWouldRemain() {
        let cache = PrefixCache()
        cache.notePrompt(tokens(0 ..< 4096))
        #expect(cache.pinLength(for: tokens(0 ..< 4096), startingAt: 0) == nil)
    }

    @Test("a boundary no further along than the cache already is buys nothing")
    func noPinBehindTheBase() {
        let cache = PrefixCache()
        cache.notePrompt(tokens(0 ..< 4096))
        #expect(cache.pinLength(for: tokens(0 ..< 5000), startingAt: 4096) == nil)
    }

    @Test("a prefix too short to be worth copying is not pinned")
    func noPinBelowTheFloor() {
        let cache = PrefixCache()
        cache.notePrompt(tokens(0 ..< 100))
        var next = tokens(0 ..< 100)
        next.append(contentsOf: (9000 ..< 9100))
        #expect(cache.pinLength(for: next, startingAt: 0) == nil)
    }

    @Test("with no pin held, every prompt starts cold")
    func planIsColdWithoutAPin() {
        let cache = PrefixCache()
        cache.notePrompt(tokens(0 ..< 4096))
        guard case .cold = cache.plan(for: tokens(0 ..< 5000)) else {
            Issue.record("expected a cold start with nothing pinned")
            return
        }
    }
}
