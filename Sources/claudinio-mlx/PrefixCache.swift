import Foundation
import MLX
import MLXLMCommon

/// A snapshot of the KV cache at a prefix the next request is likely to share,
/// kept so a follow-up does not re-run prefill over the whole conversation.
///
/// This buys more here than it would on a server with a batched prefill. The
/// hybrid Qwen3.5 family is mostly gated-delta (linear attention) — 48 of the
/// 27B's 64 layers — and that kernel walks the sequence one position at a time
/// whether it is prefilling or decoding. Prefill therefore costs roughly what
/// generation costs per token: measured on an M2 Max, 84 tok/s prefill against
/// 18 tok/s decode, flat from 2k to 8k tokens.
///
/// ### Why a pin and not an append
///
/// The obvious scheme — keep the cache a request left behind, append the next
/// prompt's new tail — does not survive a reasoning model. Qwen3.5's template
/// drops prior-turn reasoning and re-renders that turn with an *empty* think
/// block, so turn N+1 diverges from turn N's prompt one token past `<think>`:
///
///     held:   … 248068, 198, 760, …            <think> ␤ …
///     prompt: … 248068, 271, 248069, …         <think> ␤␤ </think> …
///
/// Reuse therefore has to rewind, and gated-delta state is a recurrence with
/// no rewind — `MambaCache` reports `isTrimmable == false`. So instead of
/// rewinding a cache we keep a *copy* taken at a boundary two consecutive
/// prompts agreed on, and copy forward from it.
///
/// `@unchecked Sendable` because `KVCache` holds `MLXArray`. Every access, and
/// the whole generation that mutates the borrowed cache, happens while the
/// caller holds ``GenerationGate`` — that, not this type, is what serialises.
final class PrefixCache: @unchecked Sendable {
    /// Below this a pin costs more in copying than it saves in prefill.
    private static let minimumPin = 256

    enum Plan {
        /// Nothing reusable: prefill the whole prompt into a fresh cache.
        case cold
        /// Start from this copy of the pin, which already holds `prefix`
        /// tokens of the prompt.
        case fromPin(cache: [KVCache], prefix: Int)
    }

    private var pinnedTokens: [Int] = []
    private var pinnedCache: [KVCache] = []
    /// The previous request's prompt. What two consecutive prompts agree on is
    /// the only evidence available for where the stable head ends.
    private var lastPrompt: [Int] = []

    func plan(for prompt: [Int]) -> Plan {
        guard !pinnedTokens.isEmpty,
            Self.position(of: pinnedCache) == pinnedTokens.count,
            pinnedTokens.count < prompt.count,
            prompt.starts(with: pinnedTokens),
            let copy = Self.deepCopy(pinnedCache)
        else {
            return .cold
        }
        return .fromPin(cache: copy, prefix: pinnedTokens.count)
    }

    /// Where to snapshot this request's cache, or nil for "do not bother".
    ///
    /// The boundary is what this prompt and the previous one still agree on.
    /// In an agent loop that lands just short of the previous prompt's end —
    /// everything except the think block being rewritten — and it grows every
    /// turn, so the pin chases the stable head instead of guessing at it.
    /// Note this does not reproduce an uncached run bit for bit. Resuming
    /// splits the prefill in two, and the two halves chunk and accumulate
    /// differently from one unbroken pass, which moves the last bits of the
    /// logits; at a near-tie between two words that is enough to pick the
    /// other one. Rounding the boundary to the prefill step was tried and does
    /// not close the gap — it only costs a re-read. This is the same trade
    /// every prefix cache makes, and the answers agree, not the bytes.
    func pinLength(for prompt: [Int], startingAt base: Int) -> Int? {
        guard !lastPrompt.isEmpty else { return nil }
        let stable = Self.commonPrefix(prompt, lastPrompt)
        guard stable > base, stable >= Self.minimumPin, stable < prompt.count else {
            return nil
        }
        return stable
    }

    /// Adopt a copy of `cache`, which must hold exactly `tokens`.
    func pin(tokens: [Int], cache: [KVCache]) {
        guard Self.position(of: cache) == tokens.count,
            let copy = Self.deepCopy(cache)
        else {
            drop()
            return
        }
        // Materialise it. The copy is a slice of the working cache's arrays,
        // and left lazy it would pin that whole graph alive instead of the
        // handful of buffers actually wanted.
        eval(copy.flatMap { $0.innerState() })
        pinnedTokens = tokens
        pinnedCache = copy
    }

    func notePrompt(_ prompt: [Int]) {
        lastPrompt = prompt
    }

    func drop() {
        pinnedTokens = []
        pinnedCache = []
    }

    /// The sequence position a cache has reached.
    ///
    /// Deliberately not `cache[0].offset`. In a hybrid model layer 0 is often
    /// linear attention, and those entries hold a fixed-size recurrent state
    /// with no position in it — `ArraysCache` never touches `offset`, so it
    /// reads 0 no matter how much has gone through. The full-attention layers
    /// do track it, so the furthest-advanced entry is the one telling the
    /// truth. A model with no such layer reports 0 and is never pinned.
    private static func position(of cache: [KVCache]) -> Int {
        cache.map(\.offset).max() ?? 0
    }

    private static func commonPrefix(_ a: [Int], _ b: [Int]) -> Int {
        var i = 0
        let limit = min(a.count, b.count)
        while i < limit, a[i] == b[i] { i += 1 }
        return i
    }

    /// Copy a cache so the original can be generated into without touching it.
    ///
    /// `copy()` is upstream's, and is right for everything except the
    /// recurrent entries: `ArraysCache.copyContents(to:)` is an empty function,
    /// so copying a gated-delta or Mamba entry silently returns a blank state
    /// rather than failing. Its `state` is public, so put it back by hand.
    ///
    /// Returns nil for a shape this cannot vouch for — a `CacheList` hides its
    /// children behind a flattened `state`, and a nested recurrent entry would
    /// come back blank with nothing here noticing.
    private static func deepCopy(_ cache: [KVCache]) -> [KVCache]? {
        var copied: [KVCache] = []
        copied.reserveCapacity(cache.count)
        for entry in cache {
            if entry is CacheList { return nil }
            let duplicate = entry.copy()
            if let source = entry as? ArraysCache,
                let target = duplicate as? ArraysCache
            {
                target.state = source.state
                target.offset = source.offset
            }
            copied.append(duplicate)
        }
        return copied
    }
}

/// Serialises generation.
///
/// Sharing one KV cache across concurrent requests is not merely contention:
/// the two would interleave writes into the same recurrence and produce
/// garbage for both. `Server` answers every request in its own `Task`, so
/// without this they would overlap. Requests queue instead.
actor GenerationGate {
    private var busy = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        guard busy else {
            busy = true
            return
        }
        await withCheckedContinuation { waiting.append($0) }
    }

    func release() {
        if waiting.isEmpty {
            busy = false
        } else {
            waiting.removeFirst().resume()
        }
    }
}
