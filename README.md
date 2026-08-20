# claudinio-mlx

The MLX inference engine for [Claudinio Code](https://github.com/claudin-io/claudinio-code),
built as a standalone sidecar for Apple Silicon.

It speaks the same OpenAI-compatible subset that `llama-server` does, so the app
talks to either engine through one client:

| Endpoint | Auth | Notes |
|---|---|---|
| `GET /health` | open | 200 once the weights are loaded |
| `GET /v1/models` | bearer | the single served alias |
| `POST /v1/chat/completions` | bearer | streaming (SSE) or buffered, with tool calls |
| `GET /stats` | bearer | tokens/s, context used, for the status bar |

```
claudinio-mlx serve --model <dir> --port 8080 --api-key <key> [--alias NAME] [--ctx-size N]
claudinio-mlx bench --model <dir> [--prompt TEXT]
```

## Why this exists instead of `mlx_lm.server`

Apple's [mlx-lm](https://github.com/ml-explore/mlx-lm) ships a Python server that
does the same job. This one exists because of three things it cannot do:

- **No Python.** A single binary, so the app never has to provision an
  interpreter on a user's machine.
- **Authentication.** `mlx_lm.server` has no api-key option — its own docs say
  it "is not recommended for production as it only implements basic security
  checks". An unauthenticated inference endpoint on loopback is reachable by
  every process on the machine, and by any web page the user's browser visits.
  `--api-key` here is required, not optional.
- **Weights stay off the network path.** Models are loaded from a directory.
  The app downloads and verifies them against the Hub's own sha256 before this
  process ever sees them.

Tool-call parsing is *not* reimplemented: it comes from
[`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm)'s per-family
parsers (Qwen, Llama 3, Mistral, GLM4, Kimi K2, Gemma, Harmony…), which is
precisely the part that fails silently when rewritten.

## Building

`swift build` compiles, but the resulting binary **cannot run**: SwiftPM on the
command line does not compile MLX's Metal shaders, and the process dies with
"Failed to load the default metallib". Use `xcodebuild`, which does:

```sh
xcodebuild -downloadComponent MetalToolchain   # once, ~690 MB on Xcode 26
xcodebuild build -scheme claudinio-mlx -destination 'platform=OS X' \
  -configuration Release -derivedDataPath .xcbuild \
  -skipPackagePluginValidation -skipMacroValidation
```

The binary and its `mlx-swift_Cmlx.bundle` (which carries `default.metallib`)
both live in `.xcbuild/Build/Products/Release/` and must ship together.

`swift test` runs the unit tests and needs none of the above.

## License

MIT. Depends on `mlx-swift`, `mlx-swift-lm` (both MIT, Apple) and
`swift-transformers` (Apache-2.0).
