# PLAN.md — Port SentencePiece C++ to Rust

## Mission

Port the original C++ SentencePiece implementation into a clean, idiomatic Rust crate named `sentencepiece-rs`.

This is **not** a thin wrapper around the C++ code. The goal is to re-implement the core logic in Rust while using the C++ source as the behavioral reference.

Codex should treat these files as the main source of truth:

- `./.agents/SENTENCEPIECE_SKILL.md` — local notes about the C++ codebase
- https://github.com/google/sentencepiece — original upstream C++ implementation
- `./.paper/sentencepiece.pdf` — local paper copy when algorithm/math details are unclear

The upstream C++ repository is **not vendored** in this project anymore. If exact C++ behavior must be inspected, use the upstream GitHub repository or a temporary clone outside this crate. Do not add the full upstream repository back into the crate.

## Core Goals

1. Recreate SentencePiece behavior in Rust.
2. Preserve compatibility with existing SentencePiece model files where practical.
3. Keep the Rust code readable, testable, and crate-ready.
4. Add tests for important behavior and edge cases.
5. Run quality checks with `cargo fmt`, `cargo test`, and `cargo clippy`.
6. Write a Gen-Z-style but still clear and useful `README.md` for crates.io.

## Non-Goals

Do **not** blindly transliterate C++ line-by-line.

Avoid:

- unsafe Rust unless absolutely necessary
- direct C++ bindings
- over-engineered abstractions
- hidden global mutable state
- copying C++ naming conventions when Rust naming is clearer

Prefer:

- idiomatic Rust modules
- clear ownership and borrowing
- small focused structs
- explicit error types
- testable pure functions

## Suggested Rust Crate Structure

```text
.
├── Cargo.toml
├── README.md
├── PLAN.md
├── src
│   ├── lib.rs
│   ├── error.rs
│   ├── model.rs
│   ├── normalizer.rs
│   ├── trainer.rs
│   ├── processor.rs
│   ├── vocab.rs
│   ├── sentencepiece_model.rs
│   └── util.rs
├── tests
│   ├── processor_tests.rs
│   ├── normalizer_tests.rs
│   ├── model_loading_tests.rs
│   └── compatibility_tests.rs
└── benches
    └── processor_bench.rs
```

This structure can change if the C++ code suggests a cleaner split.

## Implementation Plan

### Phase 1 — Read and Map the C++ Code

Start by reading:

1. `./.agents/SENTENCEPIECE_SKILL.md`
2. key files in the upstream C++ repository

Understand and document the responsibilities of the original components before writing Rust.

Pay special attention to:

- tokenizer / processor flow
- model loading
- protobuf model format
- normalization
- vocabulary handling
- Unigram model
- BPE model
- error handling
- training flow, if included in the initial scope

Create or update internal notes while working if needed.

### Phase 2 — Define the Public Rust API

Design an API that feels natural for Rust users.

Example target shape:

```rust
use sentencepiece_rs::SentencePieceProcessor;

let processor = SentencePieceProcessor::open("model.spm")?;
let pieces = processor.encode("hello world")?;
let text = processor.decode(&pieces)?;
```

Potential public types:

```rust
pub struct SentencePieceProcessor;
pub struct SentencePieceModel;
pub struct EncodeOptions;
pub struct DecodeOptions;

pub enum Error;
pub type Result<T> = std::result::Result<T, Error>;
```

The public API should be simple enough that a new user can understand it from the README in under one minute.

### Phase 3 — Model File Support

Implement loading of SentencePiece model files.

Tasks:

- inspect the original C++ model serialization path
- identify protobuf schema usage
- decide whether to use `prost`, `protobuf`, or generated Rust types
- parse `.model` / `.spm` files
- expose model metadata
- load vocabulary pieces
- preserve scores, piece types, and special tokens

Important behavior to preserve:

- unknown token handling
- BOS / EOS / PAD token IDs
- control tokens
- user-defined symbols
- byte fallback if supported
- normalization rules stored in the model

### Phase 4 — Normalization

Port the normalization pipeline.

Tasks:

- identify C++ normalizer behavior
- implement Unicode-aware normalization in Rust
- support model-provided normalization specs
- preserve whitespace handling
- preserve dummy prefix behavior
- preserve escape / meta symbol behavior where applicable

Add tests for:

- whitespace
- repeated spaces
- leading / trailing spaces
- Unicode text
- mixed ASCII + Unicode text
- empty string
- normalization edge cases from C++ tests

### Phase 5 — Encoding

Implement encoding behavior.

Support at minimum:

- text to pieces
- text to IDs
- handling of unknown tokens
- model-specific segmentation

Port model algorithms in priority order:

1. Unigram
2. BPE
3. Word / Char models if present and feasible

For Unigram:

- port lattice construction
- port Viterbi / best path logic
- preserve score behavior
- handle unknown fallback

For BPE:

- port pair merge logic
- preserve merge priority behavior
- handle vocabulary lookup correctly

Add compatibility tests against known model fixtures where available. Keep any committed fixtures under `tests/fixtures/` and small enough for crates.io packaging.

### Phase 6 — Decoding

Implement decoding from:

- pieces to text
- IDs to text

Preserve behavior for:

- meta whitespace symbol
- special tokens
- control tokens
- unknown pieces
- byte fallback
- empty input

### Phase 7 — Training Support

Training may be large. If full training is too much for the first pass, split it behind a feature flag or document it as incomplete.

Suggested approach:

```toml
[features]
default = ["processor"]
processor = []
trainer = []
```

If implementing training:

- port corpus loading
- seed sentence pieces
- EM training loop
- pruning
- vocab size logic
- character coverage
- model export

If not implementing training yet:

- expose clear `unimplemented!()` only in non-public or feature-gated paths
- document missing support in README
- create TODO issues in comments

### Phase 8 — Error Handling

Create a crate-level error type.

Recommended:

```rust
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("model parse error: {0}")]
    ModelParse(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("unsupported feature: {0}")]
    Unsupported(String),
}
```

Avoid panics in library code unless the condition is truly impossible.

### Phase 9 — Testing

Use the original C++ tests and fixtures as behavioral references, but do not depend on a checked-in `./sentencepiece` directory.

Minimum test coverage:

- model loading
- vocab lookup
- special token IDs
- normalization
- encode to pieces
- encode to IDs
- decode from pieces
- decode from IDs
- unknown token behavior
- empty input
- Unicode input
- compatibility with small known `.model` files

Recommended commands:

```bash
cargo fmt --all
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

If tests require fixture files, place them under:

```text
tests/fixtures/
```

Do not depend on machine-specific absolute paths.

### Phase 10 — Quality Bar

Before considering the port complete, run:

```bash
cargo fmt --all
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

All must pass.

Also check:

```bash
cargo doc --all-features --no-deps
```

Fix public API docs where needed.

## README.md Requirements

Write `README.md` in a Gen-Z style, but keep it useful and professional enough for crates.io.

Tone target:

- casual
- fast to read
- clear examples
- no corporate yapping
- no meme overload
- beginner-friendly

README should include:

1. What the crate does
2. Current status
3. Installation
4. Quick start
5. Encoding examples
6. Decoding examples
7. Model loading examples
8. Feature flags, if any
9. Compatibility notes
10. Limitations
11. Testing instructions
12. License note

Example vibe:

```md
# sentencepiece-rs

SentencePiece in Rust. No C++ wrapper. No weird setup. Just load a model and tokenize text.

## Install

```toml
[dependencies]
sentencepiece-rs = "0.1"
```

## Quick Start

```rust
use sentencepiece_rs::SentencePieceProcessor;

fn main() -> sentencepiece_rs::Result<()> {
    let sp = SentencePieceProcessor::open("tokenizer.model")?;

    let ids = sp.encode_to_ids("hello rust world")?;
    let text = sp.decode_ids(&ids)?;

    println!("{ids:?}");
    println!("{text}");

    Ok(())
}
```

## Status

This crate is a Rust reimplementation of SentencePiece based on the original C++ source.

Currently supported:

- model loading
- encoding
- decoding

Still cooking:

- full training support
- advanced compatibility edge cases
```

Keep it fun, but make sure users can actually use the crate immediately.

## Compatibility Strategy

Whenever possible, compare Rust output against the original C++ SentencePiece behavior.

Suggested compatibility workflow:

1. Use the same `.model` file.
2. Feed the same input strings into C++ SentencePiece and Rust implementation.
3. Compare:
   - pieces
   - IDs
   - decoded text
4. Add failing cases as tests.
5. Fix Rust behavior until it matches.

Compatibility test examples should include:

```text
hello world
Hello, world!
こんにちは世界
สวัสดีครับ
hello   world
 leading space
trailing space 
emoji 🚀 test
unknown_token_zzzz
```

## Coding Style

Use idiomatic Rust.

Prefer:

```rust
pub fn encode(&self, input: &str) -> Result<Vec<String>>
```

over C++-style APIs.

Use:

- `thiserror` for errors
- `prost` or another reasonable protobuf crate if needed
- `unicode-normalization` if useful
- `criterion` for benchmarks if benchmarks are added

Avoid:

- unnecessary macros
- massive god objects
- panic-heavy code
- unsafe blocks
- cloning huge data structures without reason

## Definition of Done

The task is complete when:

- Rust implementation is created from the C++ reference
- core processor API works
- model files can be loaded
- encode/decode behavior is tested
- compatibility tests exist
- `cargo fmt --all` passes
- `cargo test --all-features` passes
- `cargo clippy --all-targets --all-features -- -D warnings` passes
- `cargo doc --all-features --no-deps` passes
- `README.md` exists and is crates.io-ready
- public API examples in README compile or are clearly marked as pseudo-code

## Final Notes for Codex

Read before coding:

```bash
cat ./.agents/SENTENCEPIECE_SKILL.md
```

Then inspect the C++ source from https://github.com/google/sentencepiece or a temporary clone outside the crate.

Use the C++ code as the reference, but write Rust like Rust.

Ship small, tested pieces. Do not try to port the entire project in one giant commit.
