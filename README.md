# sentencepiece-rs

SentencePiece model loading, normalization, encoding, and decoding in pure Rust. Load a `.model` file, tokenize text, and decode tokens back into text.

## Status

This is a runtime-focused Rust port of the original Google SentencePiece C++ implementation.

Working now:

- load serialized SentencePiece `.model` / `.spm` files
- parse the model protobuf directly in Rust
- run model-provided normalization rules
- encode and decode with Unigram, BPE, Word, and Char models
- handle unknown tokens, control tokens, user-defined symbols, Unicode, and byte fallback
- optimized BPE encoding: reusable merge-key buffer, dead-slot marking instead of `Vec::remove`, eliminated per-pair `format!()` allocation
- optimized Unigram encoding: no intermediate `Vec` collection, inlined path updates without cloning `BestPathNode`

Still cooking:

- training new models
- n-best and sampling APIs
- every last obscure compatibility corner from the C++ processor

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

    let pieces = sp.encode("hello rust world")?;
    let ids = sp.encode_to_ids("hello rust world")?;
    let text = sp.decode_ids(&ids)?;

    println!("{pieces:?}");
    println!("{ids:?}");
    println!("{text}");

    Ok(())
}
```

## Decode Pieces

```rust
use sentencepiece_rs::SentencePieceProcessor;

let sp = SentencePieceProcessor::open("tokenizer.model")?;
let text = sp.decode(&["▁hello", "▁world"])?;
assert_eq!(text, "hello world");
# Ok::<(), sentencepiece_rs::Error>(())
```

## Extra Options

```rust
use sentencepiece_rs::SentencePieceProcessor;

let mut sp = SentencePieceProcessor::open("tokenizer.model")?;
sp.set_encode_extra_options("bos:eos")?;

let ids = sp.encode_to_ids("ship it")?;
# Ok::<(), sentencepiece_rs::Error>(())
```

Supported options: `bos`, `eos`, `reverse`, `unk_piece`.

## Performance

The BPE and Unigram encoding paths are optimized for throughput:

- **BPE**: A single reusable `String` builds merge keys in-place (`clear` + `push_str`) instead of calling `format!()` for every adjacent pair on every merge iteration. Dead slots are marked by clearing the piece string rather than calling `Vec::remove`, turning the per-merge cost from O(n) to O(1). On real-world workloads (e.g. TinyStories tokenization) this alone yields roughly a **10× speedup** for BPE-heavy models.

- **Unigram**: `char_indices()` is iterated directly instead of being collected into an intermediate `Vec`. Path updates use `as_ref()` instead of `.clone()` on `BestPathNode`, and the `update_best` helper has been inlined to avoid an extra function call per candidate.

Both paths produce identical output to the previous implementation—all existing tests pass unchanged.

## Compatibility Notes

The crate reads standard SentencePiece model protobufs and uses the embedded normalizer trie, so normal `.model` files should load without drama. The runtime behavior follows the C++ implementation, but this is not a line-by-line rewrite.

The big missing chunk is training. Bring your own existing model for now.

## Documentation

[docs.rs/sentencepiece-rs](https://docs.rs/sentencepiece-rs)

## References

- Original SentencePiece repo: [google/sentencepiece](https://github.com/google/sentencepiece)
- Paper: [arXiv PDF](https://arxiv.org/pdf/1808.06226)

## Test Locally

```bash
cargo fmt --all
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --all-features --no-deps
```

## License and Terms of Use

This crate is licensed under the Apache License, Version 2.0. See `./LICENSE`.

The original SentencePiece source is also licensed under the Apache License, Version 2.0. This crate is a Rust reimplementation that uses the upstream source as the behavioral reference; it is not a C++ binding and does not link against the C++ library.

Apache 2.0 is permissive. In normal-human terms, you can use, modify, distribute, sublicense, and ship derivative work, including commercially, as long as you follow the license terms.

If you redistribute this crate, the upstream source, or a modified version, keep the important bits intact:

- include a copy of the Apache 2.0 license
- preserve copyright, patent, trademark, and attribution notices that apply
- mark modified files when you change Apache-licensed source
- include upstream `NOTICE` content if a distributed upstream package includes one
- do not imply Google or SentencePiece trademark endorsement

The software is provided as-is, without warranty, and the Apache 2.0 patent grant terminates if you sue over patent infringement involving the licensed work. That is the deal. Pretty reasonable, honestly.

This README is a practical summary, not legal advice. The actual license text in `./LICENSE` and the upstream SentencePiece license are the source of truth.
