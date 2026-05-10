# Contributing to sentencepiece-rs

Thanks for helping push this crate forward. The mission is simple: SentencePiece runtime behavior in idiomatic Rust, with no C++ bindings and no sneaky native setup tax.

## Project Scope

This crate is a Rust reimplementation of the SentencePiece runtime.

Good contributions:

- model loading compatibility
- normalization correctness
- encode/decode behavior
- Unigram and BPE accuracy
- Unicode and byte fallback fixes
- public API docs and examples
- focused tests against real `.model` fixtures

Out of scope for random drive-by patches:

- C++ FFI bindings
- giant rewrites without compatibility evidence
- training support mixed into unrelated runtime changes
- unsafe code unless there is a brutally clear reason

Training support is welcome eventually, but it is a big subsystem. Split it into a clear design and small patches.

## Setup

You need Rust stable with Cargo.

```bash
cargo check --all-features
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --all-features --no-deps
```

Run `cargo fmt --all` before submitting anything.

## Compatibility Rules

Use the upstream implementation as the behavioral reference:

- Original repo: https://github.com/google/sentencepiece
- Local notes: `./.agents/SENTENCEPIECE_SKILL.md`

The upstream C++ repository is not vendored in this project. When behavior is unclear, inspect the upstream source from GitHub or a temporary local clone outside the crate. If the algorithm still looks cursed, check the paper linked in the README.

Compatibility beats personal taste. If Rust code looks nicer but changes tokenization output, the code is wrong.

## Testing

Every behavior change needs tests.

At minimum, cover the relevant path:

- model loading
- normalization
- encode pieces
- encode ids
- decode pieces
- decode ids
- unknown tokens
- Unicode
- whitespace edge cases
- byte fallback, if touched

Small synthetic protobuf models are preferred for narrow algorithm tests. If a compatibility fixture is needed, put the minimal fixture under `tests/fixtures/` and keep it small enough for crates.io packaging.

## Code Style

Write Rust like Rust.

- keep modules small
- return `Result`, do not panic in library code
- avoid `unsafe`
- keep public API boring and obvious
- prefer explicit errors over magic fallbacks
- do not add dependencies unless they earn their keep

If you touch hot tokenization paths, watch allocations and avoid cloning big strings for fun.

## Pull Request Checklist

Before opening a PR:

- `cargo fmt --all`
- `cargo test --all-features`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo doc --all-features --no-deps`
- README/docs updated if public behavior changed
- tests added or updated
- compatibility impact explained

If a check cannot run, say exactly why. Do not bury broken checks in vibes.

## Commit Messages

Use short Conventional Commit-style messages:

```text
feat: add pure Rust SentencePiece runtime
fix: handle byte fallback decode edge case
docs: clarify crates.io publish steps
test: add Unicode normalization coverage
ci: add GitHub Actions quality gate
```

Recommended types:

- `feat` for user-visible features
- `fix` for bug fixes
- `docs` for README, license, and contribution docs
- `test` for test-only changes
- `ci` for GitHub Actions and automation
- `refactor` for code cleanup without behavior changes
- `chore` for maintenance that does not affect runtime behavior

Keep the subject line lowercase after the type, imperative-ish, and under about 72 characters. If the commit needs context, add a body with bullet points.

## CI

GitHub Actions runs the same quality gate on every push to `main`/`master` and every pull request:

- formatting check
- full test suite
- clippy with warnings denied
- docs build

The workflow lives at `.github/workflows/ci.yml`.

## License

By contributing, you agree that your contribution is provided under the Apache License, Version 2.0, matching this crate.

The upstream SentencePiece source is also Apache-2.0. Preserve notices and do not imply endorsement by Google or the SentencePiece project.
