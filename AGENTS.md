# AGENTS.md

This file defines repository-wide instructions for AI coding agents working on HexGo.

## Start Every Task Safely

- Read `README.md`, `CONTRIBUTING.md`, and the relevant files before proposing or making changes.
- Run `git status --short --branch` before editing tracked files.
- Never work directly on `main`. If the current branch is `main`, create a focused branch first, following the naming rules in `CONTRIBUTING.md`.
- Preserve all existing user changes. Do not discard, overwrite, or reformat unrelated work.
- Do not use destructive Git commands or bypass hooks.
- Do not commit, push, open a pull request, merge, or change remote repository settings unless the user explicitly requests that action.

## Sources of Truth

- `docs/RULES.md` is the normative English game specification.
- `docs/RULES.zh-CN.md` is the official Simplified Chinese translation.
- When the rule documents disagree, report the discrepancy instead of guessing. Do not silently choose or invent game behavior.
- A rule change requires explicit approval and must update both language versions in the same pull request.
- English documents are normative. Keep every paired `.zh-CN.md` translation synchronized with its English source.

## Rust Engineering Standards

- Use stable Rust, edition 2024, and the toolchain declared in `rust-toolchain.toml`.
- Use standard `rustfmt`; do not introduce custom formatting conventions.
- Keep `cargo clippy --all-targets --all-features -- -D warnings` clean.
- Keep domain rules independent from Bevy, rendering, input, and platform concerns, as described in `docs/ARCHITECTURE.md`.
- Prefer small, cohesive modules, explicit types, and deterministic rule logic.
- Add or update tests for every behavior change and every fixed bug.
- Do not add dependencies, introduce `unsafe`, or change public behavior without explicit approval.
- Never edit generated files such as `Cargo.lock` by hand.
- Preserve workspace dev build settings (`[profile.dev]` with `debug = 1` and optimized dependencies). Heavy Bevy and Burn dependencies cause default `debug = 2` binaries to exceed 1.5GB each and consume massive disk space (>100GB in `target/`). Never revert or bypass these settings.

## Language and Documentation

- Use English for source code, identifiers, code comments, Rust doc comments, commit messages, branch names, logs, and user-facing technical diagnostics.
- Comments should explain intent, invariants, or non-obvious tradeoffs. Do not narrate obvious code.
- Use English as the default documentation language and provide Simplified Chinese for all paired project documents.
- Update documentation whenever behavior, commands, architecture, or contributor workflows change.
- Keep relative links valid and preserve the language switcher at the top of paired documents.

## Required Validation

Run checks appropriate to the change. Before handing off a completed code change, normally run:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --all-features
cargo test --all-features
```

For documentation-only changes, also run `git diff --check` and verify all changed relative links. Do not claim a check passed unless it was actually run.

## Pull Request and AI Accountability

- Keep one pull request focused on one concern.
- Use Conventional Commit style for commit messages and pull request titles.
- Describe what changed, why it changed, tests performed, rule impact, and translation status.
- Disclose AI assistance in the pull request, including its scope, the human review performed, and validation results.
- Human contributors remain responsible for correctness, licensing, security, and the final submitted change.
- During review, prioritize rule correctness, regressions, missing tests, unsafe behavior, stale translations, and violations of the protected-branch workflow.
