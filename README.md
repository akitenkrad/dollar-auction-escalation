# dollar-auction-escalation

This experiment studies whether language-model agents escalate in Shubik's dollar auction and how their behavior changes with the framing and announced opponent type.

Both the winner and the loser pay their own last bid. With prize value `s` and a commonly known budget `b`, the exact first bid is `(b - 1) mod (s - 1) + 1`; the second mover then drops. The repository includes a Rust exact solver, a separate Python implementation, a full-game agent-QRE baseline, prompt templates, protocol enforcement, and runvault recording.

The LLM execution path is network-free: `scripted:FILE` sends one JSON-lines response per call through socsim-llm's scripted client. No response cache or provider fallback is used.

## Layout

- `simulation/` contains the Rust game, socsim engine mechanisms, baselines, scripted LLM bidder, protocol, and runvault recorder.
- `simulation/tests/` contains rule, solver, QRE, prompt, protocol, condition, recording, and LLM integration tests.
- `prompts/` contains the six framing/paraphrase templates and shared prompt fragments.
- `tools/` contains the standard-library-only Python reference solver and vector generator.
- `runvault.toml` declares every recorded aggregate metric and parameter.

## Build and test

All Rust commands use the local dependency cache:

```bash
cargo build --offline
cargo test --offline
cargo clippy --offline --all-targets -- -D warnings
cargo fmt --check
```

Run the Python tests without third-party packages:

```bash
PYTHONPATH=tools/src python3 -m unittest discover tools/tests
```

## Commands

Solve one game or verify the complete test grid:

```bash
cargo run --offline -- solve --s 100 --b 250
cargo run --offline -- solve --s 100 --b 250 --tie mixed
cargo run --offline -- verify-solver
```

`solve` uses the proposition's drop-on-ties rule by default. `--tie mixed`
reports the subgame-perfect limit that mixes uniformly over payoff ties, which
is the large-`lambda` limit of the agent-logit QRE.

Every `play` run must be explicitly marked as scratch data:

```bash
cargo run --offline -- play \
  --p1 solver --p2 solver --s 100 --b 250 --scratch

cargo run --offline -- play \
  --p1 scripted:simulation/tests/fixtures/bid_cap_then_drop.jsonl \
  --p2 solver --s 3 --b 1 --scratch
```

Bidder specifications are `solver`, `qre:LAMBDA`, and `scripted:FILE`. Prompt controls are `--framing named|disguised`, `--opponent rational_ai|human|same_model`, `--paraphrase 0|1|2`, and `--calculator`.

Generate the cross-language vectors with:

```bash
PYTHONPATH=tools/src python3 -m dollar_auction_tools.solver_ref gen-vectors \
  --output simulation/tests/vectors.json
```

Recorded trials are stored under the runvault scratch tree. Per-trial observations and outcomes are in `events.jsonl`, cell aggregates are in `metrics.csv`, and complete scripted model calls are in `artifacts/transcripts.jsonl`.
Every model request explicitly sets `think` to `false`. A model response may contain either plain JSON or exactly one enclosing three-backtick Markdown fence, optionally tagged `json`; transcript records preserve whether the accepted envelope was fenced.

## License

MIT
