# dollar-auction-escalation

This experiment studies whether language-model agents escalate in Shubik's dollar auction and how their behavior changes with the framing and announced opponent type.

Both the winner and the loser pay their own last bid. With prize value `s` and a commonly known budget `b`, the exact first bid is `(b - 1) mod (s - 1) + 1`; the second mover then drops. The repository includes a Rust exact solver, a separate Python implementation, a full-game agent-QRE baseline, prompt templates, protocol enforcement, and runvault recording.

The model execution path talks directly to a local Ollama server. Development and tests remain model-free by using socsim-llm's scripted client or a test double. No response cache or provider fallback is used.

## Layout

- `simulation/` contains the Rust game, socsim engine mechanisms, baselines, Ollama and scripted LLM paths, pilot and quiz runners, QRE fitting, protocol, and runvault recording.
- `simulation/tests/` contains rule, solver, QRE, prompt, protocol, condition, recording, and LLM integration tests.
- `prompts/` contains the six framing/paraphrase templates and shared prompt fragments.
- `tools/` contains the independent Python solver, registered analyses, plots, and reason-coding utilities.
- `runvault.toml` declares every recorded aggregate metric and parameter.

## Build and test

All Rust commands use the local dependency cache:

```bash
cargo build --offline
cargo test --offline
cargo clippy --offline --all-targets -- -D warnings
cargo fmt --check
```

Run the Python tests in the repository environment:

```bash
.venv/bin/python -m unittest discover tools/tests
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

Run a complete pilot or the separate solution quiz only after the configured Ollama tags and digest prefixes have been checked:

```bash
cargo run --offline -- pilot --config pilot.toml --scratch
cargo run --offline -- pilot --config pilot.toml --resume RESULTS/PARENT --scratch
cargo run --offline -- quiz --config pilot.toml --scratch
```

`--prod` is the explicit alternative to `--scratch` for a production run. Pilot cells are ordered model-first so both models are not loaded together. `pilot.smoke.toml` provides a small live smoke configuration for the configured Gemma model.

Pilot configuration sets `max_bids` (50 in the shipped configurations) and per-cell trial `concurrency`. A trial reaching `max_bids` is recorded as the censored `turn_cap` outcome. `play` remains unlimited unless `--max-bids` is supplied explicitly.

Fit agent-QRE to a recorded child or sweep, then run the registered analysis without modifying any input run:

```bash
cargo run --offline -- fit-qre --run RESULTS/RUN --scratch
.venv/bin/dollar-auction-tools analyze --sweep RESULTS/PARENT \
  --quiz RESULTS/QUIZ --qre RESULTS/QRE_FIT --out analysis/example
.venv/bin/dollar-auction-tools code-reasons export \
  --run RESULTS/PARENT --out analysis/example/reasons.csv
.venv/bin/dollar-auction-tools code-reasons agreement CODER_A.csv CODER_B.csv
.venv/bin/dollar-auction-tools visualize analysis/example
```

Generate the cross-language vectors with:

```bash
.venv/bin/python -m dollar_auction_tools.solver_ref gen-vectors \
  --output simulation/tests/vectors.json
```

Recorded trials are stored under the runvault scratch tree. Per-trial observations and outcomes are in `events.jsonl`, cell aggregates are in `metrics.csv`, and complete scripted model calls are in `artifacts/transcripts.jsonl`.
Every model request explicitly sets `think` to `false`. A model response may contain either plain JSON or exactly one enclosing three-backtick Markdown fence, optionally tagged `json`; transcript records preserve whether the accepted envelope was fenced.

## License

MIT
