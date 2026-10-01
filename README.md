# dollar-auction-escalation

Do LLM agents escalate in Shubik's dollar auction, or do they play the subgame-perfect equilibrium?

In the dollar auction both the winner and the loser pay their last bid. With a commonly known budget `b`, backward induction predicts that the first mover bids `(b - 1) mod (s - 1) + 1` and the second mover drops out immediately (O'Neill, 1986), while human subjects are known to bid far beyond the prize value. This experiment places LLM agents in the same game and measures where they land between theory and human behaviour, and which manipulations (naming the game, announcing the opponent's type) move them.

## Layout

- `simulation/` — Rust (socsim library mode): the game, an exact backward-induction solver, agent-QRE baselines, the LLM protocol, and runvault recording. Crate `dollar-auction-simulation`, binary `dollar-auction`.
- `tools/` — Python analysis (`dollar-auction-tools`).

## Build

```bash
cargo build --release
uv sync
uv run dollar-auction-tools --help
```

Models are served locally by [Ollama](https://ollama.com) (`OLLAMA_HOST`, default `http://localhost:11434`).

## License

MIT
