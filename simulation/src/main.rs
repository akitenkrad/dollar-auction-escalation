use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand, ValueEnum};
use dollar_auction_simulation::bidder::{
    Bidder, LlmBidder, QreBidder, SolverBidder, TranscriptSink,
};
use dollar_auction_simulation::config::DEFAULT_ROOT_SEED;
use dollar_auction_simulation::llm::scripted_client_from_file;
use dollar_auction_simulation::llm::{ollama_host, HttpTagsFetcher};
use dollar_auction_simulation::pilot::{run_pilot, OllamaClientFactory, PilotConfig};
use dollar_auction_simulation::prompt::{Framing, OpponentAnnouncement, PromptCondition};
use dollar_auction_simulation::qre_fit::run_qre_fit;
use dollar_auction_simulation::quiz::run_quiz;
use dollar_auction_simulation::record::{record_play, RecordConfig};
use dollar_auction_simulation::seeds::trial_seed;
use dollar_auction_simulation::simulation::empty_transcript_sink;
use dollar_auction_simulation::solver::{
    proposition_first_bid, solve, solve_mixed_ties, verify_reference_anchors,
};

#[derive(Parser, Debug)]
#[command(
    name = "dollar-auction",
    about = "Dollar-auction experiment and exact baselines"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Solve one game exactly.
    Solve(SolveArgs),
    /// Verify the exact solver against Proposition 3.1.
    VerifySolver,
    /// Play and record one scratch trial.
    Play(PlayArgs),
    /// Run the configured model-by-condition pilot sweep.
    Pilot(PilotArgs),
    /// Run the separate first-bid solution quiz.
    Quiz(QuizArgs),
    /// Fit agent-QRE lambda values from recorded LLM decisions.
    FitQre(FitQreArgs),
}

#[derive(Args, Debug)]
struct SolveArgs {
    /// Prize value.
    #[arg(long)]
    s: u32,
    /// Common budget.
    #[arg(long)]
    b: u32,
    /// Tie rule: drop or uniform mixing over optimal actions.
    #[arg(long, value_enum, default_value_t = TieRule::Drop)]
    tie: TieRule,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum TieRule {
    Drop,
    Mixed,
}

#[derive(Args, Debug)]
struct PlayArgs {
    /// Player 1 bidder: solver, qre:LAMBDA, or scripted:FILE.
    #[arg(long)]
    p1: String,
    /// Player 2 bidder: solver, qre:LAMBDA, or scripted:FILE.
    #[arg(long)]
    p2: String,
    /// Prize or contract value.
    #[arg(long)]
    s: u32,
    /// Common budget or cap.
    #[arg(long)]
    b: u32,
    /// Maximum accepted bids before censoring; omitted means unlimited.
    #[arg(long)]
    max_bids: Option<u32>,
    /// Prompt framing: named or disguised.
    #[arg(long, default_value = "named")]
    framing: String,
    /// Announced opponent type: rational_ai, human, or same_model.
    #[arg(long, default_value = "rational_ai")]
    opponent: String,
    /// Prompt paraphrase identifier: 0, 1, or 2.
    #[arg(long, default_value_t = 0)]
    paraphrase: u8,
    /// Enable the integer calculator for scripted LLM bidders.
    #[arg(long)]
    calculator: bool,
    /// Write paid-so-far in observations.
    #[arg(long)]
    display_paid_so_far: bool,
    /// Root seed used to derive the trial seed.
    #[arg(long, default_value_t = DEFAULT_ROOT_SEED)]
    seed: u64,
    /// Development-only run stored below the results root's _scratch directory.
    #[arg(long, required = true)]
    scratch: bool,
    /// Results root. Primarily used by isolated integration tests.
    #[arg(long, default_value = "results")]
    results_dir: PathBuf,
}

#[derive(Args, Debug)]
struct RunModeArgs {
    /// Store the run below the results root's _scratch directory.
    #[arg(long, required_unless_present = "prod", conflicts_with = "prod")]
    scratch: bool,
    /// Explicitly opt in to a non-scratch production run.
    #[arg(long, required_unless_present = "scratch", conflicts_with = "scratch")]
    prod: bool,
    /// Results root.
    #[arg(long, default_value = "results")]
    results_dir: PathBuf,
}

impl RunModeArgs {
    fn is_scratch(&self) -> bool {
        debug_assert_ne!(self.scratch, self.prod);
        self.scratch
    }
}

#[derive(Args, Debug)]
struct PilotArgs {
    /// Pilot configuration TOML.
    #[arg(long)]
    config: PathBuf,
    /// Previous sweep parent whose finished matching cells should be skipped.
    #[arg(long)]
    resume: Option<PathBuf>,
    #[command(flatten)]
    mode: RunModeArgs,
}

#[derive(Args, Debug)]
struct QuizArgs {
    /// Pilot configuration TOML supplying models, budgets, and trial count.
    #[arg(long)]
    config: PathBuf,
    #[command(flatten)]
    mode: RunModeArgs,
}

#[derive(Args, Debug)]
struct FitQreArgs {
    /// Pilot child or parent run directory.
    #[arg(long)]
    run: PathBuf,
    #[command(flatten)]
    mode: RunModeArgs,
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::Solve(args) => match args.tie {
            TieRule::Drop => {
                let result = solve(args.s, args.b);
                println!(
                    "{}",
                    serde_json::json!({
                        "domain": "analysis",
                        "s": args.s,
                        "b": args.b,
                        "tie": "drop",
                        "x1_star": result.first_bid,
                        "second_reply": result.second_reply,
                    })
                );
            }
            TieRule::Mixed => {
                let result = solve_mixed_ties(args.s, args.b);
                println!(
                    "{}",
                    serde_json::json!({
                        "domain": "analysis",
                        "s": args.s,
                        "b": args.b,
                        "tie": "mixed",
                        "first_bid": result.first_bid,
                        "first_mover_value": result.first_mover_value,
                    })
                );
            }
        },
        Command::VerifySolver => verify_solver()?,
        Command::Play(args) => play(args)?,
        Command::Pilot(args) => {
            let config = PilotConfig::from_path(&args.config)?;
            let factory = OllamaClientFactory::new(ollama_host());
            let result = run_pilot(
                &config,
                &args.mode.results_dir,
                args.mode.is_scratch(),
                args.resume.as_deref(),
                &HttpTagsFetcher,
                &factory,
            )?;
            println!("Parent run directory: {}", result.parent_dir.display());
            println!("Completed child runs: {}", result.child_dirs.len());
            println!("Skipped finished cells: {}", result.skipped_cells);
        }
        Command::Quiz(args) => {
            let config = PilotConfig::from_path(&args.config)?;
            let factory = OllamaClientFactory::new(ollama_host());
            let result = run_quiz(
                &config,
                &args.mode.results_dir,
                args.mode.is_scratch(),
                &HttpTagsFetcher,
                &factory,
            )?;
            println!("Parent run directory: {}", result.parent_dir.display());
            println!("Completed child runs: {}", result.child_dirs.len());
        }
        Command::FitQre(args) => {
            let result = run_qre_fit(&args.run, &args.mode.results_dir, args.mode.is_scratch())?;
            println!("Run directory: {}", result.display());
        }
    }
    Ok(())
}

fn verify_solver() -> Result<(), String> {
    for s in [3, 5, 10] {
        for b in 1..(4 * s) {
            let result = solve(s, b);
            let expected = proposition_first_bid(s, b);
            if result.first_bid != expected || result.second_reply.is_some() {
                return Err(format!(
                    "solver mismatch for s={s}, b={b}: {result:?}, expected first bid {expected} and immediate drop"
                ));
            }
        }
    }
    for (b, expected) in [(100, 1), (250, 52)] {
        let result = solve(100, b);
        if result.first_bid != expected || result.second_reply.is_some() {
            return Err(format!("solver mismatch for s=100, b={b}: {result:?}"));
        }
    }
    verify_reference_anchors(include_str!("../../reference.csv"))?;
    println!("Exact and mixed-tie solver verification passed.");
    Ok(())
}

fn play(args: PlayArgs) -> Result<(), String> {
    if !args.scratch {
        return Err("play requires --scratch".to_string());
    }
    if args.s < 2 || args.b < 1 {
        return Err("--s must be at least 2 and --b must be positive".to_string());
    }
    if args.max_bids == Some(0) {
        return Err("--max-bids must be positive when provided".to_string());
    }
    if args.paraphrase > 2 {
        return Err("--paraphrase must be 0, 1, or 2".to_string());
    }
    let framing = parse_framing(&args.framing)?;
    let opponent = parse_opponent(&args.opponent)?;
    let condition = PromptCondition {
        framing,
        opponent,
        paraphrase: args.paraphrase,
        calculator: args.calculator,
    };
    let trial = trial_seed(args.seed, 0, 0);
    let transcripts = empty_transcript_sink();
    let p1 = bidder_from_spec(&args.p1, condition, trial, 0, &transcripts)?;
    let p2 = bidder_from_spec(&args.p2, condition, trial, 1, &transcripts)?;
    let config = RecordConfig {
        s: args.s,
        b: args.b,
        p1: args.p1,
        p2: args.p2,
        framing: framing.as_str().to_string(),
        opponent: opponent.as_str().to_string(),
        paraphrase: args.paraphrase,
        calculator: args.calculator,
        display_paid_so_far: args.display_paid_so_far,
        root_seed: args.seed,
        trial_seed: trial,
        max_bids: args.max_bids,
    };
    let (directory, result) = record_play(
        &config,
        p1,
        p2,
        transcripts,
        &args.results_dir,
        args.scratch,
    )?;
    println!("Outcome: {}", result.outcome);
    println!("Run directory: {}", directory.display());
    Ok(())
}

fn bidder_from_spec(
    specification: &str,
    condition: PromptCondition,
    trial_seed: u64,
    seat: u64,
    transcripts: &TranscriptSink,
) -> Result<Box<dyn Bidder>, String> {
    if specification == "solver" {
        return Ok(Box::new(SolverBidder::new()));
    }
    if let Some(value) = specification.strip_prefix("qre:") {
        let lambda: f64 = value
            .parse()
            .map_err(|_| format!("invalid QRE lambda: {value}"))?;
        if !lambda.is_finite() || lambda < 0.0 {
            return Err("QRE lambda must be finite and non-negative".to_string());
        }
        return Ok(Box::new(QreBidder::new(
            lambda,
            socsim_core::derive_seed(trial_seed, &[seat]),
        )));
    }
    if let Some(path) = specification.strip_prefix("scripted:") {
        let client = scripted_client_from_file(Path::new(path))?;
        return Ok(Box::new(LlmBidder::with_sink(
            client,
            condition,
            trial_seed,
            transcripts.clone(),
        )));
    }
    Err(format!(
        "unsupported bidder {specification:?}; expected solver, qre:LAMBDA, or scripted:FILE"
    ))
}

fn parse_framing(value: &str) -> Result<Framing, String> {
    match value {
        "named" => Ok(Framing::Named),
        "disguised" => Ok(Framing::Disguised),
        _ => Err(format!("unsupported framing: {value}")),
    }
}

fn parse_opponent(value: &str) -> Result<OpponentAnnouncement, String> {
    match value {
        "rational_ai" => Ok(OpponentAnnouncement::RationalAi),
        "human" => Ok(OpponentAnnouncement::Human),
        "same_model" => Ok(OpponentAnnouncement::SameModel),
        _ => Err(format!("unsupported opponent announcement: {value}")),
    }
}
