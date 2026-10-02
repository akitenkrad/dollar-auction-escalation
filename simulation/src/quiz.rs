use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use runvault::{Lineage, Run, RunOptions};
use serde::Deserialize;
use serde_json::json;
use socsim_core::derive_seed;

use crate::bidder::Transcript;
use crate::llm::{check_model_digest, model_config, TagsFetcher};
use crate::pilot::{LlmClientFactory, PilotConfig};
use crate::prompt::Framing;
use crate::seeds::llm_call_seed;
use crate::solver::{proposition_first_bid, solve_mixed_ties};
use crate::usage::UsageSummary;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuizPrompt {
    pub system: String,
    pub user: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuizAnswer {
    pub first_bid: u32,
    pub reason: String,
    pub fenced: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuizScore {
    pub first_bid: Option<u32>,
    pub correct_drop_tie: bool,
    pub correct_mixed_tie: bool,
    pub invalid: bool,
}

#[derive(Deserialize)]
struct RawQuizAnswer {
    first_bid: u32,
    reason: String,
}

pub fn render_quiz_prompt(s: u32, b: u32, framing: Framing, paraphrase: u8) -> QuizPrompt {
    let template = match (framing, paraphrase) {
        (Framing::Named, 0) => include_str!("../../prompts/named_0.txt"),
        (Framing::Named, 1) => include_str!("../../prompts/named_1.txt"),
        (Framing::Named, 2) => include_str!("../../prompts/named_2.txt"),
        (Framing::Disguised, 0) => include_str!("../../prompts/disguised_0.txt"),
        (Framing::Disguised, 1) => include_str!("../../prompts/disguised_1.txt"),
        (Framing::Disguised, 2) => include_str!("../../prompts/disguised_2.txt"),
        (_, id) => panic!("unsupported paraphrase id: {id}"),
    };
    let system = template
        .replace("{{s}}", &s.to_string())
        .replace("{{b}}", &b.to_string())
        .replace("{{player}}", "A")
        .replace(
            "Player 1 moves first",
            "The designated first mover acts first",
        )
        .replace(
            "Player 1 acts first",
            "The designated first mover acts first",
        )
        .replace(
            "Player 1 has the opening turn",
            "The designated first mover has the opening turn",
        )
        .replace("Firm 1 moves first", "The designated first firm acts first")
        .replace("Firm 1 acts first", "The designated first firm acts first")
        .replace(
            "Firm 1 has the opening turn",
            "The designated first firm has the opening turn",
        );
    let user = match framing {
        Framing::Named => include_str!("../../prompts/quiz_named.txt"),
        Framing::Disguised => include_str!("../../prompts/quiz_disguised.txt"),
    }
    .trim_end()
    .to_string();
    QuizPrompt { system, user }
}

pub fn parse_quiz_answer(raw: &str) -> Result<QuizAnswer, String> {
    let fenced_payload = single_fenced_payload(raw);
    let fenced = fenced_payload.is_some();
    let normalized = fenced_payload.unwrap_or_else(|| raw.trim().to_string());
    let parsed: RawQuizAnswer =
        serde_json::from_str(&normalized).map_err(|error| format!("invalid quiz JSON: {error}"))?;
    if parsed.reason.trim().is_empty() {
        return Err("quiz reason must not be empty".to_string());
    }
    Ok(QuizAnswer {
        first_bid: parsed.first_bid,
        reason: parsed.reason,
        fenced,
    })
}

pub fn score_quiz_answer(s: u32, b: u32, parsed: Result<QuizAnswer, String>) -> QuizScore {
    match parsed {
        Ok(answer) => QuizScore {
            first_bid: Some(answer.first_bid),
            correct_drop_tie: answer.first_bid == proposition_first_bid(s, b),
            correct_mixed_tie: answer.first_bid == solve_mixed_ties(s, b).first_bid,
            invalid: false,
        },
        Err(_) => QuizScore {
            first_bid: None,
            correct_drop_tie: false,
            correct_mixed_tie: false,
            invalid: true,
        },
    }
}

#[derive(Debug)]
pub struct QuizOutcome {
    pub parent_dir: PathBuf,
    pub child_dirs: Vec<PathBuf>,
}

pub fn run_quiz(
    config: &PilotConfig,
    results_root: &Path,
    scratch: bool,
    tags: &dyn TagsFetcher,
    factory: &dyn LlmClientFactory,
) -> Result<QuizOutcome, String> {
    if (config.temperature - crate::config::DEFAULT_TEMPERATURE).abs() > f32::EPSILON {
        return Err("quiz temperature must be 0.7".to_string());
    }
    let host = crate::llm::ollama_host();
    let mut verified_digests = BTreeMap::new();
    for model in &config.models {
        let digest = check_model_digest(tags, &host, model)?;
        verified_digests.insert(model.tag.clone(), digest);
    }
    let parameters = json!({
        "command": "quiz",
        "s": config.s,
        "budgets": config.budgets,
        "trials": config.trials,
        "temperature": config.temperature,
        "think": false,
        "root_seed": config.root_seed,
        "models": config.models,
        "verified_model_digests": verified_digests,
    });
    let parent = Run::start(
        RunOptions::new("dollar-auction", "quiz")
            .scratch(scratch)
            .repo_id("dollar-auction-escalation")
            .domain("analysis")
            .results_root(results_root)
            .parameters(&parameters)
            .map_err(|error| error.to_string())?
            .seed_pointers(["/root_seed"])
            .sweep_parent(),
    )
    .map_err(|error| error.to_string())?;
    let parent_dir = parent.dir().to_path_buf();
    let parent_uid = parent.run_uid().to_string();
    let sweep_id = parent
        .sweep_id()
        .ok_or_else(|| "quiz parent has no sweep id".to_string())?
        .to_string();
    let total = config.models.len() * 2 * config.budgets.len();
    let mut child_dirs = Vec::with_capacity(total);
    let mut cell_index = 0;
    for model in &config.models {
        for framing in [Framing::Named, Framing::Disguised] {
            for &b in &config.budgets {
                let digest = check_model_digest(tags, &host, model)?;
                let child = run_quiz_cell(
                    config,
                    model,
                    framing,
                    b,
                    cell_index,
                    total,
                    results_root,
                    scratch,
                    &sweep_id,
                    &parent_uid,
                    &digest,
                    factory,
                )?;
                child_dirs.push(child);
                cell_index += 1;
            }
        }
    }
    parent.finish().map_err(|error| error.to_string())?;
    Ok(QuizOutcome {
        parent_dir,
        child_dirs,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_quiz_cell(
    config: &PilotConfig,
    model: &crate::config::ModelSpec,
    framing: Framing,
    b: u32,
    cell_index: usize,
    total: usize,
    results_root: &Path,
    scratch: bool,
    sweep_id: &str,
    parent_uid: &str,
    model_digest: &str,
    factory: &dyn LlmClientFactory,
) -> Result<PathBuf, String> {
    let parameters = json!({
        "command": "quiz-cell",
        "model": model.tag,
        "model_digest": model_digest,
        "model_digest_prefix": model.digest_prefix,
        "s": config.s,
        "b": b,
        "framing": framing.as_str(),
        "trials": config.trials,
        "temperature": config.temperature,
        "think": false,
        "root_seed": config.root_seed,
    });
    let mut run = Run::start(
        RunOptions::new("dollar-auction", "quiz-cell")
            .scratch(scratch)
            .repo_id("dollar-auction-escalation")
            .domain("analysis")
            .results_root(results_root)
            .parameters(&parameters)
            .map_err(|error| error.to_string())?
            .seed_pointers(["/root_seed"])
            .sweep_point(cell_index as u64, total as u64)
            .lineage(Lineage {
                sweep_id: Some(sweep_id.to_string()),
                parent_run_uid: Some(parent_uid.to_string()),
                ..Default::default()
            }),
    )
    .map_err(|error| error.to_string())?;
    let mut transcripts = Vec::with_capacity(config.trials);
    let mut scores = Vec::with_capacity(config.trials);
    let mut metadata = Vec::with_capacity(config.trials);
    for trial in 0..config.trials {
        let prompt = render_quiz_prompt(config.s, b, framing, (trial % 3) as u8);
        let trial_seed = derive_seed(config.root_seed, &[cell_index as u64, trial as u64]);
        let seed = llm_call_seed(trial_seed, 1, 1);
        let request = model_config(prompt.system.clone(), seed);
        let client = factory.create(&model.tag);
        let response = client
            .complete(&prompt.user, &request)
            .map_err(|error| format!("quiz model call failed: {error}"))?;
        let thinking_observed = response.thinking.is_some();
        let parsed = parse_quiz_answer(&response.text);
        let reason = parsed.as_ref().ok().map(|answer| answer.reason.clone());
        let fenced = parsed.as_ref().is_ok_and(|answer| answer.fenced);
        let score = score_quiz_answer(config.s, b, parsed);
        run.log_event(
            "x.dollar-auction-escalation.quiz",
            &json!({
                "unit_id": format!("quiz-{trial}"),
                "trial": trial,
                "framing": framing.as_str(),
                "model": model.tag,
                "b": b,
                "paraphrase_id": trial % 3,
                "first_bid": score.first_bid,
                "correct_drop_tie": score.correct_drop_tie,
                "correct_mixed_tie": score.correct_mixed_tie,
                "invalid": score.invalid,
                "thinking_observed": thinking_observed,
                "reason": reason,
            }),
        )
        .map_err(|error| error.to_string())?;
        metadata.push(response.metadata.clone());
        transcripts.push(Transcript {
            trial: Some(trial),
            turn: 1,
            attempt: 1,
            player: 1,
            system: prompt.system,
            user: prompt.user,
            raw_response: response.text,
            thinking: response.thinking,
            seed,
            token_usage: response.metadata.usage,
            metadata: response.metadata,
            think: request.think,
            fenced,
            raw_action: None,
            reason,
        });
        scores.push(score);
    }
    write_quiz_transcripts(run.dir(), &transcripts)?;
    let usage = UsageSummary::from_metadata(&metadata);
    let n = scores.len() as f64;
    let mut metrics = vec![
        ("n_trials", n),
        (
            "quiz_drop_solve_rate",
            scores.iter().filter(|score| score.correct_drop_tie).count() as f64 / n,
        ),
        (
            "quiz_mixed_solve_rate",
            scores
                .iter()
                .filter(|score| score.correct_mixed_tie)
                .count() as f64
                / n,
        ),
        (
            "quiz_invalid_rate",
            scores.iter().filter(|score| score.invalid).count() as f64 / n,
        ),
    ];
    metrics.extend(usage.metrics());
    run.log_metrics("run", &metrics)
        .map_err(|error| error.to_string())?;
    run.finish().map_err(|error| error.to_string())
}

fn write_quiz_transcripts(run_dir: &Path, transcripts: &[Transcript]) -> Result<(), String> {
    let artifacts = run_dir.join("artifacts");
    fs::create_dir_all(&artifacts).map_err(|error| error.to_string())?;
    let mut writer = BufWriter::new(
        File::create(artifacts.join("transcripts.jsonl")).map_err(|error| error.to_string())?,
    );
    for transcript in transcripts {
        serde_json::to_writer(&mut writer, transcript).map_err(|error| error.to_string())?;
        writer.write_all(b"\n").map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

fn single_fenced_payload(raw: &str) -> Option<String> {
    let lines: Vec<_> = raw.trim().lines().collect();
    if lines.len() < 3 {
        return None;
    }
    let opening = lines.first()?.trim();
    if opening != "```" && !opening.eq_ignore_ascii_case("```json") {
        return None;
    }
    if lines.last()?.trim() != "```"
        || lines[1..lines.len() - 1]
            .iter()
            .any(|line| line.trim().starts_with("```"))
    {
        return None;
    }
    Some(lines[1..lines.len() - 1].join("\n"))
}
