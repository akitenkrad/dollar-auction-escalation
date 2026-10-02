use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use runvault::config::{Exclusions, RunvaultBlock};
use runvault::{Lineage, Run, RunOptions};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use socsim_core::Recorder;
use socsim_llm::LlmClient;

use crate::bidder::{Bidder, LlmBidder, SolverBidder, Transcript, TranscriptSink};
use crate::config::ModelSpec;
use crate::llm::{check_model_digest, TagsFetcher};
use crate::prompt::{Framing, OpponentAnnouncement, PromptCondition};
use crate::seeds::trial_seed;
use crate::simulation::{run_trial_with_recorder, TrialResult, TrialRunConfig};
use crate::usage::UsageSummary;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PilotConfig {
    pub s: u32,
    pub budgets: Vec<u32>,
    pub trials: usize,
    pub temperature: f32,
    #[serde(default = "default_root_seed")]
    pub root_seed: u64,
    #[serde(default = "default_max_bids")]
    pub max_bids: u32,
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    pub models: Vec<ModelSpec>,
    pub cells: Vec<PilotCellConfig>,
}

fn default_root_seed() -> u64 {
    42
}

fn default_max_bids() -> u32 {
    50
}

fn default_concurrency() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PilotCellConfig {
    pub kind: String,
    pub framing: String,
    pub opponent: String,
    pub b: u32,
    #[serde(default)]
    pub calculator: bool,
}

impl PilotConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        toml::from_str(&text)
            .map_err(|error| format!("failed to parse {}: {error}", path.display()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedCell {
    pub model: ModelSpec,
    pub kind: String,
    pub framing: Framing,
    pub opponent: OpponentAnnouncement,
    pub b: u32,
    pub calculator: bool,
    pub index: usize,
}

impl PlannedCell {
    pub fn key(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}|{}",
            self.model.tag,
            self.kind,
            self.framing.as_str(),
            self.opponent.as_str(),
            self.b,
            self.calculator
        )
    }
}

pub fn plan_cells(config: &PilotConfig) -> Result<Vec<PlannedCell>, String> {
    if config.models.is_empty() {
        return Err("pilot config must contain at least one model".to_string());
    }
    if config.cells.is_empty() || config.trials == 0 {
        return Err("pilot config must contain cells and positive trials".to_string());
    }
    if config.max_bids == 0 {
        return Err("pilot max_bids must be positive".to_string());
    }
    if config.concurrency == 0 {
        return Err("pilot concurrency must be positive".to_string());
    }
    if (config.temperature - crate::config::DEFAULT_TEMPERATURE).abs() > f32::EPSILON {
        return Err("pilot temperature must be 0.7".to_string());
    }
    let mut planned = Vec::with_capacity(config.models.len() * config.cells.len());
    for model in &config.models {
        for cell in &config.cells {
            if !config.budgets.contains(&cell.b) {
                return Err(format!("cell budget {} is not declared in budgets", cell.b));
            }
            let framing = parse_framing(&cell.framing)?;
            let opponent = parse_opponent(&cell.opponent)?;
            if !matches!(cell.kind.as_str(), "main" | "llm_vs_solver" | "calculator") {
                return Err(format!("unsupported pilot cell kind: {}", cell.kind));
            }
            if cell.kind != "main" && framing != Framing::Disguised {
                return Err(format!("{} cell must use disguised framing", cell.kind));
            }
            planned.push(PlannedCell {
                model: model.clone(),
                kind: cell.kind.clone(),
                framing,
                opponent,
                b: cell.b,
                calculator: cell.calculator || cell.kind == "calculator",
                index: planned.len(),
            });
        }
    }
    Ok(planned)
}

pub trait LlmClientFactory: Sync {
    fn create(&self, model: &str) -> Box<dyn LlmClient>;
}

#[derive(Debug, Clone)]
pub struct OllamaClientFactory {
    host: String,
}

impl OllamaClientFactory {
    pub fn new(host: impl Into<String>) -> Self {
        Self { host: host.into() }
    }
}

impl LlmClientFactory for OllamaClientFactory {
    fn create(&self, model: &str) -> Box<dyn LlmClient> {
        crate::llm::direct_ollama_client(&self.host, model)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExistingChild {
    pub config_hash: String,
    pub state: String,
    pub observed_trials: Vec<usize>,
}

impl ExistingChild {
    pub fn new(config_hash: &str, state: &str, observed_trials: Vec<usize>) -> Self {
        Self {
            config_hash: config_hash.to_string(),
            state: state.to_string(),
            observed_trials,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeDecision {
    SkipFinished,
    RunFromTrialZero,
}

pub fn resume_decision(config_hash: &str, existing: &[ExistingChild]) -> ResumeDecision {
    if existing
        .iter()
        .any(|child| child.config_hash == config_hash && child.state == "finished")
    {
        ResumeDecision::SkipFinished
    } else {
        ResumeDecision::RunFromTrialZero
    }
}

#[derive(Debug)]
pub struct ProgressState {
    total: usize,
    last_report: Option<Duration>,
}

impl ProgressState {
    pub fn new(total: usize) -> Self {
        Self {
            total,
            last_report: None,
        }
    }

    pub fn update(
        &mut self,
        completed: usize,
        calls: usize,
        elapsed: Duration,
        completed_cell: bool,
    ) -> Option<String> {
        let due = self.last_report.is_none()
            || completed_cell
            || self
                .last_report
                .is_some_and(|last| elapsed.saturating_sub(last) >= Duration::from_secs(30));
        if !due {
            return None;
        }
        self.last_report = Some(elapsed);
        let eta_seconds = if completed == 0 {
            None
        } else {
            let remaining = self.total.saturating_sub(completed) as f64;
            Some(elapsed.as_secs_f64() / completed as f64 * remaining)
        };
        let eta = eta_seconds.map_or_else(|| "unknown".to_string(), format_duration);
        Some(format!(
            "progress: trials {completed}/{} calls={calls} elapsed={} ETA={eta}",
            self.total,
            format_duration(elapsed.as_secs_f64())
        ))
    }
}

fn format_duration(seconds: f64) -> String {
    let rounded = seconds.max(0.0).round() as u64;
    format!("{}m{:02}s", rounded / 60, rounded % 60)
}

#[derive(Debug)]
pub struct PilotOutcome {
    pub parent_dir: PathBuf,
    pub child_dirs: Vec<PathBuf>,
    pub skipped_cells: usize,
}

pub fn run_pilot(
    config: &PilotConfig,
    results_root: &Path,
    scratch: bool,
    resume: Option<&Path>,
    tags: &dyn TagsFetcher,
    factory: &dyn LlmClientFactory,
) -> Result<PilotOutcome, String> {
    let cells = plan_cells(config)?;
    let host = crate::llm::ollama_host();
    let mut verified_digests = BTreeMap::new();
    for model in &config.models {
        let digest = check_model_digest(tags, &host, model)?;
        verified_digests.insert(model.tag.clone(), digest);
    }
    let existing = resume
        .map(read_existing_children)
        .transpose()?
        .unwrap_or_default();
    let parent_parameters = json!({
        "command": "pilot",
        "s": config.s,
        "budgets": config.budgets,
        "trials": config.trials,
        "temperature": config.temperature,
        "think": false,
        "root_seed": config.root_seed,
        "max_bids": config.max_bids,
        "concurrency": config.concurrency,
        "models": config.models,
        "verified_model_digests": verified_digests,
        "cells": config.cells,
        "resume": resume.map(|path| path.display().to_string()),
    });
    let mut parent_options = RunOptions::new("dollar-auction", "pilot")
        .scratch(scratch)
        .repo_id("dollar-auction-escalation")
        .domain("simulation")
        .results_root(results_root)
        .parameters(&parent_parameters)
        .map_err(|error| error.to_string())?
        .seed_pointers(["/root_seed"])
        .sweep_parent();
    if let Some(previous) = resume {
        let previous_meta = read_json(previous.join("run.json"))?;
        if let Some(uid) = previous_meta.get("run_uid").and_then(Value::as_str) {
            parent_options = parent_options.lineage(Lineage {
                resumed_from: Some(uid.to_string()),
                ..Default::default()
            });
        }
    }
    let parent = Run::start(parent_options).map_err(|error| error.to_string())?;
    let parent_dir = parent.dir().to_path_buf();
    let parent_uid = parent.run_uid().to_string();
    let sweep_id = parent
        .sweep_id()
        .ok_or_else(|| "pilot parent has no sweep id".to_string())?
        .to_string();
    let total_trials = cells.len() * config.trials;
    let started = Instant::now();
    let mut progress = ProgressState::new(total_trials);
    let mut completed_trials = 0;
    let mut completed_calls = 0;
    let shared_trials = Arc::new(AtomicUsize::new(0));
    let shared_calls = Arc::new(AtomicUsize::new(0));
    write_progress(
        &parent_dir,
        &progress
            .update(0, 0, Duration::ZERO, false)
            .expect("initial progress line"),
    )?;
    let heartbeat = ProgressHeartbeat::start(
        parent_dir.clone(),
        total_trials,
        started,
        Arc::clone(&shared_trials),
        Arc::clone(&shared_calls),
    );
    let mut child_dirs = Vec::new();
    let mut skipped_cells = 0;

    for cell in &cells {
        let initial_digest = verified_digests
            .get(&cell.model.tag)
            .ok_or_else(|| format!("model {} was not preflighted", cell.model.tag))?;
        let expected_hash = cell_config_hash(config, cell, initial_digest)?;
        if resume_decision(&expected_hash, &existing) == ResumeDecision::SkipFinished {
            skipped_cells += 1;
            completed_trials += config.trials;
            shared_trials.store(completed_trials, Ordering::Relaxed);
            if let Some(line) =
                progress.update(completed_trials, completed_calls, started.elapsed(), true)
            {
                write_progress(&parent_dir, &line)?;
            }
            continue;
        }
        let digest = check_model_digest(tags, &host, &cell.model)?;
        let calls_before_cell = completed_calls;
        let child = run_cell(
            config,
            cell,
            results_root,
            scratch,
            &sweep_id,
            &parent_uid,
            cells.len(),
            &digest,
            Arc::clone(&shared_trials),
            Arc::clone(&shared_calls),
            factory,
        )?;
        completed_trials = shared_trials.load(Ordering::Relaxed);
        completed_calls = shared_calls.load(Ordering::Relaxed);
        debug_assert_eq!(child.calls, completed_calls - calls_before_cell);
        child_dirs.push(child.directory);
        if let Some(line) =
            progress.update(completed_trials, completed_calls, started.elapsed(), true)
        {
            write_progress(&parent_dir, &line)?;
        }
    }
    heartbeat.stop();
    parent.finish().map_err(|error| error.to_string())?;
    Ok(PilotOutcome {
        parent_dir,
        child_dirs,
        skipped_cells,
    })
}

struct CellOutcome {
    directory: PathBuf,
    calls: usize,
}

#[allow(clippy::too_many_arguments)]
fn run_cell(
    config: &PilotConfig,
    cell: &PlannedCell,
    results_root: &Path,
    scratch: bool,
    sweep_id: &str,
    parent_uid: &str,
    total_cells: usize,
    model_digest: &str,
    completed_trials: Arc<AtomicUsize>,
    completed_calls: Arc<AtomicUsize>,
    factory: &dyn LlmClientFactory,
) -> Result<CellOutcome, String> {
    let parameters = cell_parameters(config, cell, model_digest);
    let mut run = Run::start(
        RunOptions::new("dollar-auction", "pilot-cell")
            .scratch(scratch)
            .repo_id("dollar-auction-escalation")
            .domain("simulation")
            .results_root(results_root)
            .parameters(&parameters)
            .map_err(|error| error.to_string())?
            .seed_pointers(["/root_seed"])
            .master_seed(config.root_seed)
            .sweep_point(cell.index as u64, total_cells as u64)
            .lineage(Lineage {
                sweep_id: Some(sweep_id.to_string()),
                parent_run_uid: Some(parent_uid.to_string()),
                ..Default::default()
            }),
    )
    .map_err(|error| error.to_string())?;
    let mut buffered = Vec::with_capacity(config.trials);
    for first in (0..config.trials).step_by(config.concurrency) {
        let last = (first + config.concurrency).min(config.trials);
        let batch = thread::scope(|scope| {
            let mut handles = Vec::with_capacity(last - first);
            for trial_index in first..last {
                handles
                    .push(scope.spawn(move || run_cell_trial(config, cell, trial_index, factory)));
            }
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| "pilot trial thread panicked".to_string())?
                })
                .collect::<Result<Vec<_>, String>>()
        })?;
        for trial in &batch {
            completed_calls.fetch_add(trial.transcripts.len(), Ordering::Relaxed);
            completed_trials.fetch_add(1, Ordering::Relaxed);
        }
        buffered.extend(batch);
    }
    buffered.sort_by_key(|trial| trial.trial_index);
    let mut results = Vec::with_capacity(buffered.len());
    let mut transcripts = Vec::new();
    for mut trial in buffered {
        for event in trial.events.drain(..) {
            write_captured_event(&mut run, event, trial.trial_index, config.s, cell)?;
        }
        results.push(trial.result);
        transcripts.append(&mut trial.transcripts);
    }
    write_transcripts(run.dir(), &transcripts)?;
    let metadata: Vec<_> = transcripts
        .iter()
        .map(|transcript| transcript.metadata.clone())
        .collect();
    let usage = UsageSummary::from_metadata(&metadata);
    log_cell_metrics(&mut run, &results, usage)?;
    let calls = usage.n_calls as usize;
    let directory = run.finish().map_err(|error| error.to_string())?;
    Ok(CellOutcome { directory, calls })
}

struct BufferedTrial {
    trial_index: usize,
    result: TrialResult,
    events: Vec<CapturedEvent>,
    transcripts: Vec<Transcript>,
}

fn run_cell_trial(
    config: &PilotConfig,
    cell: &PlannedCell,
    trial_index: usize,
    factory: &dyn LlmClientFactory,
) -> Result<BufferedTrial, String> {
    let transcripts: TranscriptSink = Rc::new(RefCell::new(Vec::new()));
    let seed = trial_seed(config.root_seed, cell.index as u64, trial_index as u64);
    let condition = PromptCondition {
        framing: cell.framing,
        opponent: cell.opponent,
        paraphrase: (trial_index % 3) as u8,
        calculator: cell.calculator,
    };
    let p1: Box<dyn Bidder> = Box::new(
        LlmBidder::with_sink(
            factory.create(&cell.model.tag),
            condition,
            seed,
            Rc::clone(&transcripts),
        )
        .with_trial_index(trial_index),
    );
    let p2: Box<dyn Bidder> = if cell.kind == "llm_vs_solver" {
        Box::new(SolverBidder::new())
    } else {
        Box::new(
            LlmBidder::with_sink(
                factory.create(&cell.model.tag),
                condition,
                seed,
                Rc::clone(&transcripts),
            )
            .with_trial_index(trial_index),
        )
    };
    let captured = Rc::new(RefCell::new(Vec::new()));
    let recorder = CapturingRecorder {
        events: Rc::clone(&captured),
    };
    let result = run_trial_with_recorder(
        TrialRunConfig {
            s: config.s,
            b: cell.b,
            display_paid_so_far: false,
            engine_seed: seed,
            paraphrase_id: condition.paraphrase,
            max_bids: Some(config.max_bids),
        },
        p1,
        p2,
        Box::new(recorder),
    )?;
    let events = captured.borrow_mut().drain(..).collect();
    let transcript_rows = transcripts.borrow_mut().drain(..).collect();
    Ok(BufferedTrial {
        trial_index,
        result,
        events,
        transcripts: transcript_rows,
    })
}

struct ProgressHeartbeat {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl ProgressHeartbeat {
    fn start(
        parent_dir: PathBuf,
        total: usize,
        started: Instant,
        completed_trials: Arc<AtomicUsize>,
        completed_calls: Arc<AtomicUsize>,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let handle = thread::spawn(move || loop {
            thread::park_timeout(Duration::from_secs(30));
            if thread_stop.load(Ordering::Relaxed) {
                break;
            }
            let completed = completed_trials.load(Ordering::Relaxed);
            let calls = completed_calls.load(Ordering::Relaxed);
            let elapsed = started.elapsed();
            let eta = if completed == 0 {
                "unknown".to_string()
            } else {
                format_duration(
                    elapsed.as_secs_f64() / completed as f64
                        * total.saturating_sub(completed) as f64,
                )
            };
            let line = format!(
                "progress: trials {completed}/{total} calls={calls} elapsed={} ETA={eta}",
                format_duration(elapsed.as_secs_f64())
            );
            let _ = write_progress(&parent_dir, &line);
        });
        Self {
            stop,
            thread: Some(handle),
        }
    }

    fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.thread.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
    }
}

impl Drop for ProgressHeartbeat {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.thread.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
    }
}

fn log_cell_metrics(
    run: &mut Run,
    results: &[TrialResult],
    usage: UsageSummary,
) -> Result<(), String> {
    let n = results.len() as f64;
    let mut turns: Vec<_> = results.iter().map(|result| result.bids.len()).collect();
    turns.sort_unstable();
    let median = if turns.len() % 2 == 0 {
        (turns[turns.len() / 2 - 1] + turns[turns.len() / 2]) as f64 / 2.0
    } else {
        turns[turns.len() / 2] as f64
    };
    let mut metrics = vec![
        ("n_trials", n),
        (
            "first_end_rate",
            results.iter().filter(|result| result.first_end).count() as f64 / n,
        ),
        (
            "mean_x1_dev",
            results
                .iter()
                .map(|result| result.first_bid_deviation as f64)
                .sum::<f64>()
                / n,
        ),
        ("median_t", median),
        (
            "invalid_rate",
            results
                .iter()
                .filter(|result| result.outcome == "invalid")
                .count() as f64
                / n,
        ),
        (
            "waste_mean",
            results.iter().map(|result| result.waste).sum::<f64>() / n,
        ),
    ];
    metrics.extend(usage.metrics());
    run.log_metrics("run", &metrics)
        .map_err(|error| error.to_string())
}

#[derive(Debug)]
struct CapturedEvent {
    t: u64,
    kind: String,
    payload: Value,
}

struct CapturingRecorder {
    events: Rc<RefCell<Vec<CapturedEvent>>>,
}

impl Recorder for CapturingRecorder {
    fn record_metric(&mut self, _t: u64, _key: &str, _value: f64) {}

    fn record_event(&mut self, t: u64, kind: &str, payload: Value) {
        self.events.borrow_mut().push(CapturedEvent {
            t,
            kind: kind.to_string(),
            payload,
        });
    }

    fn record_row(&mut self, _t: u64, _table: &str, _row: &[(&str, f64)]) {}
}

fn write_captured_event(
    run: &mut Run,
    event: CapturedEvent,
    trial_index: usize,
    s: u32,
    cell: &PlannedCell,
) -> Result<(), String> {
    let Value::Object(mut payload) = event.payload else {
        return Err("captured event payload is not an object".to_string());
    };
    payload.insert("unit_id".to_string(), json!(format!("trial-{trial_index}")));
    payload.insert("trial".to_string(), json!(trial_index));
    payload.insert("t".to_string(), json!(event.t));
    payload.insert("t_unit".to_string(), json!("turn"));
    payload.insert("s".to_string(), json!(s));
    payload.insert("b".to_string(), json!(cell.b));
    payload.insert("model".to_string(), json!(cell.model.tag));
    payload.insert("framing".to_string(), json!(cell.framing.as_str()));
    payload.insert("opponent".to_string(), json!(cell.opponent.as_str()));
    payload.insert("cell_kind".to_string(), json!(cell.kind));
    let kind = if matches!(event.kind.as_str(), "terminal" | "observation")
        || event.kind.starts_with("x.")
    {
        event.kind
    } else {
        format!("x.dollar-auction-escalation.{}", event.kind)
    };
    run.log_event(&kind, &Value::Object(payload))
        .map_err(|error| error.to_string())
}

fn write_transcripts(run_dir: &Path, transcripts: &[Transcript]) -> Result<(), String> {
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

fn write_progress(parent_dir: &Path, line: &str) -> Result<(), String> {
    eprintln!("{line}");
    let log_dir = parent_dir.join("logs");
    fs::create_dir_all(&log_dir).map_err(|error| error.to_string())?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_dir.join("progress.log"))
        .map_err(|error| error.to_string())?;
    writeln!(file, "{line}").map_err(|error| error.to_string())?;
    file.flush().map_err(|error| error.to_string())
}

fn read_existing_children(parent: &Path) -> Result<Vec<ExistingChild>, String> {
    let parent_meta = read_json(parent.join("run.json"))?;
    let parent_uid = parent_meta
        .get("run_uid")
        .and_then(Value::as_str)
        .ok_or_else(|| "resume parent has no run_uid".to_string())?;
    let experiment_dir = parent
        .parent()
        .ok_or_else(|| "resume parent has no experiment directory".to_string())?;
    let mut children = Vec::new();
    for entry in fs::read_dir(experiment_dir).map_err(|error| error.to_string())? {
        let path = entry.map_err(|error| error.to_string())?.path();
        if !path.is_dir() || !path.join("run.json").exists() {
            continue;
        }
        let meta = read_json(path.join("run.json"))?;
        if meta
            .pointer("/lineage/parent_run_uid")
            .and_then(Value::as_str)
            != Some(parent_uid)
        {
            continue;
        }
        let Some(config_hash) = meta.get("config_hash").and_then(Value::as_str) else {
            continue;
        };
        let status = read_json(path.join("status.json"))?;
        let state = status
            .get("state")
            .and_then(Value::as_str)
            .unwrap_or("failed");
        children.push(ExistingChild::new(config_hash, state, Vec::new()));
    }
    Ok(children)
}

fn cell_parameters(config: &PilotConfig, cell: &PlannedCell, model_digest: &str) -> Value {
    json!({
        "command": "pilot-cell",
        "cell_key": cell.key(),
        "model": cell.model.tag,
        "model_digest": model_digest,
        "model_digest_prefix": cell.model.digest_prefix,
        "s": config.s,
        "b": cell.b,
        "kind": cell.kind,
        "framing": cell.framing.as_str(),
        "opponent": cell.opponent.as_str(),
        "calculator": cell.calculator,
        "trials": config.trials,
        "temperature": config.temperature,
        "think": false,
        "root_seed": config.root_seed,
        "max_bids": config.max_bids,
        "concurrency": config.concurrency,
    })
}

fn cell_config_hash(
    config: &PilotConfig,
    cell: &PlannedCell,
    model_digest: &str,
) -> Result<String, String> {
    let parameters = cell_parameters(config, cell, model_digest);
    let block = RunvaultBlock {
        seed_pointers: vec!["/root_seed".to_string()],
        ..Default::default()
    };
    let exclusions = Exclusions::resolve(&block, &parameters).map_err(|error| error.to_string())?;
    runvault::hash::config_hash(&parameters, &exclusions, &[]).map_err(|error| error.to_string())
}

fn read_json(path: impl AsRef<Path>) -> Result<Value, String> {
    let path = path.as_ref();
    serde_json::from_str(
        &fs::read_to_string(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("failed to parse {}: {error}", path.display()))
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
