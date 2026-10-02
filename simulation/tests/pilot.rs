use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use dollar_auction_simulation::llm::TagsFetcher;
use dollar_auction_simulation::pilot::{
    plan_cells, resume_decision, run_pilot, ExistingChild, LlmClientFactory, PilotConfig,
    ProgressState, ResumeDecision,
};
use socsim_llm::mock::ScriptedClient;
use socsim_llm::LlmClient;

fn temp_dir(label: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "dollar_auction_phase2_{label}_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

struct OkTags;
impl TagsFetcher for OkTags {
    fn fetch_tags(&self, _host: &str) -> Result<String, String> {
        Ok(r#"{"models":[{"name":"gemma4:26b","digest":"001e5dafc3c7ffff"}]}"#.to_string())
    }
}

struct ScriptedFactory {
    calls: Arc<AtomicUsize>,
}

struct ConcurrencyFactory {
    active: Arc<AtomicUsize>,
    maximum: Arc<AtomicUsize>,
}

impl LlmClientFactory for ConcurrencyFactory {
    fn create(&self, model: &str) -> Box<dyn LlmClient> {
        let active = Arc::clone(&self.active);
        let maximum = Arc::clone(&self.maximum);
        Box::new(ScriptedClient::new(model, move |prompt| {
            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
            maximum.fetch_max(now, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(20));
            active.fetch_sub(1, Ordering::SeqCst);
            if prompt.contains("withdraw") {
                r#"{"action":"withdraw","reason":"stop"}"#.to_string()
            } else {
                r#"{"action":"drop","reason":"stop"}"#.to_string()
            }
        }))
    }
}
impl LlmClientFactory for ScriptedFactory {
    fn create(&self, model: &str) -> Box<dyn LlmClient> {
        let calls = Arc::clone(&self.calls);
        Box::new(ScriptedClient::new(model, move |prompt| {
            calls.fetch_add(1, Ordering::SeqCst);
            if prompt.contains("withdraw") {
                r#"{"action":"withdraw","reason":"stop"}"#.to_string()
            } else {
                r#"{"action":"drop","reason":"stop"}"#.to_string()
            }
        }))
    }
}

#[test]
fn full_plan_has_model_major_order_and_expected_cells() {
    let config =
        PilotConfig::from_path(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../pilot.toml"))
            .unwrap();
    let cells = plan_cells(&config).unwrap();
    assert_eq!(config.max_bids, 50);
    // Measured 2026-10-02: gemma4:26b throughput rises ~1.8x at 2 concurrent requests and not further at 4.
    assert_eq!(config.concurrency, 2);
    assert_eq!(cells.len(), 32);
    assert!(cells[..16]
        .iter()
        .all(|cell| cell.model.tag == "gemma4:26b"));
    assert!(cells[16..]
        .iter()
        .all(|cell| cell.model.tag == "olmo-3.1:32b-instruct"));
    for model in &config.models {
        let model_cells: Vec<_> = cells.iter().filter(|cell| cell.model == *model).collect();
        assert_eq!(model_cells.len(), 16);
        assert_eq!(
            model_cells
                .iter()
                .filter(|cell| cell.kind == "main")
                .count(),
            12
        );
        assert_eq!(
            model_cells
                .iter()
                .filter(|cell| cell.kind == "llm_vs_solver")
                .count(),
            2
        );
        assert_eq!(
            model_cells
                .iter()
                .filter(|cell| cell.kind == "calculator")
                .count(),
            2
        );
        assert!(model_cells
            .iter()
            .filter(|cell| cell.kind != "main")
            .all(|cell| cell.framing.as_str() == "disguised"));
    }
}

fn normalized_events(run: &std::path::Path) -> Vec<serde_json::Value> {
    fs::read_to_string(run.join("events.jsonl"))
        .unwrap()
        .lines()
        .map(|line| {
            let mut event: serde_json::Value = serde_json::from_str(line).unwrap();
            let object = event.as_object_mut().unwrap();
            object.remove("ts");
            object.remove("run_uid");
            event
        })
        .collect()
}

fn normalized_metrics(run: &std::path::Path) -> Vec<String> {
    fs::read_to_string(run.join("metrics.csv"))
        .unwrap()
        .lines()
        .map(|line| {
            line.split_once(',')
                .map_or(line, |(_, rest)| rest)
                .to_string()
        })
        .collect()
}

#[test]
fn trial_concurrency_preserves_sorted_deterministic_recording() {
    let template = PilotConfig::from_path(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../pilot.smoke.toml"),
    )
    .unwrap();
    let mut serial = template.clone();
    serial.cells.truncate(1);
    serial.trials = 4;
    serial.concurrency = 1;
    let mut parallel = serial.clone();
    parallel.concurrency = 4;

    let serial_factory = ConcurrencyFactory {
        active: Arc::new(AtomicUsize::new(0)),
        maximum: Arc::new(AtomicUsize::new(0)),
    };
    let parallel_maximum = Arc::new(AtomicUsize::new(0));
    let parallel_factory = ConcurrencyFactory {
        active: Arc::new(AtomicUsize::new(0)),
        maximum: Arc::clone(&parallel_maximum),
    };
    let serial_outcome = run_pilot(
        &serial,
        &temp_dir("serial"),
        true,
        None,
        &OkTags,
        &serial_factory,
    )
    .unwrap();
    let parallel_outcome = run_pilot(
        &parallel,
        &temp_dir("parallel"),
        true,
        None,
        &OkTags,
        &parallel_factory,
    )
    .unwrap();
    assert!(parallel_maximum.load(Ordering::SeqCst) >= 2);

    let serial_child = &serial_outcome.child_dirs[0];
    let parallel_child = &parallel_outcome.child_dirs[0];
    assert_eq!(
        normalized_events(serial_child),
        normalized_events(parallel_child)
    );
    assert_eq!(
        normalized_metrics(serial_child),
        normalized_metrics(parallel_child)
    );
    let trials: Vec<u64> = normalized_events(parallel_child)
        .iter()
        .filter_map(|event| event.get("trial").and_then(serde_json::Value::as_u64))
        .collect();
    assert!(trials.windows(2).all(|pair| pair[0] <= pair[1]));
    assert_eq!(
        fs::read_to_string(serial_child.join("artifacts/transcripts.jsonl")).unwrap(),
        fs::read_to_string(parallel_child.join("artifacts/transcripts.jsonl")).unwrap()
    );
}

#[test]
fn progress_is_rate_limited_but_cells_force_a_line_with_eta_and_calls() {
    let mut progress = ProgressState::new(90);
    assert!(progress.update(0, 0, Duration::ZERO, false).is_some());
    assert!(progress
        .update(1, 2, Duration::from_secs(5), false)
        .is_none());
    let timed = progress
        .update(2, 7, Duration::from_secs(31), false)
        .unwrap();
    assert!(timed.contains("2/90"));
    assert!(timed.contains("calls=7"));
    assert!(timed.contains("elapsed="));
    assert!(timed.contains("ETA="));
    assert!(progress
        .update(3, 9, Duration::from_secs(32), true)
        .is_some());
}

#[test]
fn resume_skips_only_finished_matching_cells_and_restarts_failed_at_zero() {
    let existing = [
        ExistingChild::new("a", "finished", vec![0, 1, 2]),
        ExistingChild::new("b", "failed", vec![0, 1]),
        ExistingChild::new("old-c", "finished", vec![0, 1, 2]),
    ];
    assert_eq!(
        resume_decision("a", &existing),
        ResumeDecision::SkipFinished
    );
    assert_eq!(
        resume_decision("b", &existing),
        ResumeDecision::RunFromTrialZero
    );
    assert_eq!(
        resume_decision("c", &existing),
        ResumeDecision::RunFromTrialZero
    );
}

#[test]
fn scripted_smoke_pilot_writes_parent_children_metrics_and_linkage() {
    let root = temp_dir("pilot");
    let config = PilotConfig::from_path(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../pilot.smoke.toml"),
    )
    .unwrap();
    assert_eq!(config.max_bids, 50);
    assert_eq!(config.concurrency, 2);
    let calls = Arc::new(AtomicUsize::new(0));
    let factory = ScriptedFactory {
        calls: Arc::clone(&calls),
    };
    let outcome = run_pilot(&config, &root, true, None, &OkTags, &factory).unwrap();
    assert!(outcome
        .parent_dir
        .components()
        .any(|part| part.as_os_str() == "_scratch"));
    assert_eq!(outcome.child_dirs.len(), 2);
    assert_eq!(calls.load(Ordering::SeqCst), 6);

    let parent: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(outcome.parent_dir.join("run.json")).unwrap())
            .unwrap();
    for child in &outcome.child_dirs {
        runvault::verify::deep(child).unwrap();
        let run: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(child.join("run.json")).unwrap()).unwrap();
        assert_eq!(
            run["lineage"]["parent_run_uid"],
            parent["run_uid"],
            "child={}",
            child.display()
        );
        let metrics = fs::read_to_string(child.join("metrics.csv")).unwrap();
        assert!(metrics.contains("n_trials"));
        assert!(metrics.contains("n_calls"));
        assert!(!metrics.contains("cost_usd"));
        let config: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(child.join("config.json")).unwrap()).unwrap();
        assert_eq!(config["parameters"]["model_digest"], "001e5dafc3c7ffff");
        let events = fs::read_to_string(child.join("events.jsonl")).unwrap();
        assert_eq!(
            events
                .lines()
                .filter(|line| line.contains("\"schema\":\"terminal\""))
                .count(),
            3
        );
    }
    let progress = fs::read_to_string(outcome.parent_dir.join("logs/progress.log")).unwrap();
    assert!(progress.contains("ETA="));
}
