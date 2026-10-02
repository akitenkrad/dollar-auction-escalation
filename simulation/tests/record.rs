use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use dollar_auction_simulation::bidder::{LlmBidder, SolverBidder};
use dollar_auction_simulation::prompt::PromptCondition;
use dollar_auction_simulation::record::{record_play, RecordConfig};
use dollar_auction_simulation::simulation::empty_transcript_sink;
use socsim_llm::mock::ScriptedClient;

fn temp_dir(label: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "dollar_auction_{label}_{}_{}",
        std::process::id(),
        n
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn run_play(root: &Path, p1: &str, p2: &str, s: u32, b: u32) -> PathBuf {
    let output = Command::new(env!("CARGO_BIN_EXE_dollar-auction"))
        .args([
            "play",
            "--p1",
            p1,
            "--p2",
            p2,
            "--s",
            &s.to_string(),
            "--b",
            &b.to_string(),
            "--scratch",
            "--results-dir",
        ])
        .arg(root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    PathBuf::from(
        stdout
            .lines()
            .find_map(|line| line.strip_prefix("Run directory: "))
            .unwrap(),
    )
}

fn events(run: &Path) -> Vec<serde_json::Value> {
    fs::read_to_string(run.join("events.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn scripted_scratch_run_records_terminal_trial_violation_and_transcript() {
    let root = temp_dir("scripted");
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/invalid_then_invalid.jsonl");
    let p1 = format!("scripted:{}", fixture.display());
    let run = run_play(&root, &p1, "solver", 10, 12);
    runvault::verify::deep(&run).unwrap();
    assert!(run.components().any(|c| c.as_os_str() == "_scratch"));
    let rows = events(&run);
    let terminal = rows.iter().find(|e| e["schema"] == "terminal").unwrap();
    assert_eq!(terminal["outcome"], "invalid");
    assert_eq!(terminal["censored"], false);
    assert_eq!(terminal["budget"], 12);
    assert!(terminal["h"].is_number());
    assert!(terminal["turn"].is_number());
    assert!(rows
        .iter()
        .any(|e| e["schema"] == "x.dollar-auction-escalation.violation"));
    assert!(rows
        .iter()
        .any(|e| e["schema"] == "x.dollar-auction-escalation.trial"));

    let metrics = fs::read_to_string(run.join("metrics.csv")).unwrap();
    assert!(metrics.contains("n_trials"));
    assert!(!metrics.contains("first_bid_deviation"));

    let transcript_lines: Vec<_> = fs::read_to_string(run.join("artifacts/transcripts.jsonl"))
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    assert_eq!(transcript_lines.len(), 2);
    let transcript: serde_json::Value = serde_json::from_str(&transcript_lines[0]).unwrap();
    for key in [
        "system",
        "user",
        "raw_response",
        "thinking",
        "seed",
        "token_usage",
        "think",
        "fenced",
    ] {
        assert!(transcript.get(key).is_some(), "missing {key}");
    }
    let config: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(run.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["parameters"]["think"], false);
}

#[test]
fn cap_terminal_uses_budget_equal_to_t_when_cap_is_reached_early() {
    let root = temp_dir("cap");
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let p1 = format!("scripted:{}", fixtures.join("bid_three.jsonl").display());
    let p2 = format!("scripted:{}", fixtures.join("drop.jsonl").display());
    let run = run_play(&root, &p1, &p2, 3, 3);
    runvault::verify::deep(&run).unwrap();
    let rows = events(&run);
    let terminal = rows.iter().find(|e| e["schema"] == "terminal").unwrap();
    assert_eq!(terminal["censored"], true);
    assert_eq!(terminal["budget"], terminal["t"]);
    assert_eq!(terminal["t"], 1);
    assert_eq!(terminal["outcome"], "p1_win");
}

#[test]
fn fenced_calls_and_think_setting_are_recorded() {
    let root = temp_dir("fenced");
    let transcripts = empty_transcript_sink();
    let client = ScriptedClient::constant(
        "scripted",
        "```json\n{\"action\":\"drop\",\"amount\":0,\"reason\":\"done\"}\n```",
    );
    let p1 = Box::new(LlmBidder::with_sink(
        Box::new(client),
        PromptCondition::default(),
        7,
        transcripts.clone(),
    ));
    let config = RecordConfig {
        s: 10,
        b: 12,
        p1: "scripted:test-double".to_string(),
        p2: "solver".to_string(),
        framing: "named".to_string(),
        opponent: "rational_ai".to_string(),
        paraphrase: 0,
        calculator: false,
        display_paid_so_far: false,
        root_seed: 42,
        trial_seed: 7,
    };
    let (run, _) = record_play(
        &config,
        p1,
        Box::new(SolverBidder::new()),
        transcripts,
        &root,
        true,
    )
    .unwrap();
    runvault::verify::deep(&run).unwrap();

    let rows = events(&run);
    let trial = rows
        .iter()
        .find(|event| event["schema"] == "x.dollar-auction-escalation.trial")
        .unwrap();
    assert_eq!(trial["fenced_calls"], 1);
    let transcript: serde_json::Value = serde_json::from_str(
        fs::read_to_string(run.join("artifacts/transcripts.jsonl"))
            .unwrap()
            .trim(),
    )
    .unwrap();
    assert_eq!(transcript["fenced"], true);
    assert_eq!(transcript["think"], false);
}
