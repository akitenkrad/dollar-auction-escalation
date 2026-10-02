use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use runvault::{Run, RunOptions};
use runvault_socsim::RunvaultRecorder;
use serde_json::json;

use crate::bidder::{Bidder, TranscriptSink};
use crate::simulation::{run_trial_with_recorder, TrialResult, TrialRunConfig};

#[derive(Debug, Clone)]
pub struct RecordConfig {
    pub s: u32,
    pub b: u32,
    pub p1: String,
    pub p2: String,
    pub framing: String,
    pub opponent: String,
    pub paraphrase: u8,
    pub calculator: bool,
    pub display_paid_so_far: bool,
    pub root_seed: u64,
    pub trial_seed: u64,
}

pub fn record_play(
    config: &RecordConfig,
    p1: Box<dyn Bidder>,
    p2: Box<dyn Bidder>,
    transcripts: TranscriptSink,
    results_root: &Path,
    scratch: bool,
) -> Result<(PathBuf, TrialResult), String> {
    if !scratch {
        return Err("play requires --scratch in Phase 1".to_string());
    }
    let parameters = json!({
        "s": config.s,
        "b": config.b,
        "p1": config.p1,
        "p2": config.p2,
        "framing": config.framing,
        "opponent": config.opponent,
        "paraphrase": config.paraphrase,
        "calculator": config.calculator,
        "display_paid_so_far": config.display_paid_so_far,
        "root_seed": config.root_seed,
        "trial_seed": config.trial_seed,
        "temperature": crate::config::DEFAULT_TEMPERATURE,
        "think": crate::llm::MODEL_THINK,
        "command": "play",
    });
    let run = Run::start(
        RunOptions::new("dollar-auction", "play")
            .repo_id("dollar-auction-escalation")
            .domain("simulation")
            .results_root(results_root)
            .scratch(true)
            .parameters(&parameters)
            .map_err(|error| error.to_string())?
            .seed_pointers(["/trial_seed", "/root_seed"])
            .master_seed(config.trial_seed),
    )
    .map_err(|error| error.to_string())?;
    let recorder = RunvaultRecorder::new(run).step_unit("turn").scope("run");
    let handle = recorder.handle();
    let run_dir = handle
        .dir()
        .ok_or_else(|| "run directory is unavailable".to_string())?;
    let result = match run_trial_with_recorder(
        TrialRunConfig {
            s: config.s,
            b: config.b,
            display_paid_so_far: config.display_paid_so_far,
            engine_seed: config.trial_seed,
            paraphrase_id: config.paraphrase,
        },
        p1,
        p2,
        Box::new(recorder),
    ) {
        Ok(result) => result,
        Err(error) => {
            let _ = handle.fail("simulation", &error);
            return Err(error);
        }
    };

    write_transcripts(&run_dir, &transcripts)?;
    handle
        .with_run(|run| log_aggregates(run, &result))
        .map_err(|error| error.to_string())??;
    let finished = handle.finish().map_err(|error| error.to_string())?;
    Ok((finished, result))
}

fn log_aggregates(run: &mut Run, result: &TrialResult) -> Result<(), String> {
    run.log_metrics(
        "run",
        &[
            ("n_trials", 1.0),
            ("first_end_rate", if result.first_end { 1.0 } else { 0.0 }),
            ("mean_x1_dev", result.first_bid_deviation as f64),
            ("median_t", result.bids.len() as f64),
            (
                "invalid_rate",
                if result.outcome == "invalid" {
                    1.0
                } else {
                    0.0
                },
            ),
            ("waste_mean", result.waste),
        ],
    )
    .map_err(|error| error.to_string())
}

fn write_transcripts(run_dir: &Path, transcripts: &TranscriptSink) -> Result<(), String> {
    let artifacts = run_dir.join("artifacts");
    fs::create_dir_all(&artifacts).map_err(|error| error.to_string())?;
    let file =
        File::create(artifacts.join("transcripts.jsonl")).map_err(|error| error.to_string())?;
    let mut writer = BufWriter::new(file);
    for transcript in transcripts.borrow().iter() {
        serde_json::to_writer(&mut writer, transcript).map_err(|error| error.to_string())?;
        writer.write_all(b"\n").map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}
