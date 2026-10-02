use crate::bidder::Response;
use crate::qre::{ObservedAction, QreTree};
use runvault::{Lineage, Run, RunOptions};
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitObservation {
    Valid(ObservedAction),
    Invalid,
    Violation,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QreFit {
    pub lambda_hat: f64,
    pub lambda_low: f64,
    pub lambda_high: f64,
    pub log_likelihood: f64,
    pub n_included: usize,
    pub n_excluded: usize,
}

pub fn fit_qre(s: u32, b: u32, observations: &[FitObservation]) -> QreFit {
    let included: Vec<_> = observations
        .iter()
        .filter_map(|observation| match observation {
            FitObservation::Valid(action) => Some(action.clone()),
            FitObservation::Invalid | FitObservation::Violation => None,
        })
        .collect();
    let n_excluded = observations.len() - included.len();
    if included.is_empty() {
        return QreFit {
            lambda_hat: 0.0,
            lambda_low: 0.0,
            lambda_high: 50.0,
            log_likelihood: 0.0,
            n_included: 0,
            n_excluded,
        };
    }

    let log_likelihood = |lambda: f64| QreTree::new(s, b, lambda).log_likelihood(&included);
    let mut grid = vec![0.0];
    let min_positive: f64 = 1e-4;
    let ratio = (50.0_f64 / min_positive).powf(1.0 / 60.0);
    let mut value = min_positive;
    for _ in 0..=60 {
        grid.push(value.min(50.0));
        value *= ratio;
    }
    if *grid.last().unwrap() < 50.0 {
        grid.push(50.0);
    }
    let evaluated: Vec<_> = grid
        .iter()
        .map(|&lambda| (lambda, log_likelihood(lambda)))
        .collect();
    let best_index = evaluated
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.1.total_cmp(&right.1))
        .map(|(index, _)| index)
        .unwrap();
    let lower = if best_index == 0 {
        0.0
    } else {
        evaluated[best_index - 1].0
    };
    let upper = evaluated.get(best_index + 1).map_or(50.0, |entry| entry.0);
    let lambda_hat = golden_max(lower, upper, &log_likelihood);
    let maximum = log_likelihood(lambda_hat);
    let cutoff = maximum - 1.92;
    let lambda_low = profile_boundary(0.0, lambda_hat, cutoff, &log_likelihood, true);
    let lambda_high = profile_boundary(50.0, lambda_hat, cutoff, &log_likelihood, false);
    QreFit {
        lambda_hat,
        lambda_low,
        lambda_high,
        log_likelihood: maximum,
        n_included: included.len(),
        n_excluded,
    }
}

pub fn observations_from_events(events: &[serde_json::Value]) -> Vec<FitObservation> {
    events
        .iter()
        .filter_map(
            |event| match event.get("schema").and_then(serde_json::Value::as_str) {
                Some("x.dollar-auction-escalation.decision") => {
                    if event.get("is_llm").and_then(serde_json::Value::as_bool) == Some(false) {
                        return None;
                    }
                    let me = event
                        .get("me")?
                        .as_u64()
                        .and_then(|value| u32::try_from(value).ok())?;
                    let opp = event
                        .get("opp")?
                        .as_u64()
                        .and_then(|value| u32::try_from(value).ok())?;
                    let action = match event.get("action")?.as_str()? {
                        "drop" => Response::Drop,
                        "bid" => Response::Bid(
                            event
                                .get("amount")?
                                .as_u64()
                                .and_then(|value| u32::try_from(value).ok())?,
                        ),
                        _ => return None,
                    };
                    Some(FitObservation::Valid(ObservedAction::new(me, opp, action)))
                }
                Some("x.dollar-auction-escalation.violation") => Some(FitObservation::Violation),
                Some("terminal")
                    if event.get("outcome").and_then(serde_json::Value::as_str)
                        == Some("invalid") =>
                {
                    Some(FitObservation::Invalid)
                }
                _ => None,
            },
        )
        .collect()
}

fn golden_max(mut left: f64, mut right: f64, f: &impl Fn(f64) -> f64) -> f64 {
    let ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut x1 = right - ratio * (right - left);
    let mut x2 = left + ratio * (right - left);
    let mut y1 = f(x1);
    let mut y2 = f(x2);
    for _ in 0..40 {
        if y1 < y2 {
            left = x1;
            x1 = x2;
            y1 = y2;
            x2 = left + ratio * (right - left);
            y2 = f(x2);
        } else {
            right = x2;
            x2 = x1;
            y2 = y1;
            x1 = right - ratio * (right - left);
            y1 = f(x1);
        }
    }
    (left + right) / 2.0
}

fn profile_boundary(
    outer: f64,
    maximum: f64,
    cutoff: f64,
    f: &impl Fn(f64) -> f64,
    lower: bool,
) -> f64 {
    if f(outer) >= cutoff {
        return outer;
    }
    let (mut below, mut above) = (outer, maximum);
    for _ in 0..40 {
        let midpoint = (below + above) / 2.0;
        if f(midpoint) < cutoff {
            below = midpoint;
        } else {
            above = midpoint;
        }
    }
    if lower {
        above
    } else {
        below
    }
}

pub fn run_qre_fit(input: &Path, results_root: &Path, scratch: bool) -> Result<PathBuf, String> {
    let input_meta = read_json(input.join("run.json"))?;
    let input_uid = input_meta
        .get("run_uid")
        .and_then(Value::as_str)
        .ok_or_else(|| "input run has no run_uid".to_string())?
        .to_string();
    let mut sources = child_runs(input, &input_uid)?;
    if sources.is_empty() {
        sources.push(input.to_path_buf());
    }
    let mut rows = Vec::new();
    let mut input_uids = Vec::new();
    for source in &sources {
        let meta = read_json(source.join("run.json"))?;
        let uid = meta
            .get("run_uid")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{} has no run_uid", source.display()))?
            .to_string();
        let config = read_json(source.join("config.json"))?;
        let parameters = config
            .get("parameters")
            .ok_or_else(|| format!("{} has no parameters", source.display()))?;
        let s = parameters.get("s").and_then(Value::as_u64).unwrap_or(100) as u32;
        let b = parameters
            .get("b")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("{} has no budget", source.display()))? as u32;
        let events = read_json_lines(source.join("events.jsonl"))?;
        let observations = observations_from_events(&events);
        let fit = fit_qre(s, b, &observations);
        rows.push(FitRow {
            run_uid: uid.clone(),
            model: text_parameter(parameters, "model"),
            kind: text_parameter(parameters, "kind"),
            framing: text_parameter(parameters, "framing"),
            opponent: text_parameter(parameters, "opponent"),
            b,
            fit,
        });
        input_uids.push(uid);
    }
    let parameters = json!({
        "command": "fit-qre",
        "input_run_uids": input_uids,
        "input_path": input.display().to_string(),
    });
    let run = Run::start(
        RunOptions::new("dollar-auction", "fit-qre")
            .scratch(scratch)
            .repo_id("dollar-auction-escalation")
            .domain("analysis")
            .results_root(results_root)
            .parameters(&parameters)
            .map_err(|error| error.to_string())?
            .lineage(Lineage {
                derived_from: Some(input_uid),
                ..Default::default()
            }),
    )
    .map_err(|error| error.to_string())?;
    write_fit_csv(run.dir(), &rows)?;
    run.finish().map_err(|error| error.to_string())
}

struct FitRow {
    run_uid: String,
    model: String,
    kind: String,
    framing: String,
    opponent: String,
    b: u32,
    fit: QreFit,
}

fn write_fit_csv(run_dir: &Path, rows: &[FitRow]) -> Result<(), String> {
    let artifacts = run_dir.join("artifacts");
    fs::create_dir_all(&artifacts).map_err(|error| error.to_string())?;
    let mut writer = BufWriter::new(
        File::create(artifacts.join("qre_fit.csv")).map_err(|error| error.to_string())?,
    );
    writeln!(
        writer,
        "input_run_uid,model,kind,framing,opponent,b,lambda_hat,lambda_low,lambda_high,log_likelihood,n_included,n_excluded"
    )
    .map_err(|error| error.to_string())?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            row.run_uid,
            row.model,
            row.kind,
            row.framing,
            row.opponent,
            row.b,
            row.fit.lambda_hat,
            row.fit.lambda_low,
            row.fit.lambda_high,
            row.fit.log_likelihood,
            row.fit.n_included,
            row.fit.n_excluded
        )
        .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

fn child_runs(parent: &Path, parent_uid: &str) -> Result<Vec<PathBuf>, String> {
    let Some(directory) = parent.parent() else {
        return Ok(Vec::new());
    };
    let mut children = Vec::new();
    for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
        let path = entry.map_err(|error| error.to_string())?.path();
        if !path.is_dir() || !path.join("run.json").exists() {
            continue;
        }
        let meta = read_json(path.join("run.json"))?;
        if meta
            .pointer("/lineage/parent_run_uid")
            .and_then(Value::as_str)
            == Some(parent_uid)
        {
            children.push(path);
        }
    }
    children.sort();
    Ok(children)
}

fn read_json(path: impl AsRef<Path>) -> Result<Value, String> {
    let path = path.as_ref();
    serde_json::from_str(
        &fs::read_to_string(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("failed to parse {}: {error}", path.display()))
}

fn read_json_lines(path: impl AsRef<Path>) -> Result<Vec<Value>, String> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    text.lines()
        .map(|line| serde_json::from_str(line).map_err(|error| error.to_string()))
        .collect()
}

fn text_parameter(parameters: &Value, name: &str) -> String {
    parameters
        .get(name)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}
