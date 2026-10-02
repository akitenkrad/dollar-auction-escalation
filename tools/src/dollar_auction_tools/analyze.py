"""Pre-registered analysis for dollar-auction pilot sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
import statsmodels.api as sm
from runvault.read import (
    config_parameters,
    events_table,
    load_run_meta,
    run_scope_metrics,
    sweep_children,
)


def _input_label(run_uids: Iterable[str]) -> str:
    return ";".join(sorted(set(run_uids)))


def sweep_sources(sweep: str | Path) -> tuple[list[Path], list[str]]:
    """Return finished child runs across a newest-to-oldest resume chain."""
    current = Path(sweep).resolve()
    seen_parents: set[str] = set()
    seen_configs: set[str] = set()
    children: list[Path] = []
    parent_uids: list[str] = []
    while True:
        meta = load_run_meta(current)
        uid = str(meta["run_uid"])
        if uid in seen_parents:
            raise ValueError("resume lineage contains a cycle")
        seen_parents.add(uid)
        parent_uids.append(uid)
        for raw_child in reversed(sweep_children(current)):
            child = Path(raw_child)
            status_path = child / "status.json"
            if status_path.exists():
                state = json.loads(status_path.read_text()).get("state")
                if state != "finished":
                    continue
            child_meta = load_run_meta(child)
            identity = str(child_meta.get("config_hash") or child_meta["run_uid"])
            if identity not in seen_configs:
                children.append(child)
                seen_configs.add(identity)
        prior_uid = (meta.get("lineage") or {}).get("resumed_from")
        if not prior_uid:
            break
        prior = None
        for candidate in current.parent.iterdir():
            if not candidate.is_dir():
                continue
            candidate_meta = load_run_meta(candidate, required=False)
            if candidate_meta and candidate_meta.get("run_uid") == prior_uid:
                prior = candidate
                break
        if prior is None:
            raise ValueError(f"resumed parent run_uid not found: {prior_uid}")
        current = prior
    children.reverse()
    return children, parent_uids


def _with_provenance(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    result = frame.copy()
    result["input_run_uids"] = label
    return result


def _write_csv(frame: pd.DataFrame, path: Path, label: str) -> None:
    _with_provenance(frame, label).to_csv(path, index=False)


def _drop_anchor(s: int, b: int) -> int:
    return (b - 1) % (s - 1) + 1


def _mixed_anchor(s: int, b: int) -> int:
    # These are the registered pilot anchors. Other designs retain the
    # proposition anchor rather than silently inventing a new reference value.
    if s == 100 and b == 250:
        return 50
    return _drop_anchor(s, b)


def _child_trials(child: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = load_run_meta(child)
    params = config_parameters(child)
    all_events = events_table(child, kind=None)
    terminal = all_events[all_events["schema"] == "terminal"].copy().dropna(axis=1, how="all")
    trial = all_events[
        all_events["schema"] == "x.dollar-auction-escalation.trial"
    ].copy().dropna(axis=1, how="all")
    key = ["unit_id"]
    keep = [c for c in trial.columns if c not in terminal.columns or c in key]
    merged = terminal.merge(trial[keep], on=key, how="left", suffixes=("", "_trial"))
    merged["run_uid"] = meta["run_uid"]
    for field in ("model", "framing", "opponent", "b", "s", "max_bids"):
        if field not in merged or merged[field].isna().all():
            merged[field] = params.get(field)
        else:
            merged[field] = merged[field].fillna(params.get(field))
    if "cell_kind" in merged:
        merged["cell_kind"] = merged["cell_kind"].fillna(params.get("kind"))
    elif "kind" in merged:
        merged["cell_kind"] = merged.pop("kind").fillna(params.get("kind"))
    else:
        merged["cell_kind"] = params.get("kind")
    if "T" not in merged:
        merged["T"] = merged["t"]
    if "paraphrase_id" not in merged:
        merged["paraphrase_id"] = 0
    if "turn_capped" not in merged:
        merged["turn_capped"] = merged["outcome"].eq("turn_cap")
    if "waste_is_lower_bound" not in merged:
        merged["waste_is_lower_bound"] = merged["turn_capped"]
    violations = all_events[
        all_events["schema"] == "x.dollar-auction-escalation.violation"
    ].copy()
    if not violations.empty:
        violations["run_uid"] = meta["run_uid"]
        for field in ("model", "framing", "opponent", "b"):
            if field not in violations:
                violations[field] = params.get(field)
        if "cell_kind" in violations:
            violations["cell_kind"] = violations["cell_kind"].fillna(params.get("kind"))
        else:
            violations["cell_kind"] = params.get("kind")
    return merged, violations


def _prepare_trials(children: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_frames: list[pd.DataFrame] = []
    violation_frames: list[pd.DataFrame] = []
    for child in children:
        trials, violations = _child_trials(child)
        trial_frames.append(trials)
        if not violations.empty:
            violation_frames.append(violations)
    trials = pd.concat(trial_frames, ignore_index=True)
    violations = (
        pd.concat(violation_frames, ignore_index=True)
        if violation_frames
        else pd.DataFrame(columns=["run_uid", "kind"])
    )
    trials["invalid"] = trials["outcome"].eq("invalid")
    trials["no_bid"] = trials["outcome"].eq("no_bid")
    trials["logistic_denominator"] = ~trials["no_bid"]
    trials["event_observed"] = ~(trials["censored"].astype(bool) | trials["invalid"])
    trials["duration"] = trials["T"].astype(float)
    trials["x1_star_drop_tie"] = [
        _drop_anchor(int(s), int(b)) for s, b in zip(trials["s"], trials["b"])
    ]
    trials["x1_star_mixed_tie"] = [
        _mixed_anchor(int(s), int(b)) for s, b in zip(trials["s"], trials["b"])
    ]
    trials["first_bid"] = (
        trials["first_bid_deviation"].fillna(0) + trials["x1_star_drop_tie"]
    )
    trials.loc[trials["no_bid"], "first_bid"] = 0
    trials["x1_deviation_drop_tie"] = trials["first_bid"] - trials["x1_star_drop_tie"]
    trials["x1_deviation_mixed_tie"] = trials["first_bid"] - trials["x1_star_mixed_tie"]
    return trials, violations


def _condition_label(row: pd.Series) -> str:
    return f"{row['model']} | {row['cell_kind']} | {row['framing']} | {row['opponent']} | b={int(row['b'])}"


def _km_table(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    groups = ["model", "cell_kind", "framing", "opponent", "b"]
    for key, group in trials.groupby(groups, dropna=False):
        km = KaplanMeierFitter().fit(
            group["duration"], event_observed=group["event_observed"]
        )
        curve = km.survival_function_.reset_index()
        curve.columns = ["time", "survival"]
        for field, value in zip(groups, key):
            curve[field] = value
        rows.append(curve)
    return pd.concat(rows, ignore_index=True)


def _cox_table(trials: pd.DataFrame) -> pd.DataFrame:
    covariates = pd.get_dummies(
        trials[["framing", "opponent", "model"]], drop_first=True, dtype=float
    )
    data = pd.concat(
        [
            trials[["duration", "event_observed", "paraphrase_id"]].reset_index(drop=True),
            trials[["b"]].astype(float).reset_index(drop=True),
            covariates.reset_index(drop=True),
        ],
        axis=1,
    )
    data["event_observed"] = data["event_observed"].astype(bool)
    try:
        fit = CoxPHFitter().fit(
            data,
            duration_col="duration",
            event_col="event_observed",
            cluster_col="paraphrase_id",
            robust=True,
        )
        result = fit.summary.reset_index().rename(columns={"covariate": "term"})
        return result
    except Exception as error:  # Small smoke samples can be singular.
        return pd.DataFrame([{"term": "fit_error", "message": str(error)}])


def _logistic_table(trials: pd.DataFrame) -> pd.DataFrame:
    sample = trials[trials["logistic_denominator"]].copy()
    sample["first_end_response"] = sample["first_end"].astype(int)
    features = pd.get_dummies(
        sample[["framing", "opponent", "model", "paraphrase_id"]].astype(str),
        drop_first=True,
        dtype=float,
    )
    features["b"] = sample["b"].astype(float).to_numpy()
    features = sm.add_constant(features, has_constant="add")
    try:
        fit = sm.Logit(sample["first_end_response"], features).fit(disp=False)
        interval = fit.conf_int()
        return pd.DataFrame(
            {
                "term": fit.params.index,
                "coefficient": fit.params.values,
                "std_error": fit.bse.values,
                "ci_lower": interval[0].values,
                "ci_upper": interval[1].values,
            }
        )
    except Exception as error:  # Perfect separation is possible in a pilot.
        return pd.DataFrame([{"term": "fit_error", "message": str(error)}])


def _first_bid_summary(trials: pd.DataFrame) -> pd.DataFrame:
    sample = trials[~trials["no_bid"]]
    groups = ["model", "cell_kind", "framing", "opponent", "b"]
    return (
        sample.groupby(groups, dropna=False)
        .agg(
            n=("first_bid", "size"),
            mean_first_bid=("first_bid", "mean"),
            mean_deviation_drop_tie=("x1_deviation_drop_tie", "mean"),
            exact_drop_tie=("x1_deviation_drop_tie", lambda x: float((x == 0).mean())),
            mean_deviation_mixed_tie=("x1_deviation_mixed_tie", "mean"),
            exact_mixed_tie=("x1_deviation_mixed_tie", lambda x: float((x == 0).mean())),
        )
        .reset_index()
    )


def _sensitivity(trials: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for scenario in ("main", "invalid_to_drop", "invalid_to_cap"):
        frame = trials.copy()
        frame["scenario"] = scenario
        frame["censored_sensitivity"] = frame["censored"].astype(bool) | frame["invalid"]
        frame["first_end_sensitivity"] = frame["first_end"].astype(bool)
        if scenario == "invalid_to_drop":
            frame.loc[frame["invalid"], "censored_sensitivity"] = False
            frame.loc[frame["invalid"], "first_end_sensitivity"] = (
                frame.loc[frame["invalid"], "T"] <= 1
            )
        elif scenario == "invalid_to_cap":
            mask = frame["invalid"]
            continued = frame.loc[mask, "T"] + (
                frame.loc[mask, "b"] - frame.loc[mask, "h"]
            )
            if "max_bids" in frame:
                limits = pd.to_numeric(frame.loc[mask, "max_bids"], errors="coerce")
                continued = continued.where(limits.isna(), np.minimum(continued, limits))
            frame.loc[mask, "duration"] = continued
            frame.loc[mask, "censored_sensitivity"] = True
            frame.loc[mask, "first_end_sensitivity"] = False
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _violation_rates(violations: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    groups = ["model", "cell_kind", "framing", "opponent", "b"]
    totals = (
        trials.groupby(groups)
        .agg(n_trials=("unit_id", "size"), n_turns=("T", "sum"), invalid_rate=("invalid", "mean"))
        .reset_index()
    )
    registered_kinds = ["format", "range", "contradiction"]
    observed_kinds = [] if violations.empty else violations["kind"].dropna().unique().tolist()
    kinds = list(dict.fromkeys([*registered_kinds, *observed_kinds]))
    result = totals.merge(pd.DataFrame({"kind": kinds}), how="cross")
    if violations.empty:
        result["n_violations"] = 0
    else:
        counts = (
            violations.groupby([*groups, "kind"])
            .size()
            .rename("n_violations")
            .reset_index()
        )
        result = result.merge(counts, on=[*groups, "kind"], how="left")
        result["n_violations"] = result["n_violations"].fillna(0).astype(int)
    result["rate_per_turn"] = result["n_violations"] / result["n_turns"].clip(lower=1)
    return result


def _usage(children: list[Path]) -> pd.DataFrame:
    rows = []
    for child in children:
        params = config_parameters(child)
        row = {"run_uid": load_run_meta(child)["run_uid"], "model": params.get("model")}
        row.update(run_scope_metrics(child))
        status_path = child / "status.json"
        if status_path.exists():
            row["duration_sec"] = json.loads(status_path.read_text()).get("duration_sec", 0)
        rows.append(row)
    frame = pd.DataFrame(rows)
    numeric = [
        column
        for column in (
            "tokens_in",
            "tokens_out",
            "n_calls",
            "n_calls_missing_prompt_tokens",
            "duration_sec",
        )
        if column in frame
    ]
    return frame.groupby("model", dropna=False)[numeric].sum().reset_index()


def _quiz_table(quiz_run: Path | None) -> pd.DataFrame:
    if quiz_run is None:
        return pd.DataFrame(columns=["model", "framing", "b", "n", "drop_solve_rate", "mixed_solve_rate", "invalid_rate"])
    children = [Path(path) for path in sweep_children(quiz_run)] or [quiz_run]
    rows = []
    for child in children:
        try:
            events = events_table(child, kind="x.dollar-auction-escalation.quiz")
        except (FileNotFoundError, SystemExit):
            continue
        rows.append(events)
    if not rows:
        return pd.DataFrame()
    data = pd.concat(rows, ignore_index=True)
    data["valid"] = ~data["invalid"].astype(bool)
    return (
        data.groupby(["model", "framing", "b"], dropna=False)
        .agg(
            n=("valid", "size"),
            drop_solve_rate=("correct_drop_tie", "mean"),
            mixed_solve_rate=("correct_mixed_tie", "mean"),
            invalid_rate=("valid", lambda x: 1 - float(x.mean())),
        )
        .reset_index()
    )


def _qre_table(qre_run: Path | None) -> pd.DataFrame:
    if qre_run is None:
        return pd.DataFrame(columns=["condition", "lambda_hat", "ci_lower", "ci_upper"])
    candidates = [qre_run / "artifacts" / "qre_fit.csv", qre_run / "qre_fit.csv"]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"qre_fit.csv not found under {qre_run}")


def _sensitivity_models(sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, frame in sensitivity[sensitivity["cell_kind"] == "main"].groupby("scenario", sort=False):
        adjusted = frame.copy()
        adjusted["event_observed"] = ~adjusted["censored_sensitivity"].astype(bool)
        adjusted["first_end"] = adjusted["first_end_sensitivity"].astype(bool)
        cox = _cox_table(adjusted)
        cox["model_type"] = "cox"
        logistic = _logistic_table(adjusted)
        logistic["model_type"] = "logistic"
        result = pd.concat([cox, logistic], ignore_index=True, sort=False)
        result["scenario"] = scenario
        rows.append(result)
    return pd.concat(rows, ignore_index=True, sort=False)


def _plot_km(table: pd.DataFrame, path: Path, provenance: str) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    for _, group in table.groupby(["model", "cell_kind", "framing", "opponent", "b"], dropna=False):
        axis.step(group["time"], group["survival"], where="post", label=_condition_label(group.iloc[0]))
    axis.set(xlabel="Number of bids", ylabel="Continuation probability", title="Kaplan–Meier continuation curves")
    axis.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={"Description": f"Input run_uids: {provenance}"})
    plt.close(fig)


def _plot_first_bids(trials: pd.DataFrame, path: Path, provenance: str) -> None:
    sample = trials[~trials["no_bid"]]
    fig, axis = plt.subplots(figsize=(8, 5))
    labels = sample.apply(_condition_label, axis=1)
    ordered = list(dict.fromkeys(labels))
    values = [sample.loc[labels == label, "first_bid"].to_numpy() for label in ordered]
    axis.boxplot(values, tick_labels=ordered, showfliers=False)
    axis.set(xlabel="Condition", ylabel="First bid or investment", title="First-mover choices")
    axis.tick_params(axis="x", labelrotation=45, labelsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={"Description": f"Input run_uids: {provenance}"})
    plt.close(fig)


def redraw_plots(analysis_dir: str | Path) -> None:
    """Redraw the two registered figures from an existing analysis directory."""
    root = Path(analysis_dir)
    km = pd.read_csv(root / "km_survival.csv")
    trials = pd.read_csv(root / "trials.csv")
    provenance = str(trials["input_run_uids"].iloc[0])
    _plot_km(km, root / "km_curves.png", provenance)
    _plot_first_bids(trials, root / "first_bid.png", provenance)


def analyze_sweep(
    sweep: str | Path,
    *,
    out: str | Path,
    quiz: str | Path | None = None,
    qre: str | Path | None = None,
) -> Path:
    """Analyze a sweep without modifying any input run directory."""
    sweep_path = Path(sweep).resolve()
    children, parent_uids = sweep_sources(sweep_path)
    if not children:
        raise ValueError(f"sweep has no child runs: {sweep_path}")
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    run_uids = parent_uids
    run_uids.extend(str(load_run_meta(child)["run_uid"]) for child in children)
    for optional in (quiz, qre):
        if optional is not None:
            run_uids.append(str(load_run_meta(Path(optional))["run_uid"]))
    provenance = _input_label(run_uids)
    trials, violations = _prepare_trials(children)
    primary = trials[trials["cell_kind"] == "main"]
    km = _km_table(trials)
    sensitivity = _sensitivity(trials)
    outputs = {
        "trials.csv": trials,
        "km_survival.csv": km,
        "cox_ph.csv": _cox_table(primary),
        "logistic_first_end.csv": _logistic_table(primary),
        "first_bid_summary.csv": _first_bid_summary(trials),
        "sensitivity.csv": sensitivity,
        "sensitivity_models.csv": _sensitivity_models(sensitivity),
        "violation_rates.csv": _violation_rates(violations, trials),
        "usage.csv": _usage(children),
        "quiz_solve_rates.csv": _quiz_table(Path(quiz) if quiz else None),
        "qre_lambda.csv": _qre_table(Path(qre) if qre else None),
    }
    for name, frame in outputs.items():
        _write_csv(frame, output / name, provenance)
    _plot_km(km, output / "km_curves.png", provenance)
    _plot_first_bids(trials, output / "first_bid.png", provenance)
    def section(title: str, frame: pd.DataFrame) -> list[str]:
        return [f"## {title}", "", "```text", frame.to_string(index=False), "```", ""]

    first_end = (
        primary[primary["logistic_denominator"]]
        .groupby(["model", "framing", "opponent", "b"], dropna=False)
        .agg(n=("first_end", "size"), first_end_rate=("first_end", "mean"))
        .reset_index()
    )
    summary = [
        "# Dollar-auction pilot analysis",
        "",
        f"Input run_uids: `{provenance}`",
        "",
        f"Trials: {len(trials)}",
        f"No-bid trials (excluded from the first-end denominator): {int(trials['no_bid'].sum())}",
        f"Invalid trials: {int(trials['invalid'].sum())}",
        f"Turn-capped trials: {int(trials['turn_capped'].astype(bool).sum())}",
        "",
        "Invalid trials are right-censored in the main survival analysis. The two registered reinterpretations are reported separately below.",
        "",
    ]
    summary += section("First-round endings", first_end)
    summary += section("Cox proportional-hazards model", outputs["cox_ph.csv"])
    summary += section("First-round logistic model", outputs["logistic_first_end.csv"])
    summary += section("First-bid anchors", outputs["first_bid_summary.csv"])
    summary += section("Sensitivity-model estimates", outputs["sensitivity_models.csv"])
    summary += section("Quiz solve rates", outputs["quiz_solve_rates.csv"])
    summary += section("QRE estimates", outputs["qre_lambda.csv"])
    summary += section("Violation rates", outputs["violation_rates.csv"])
    summary += section("Model usage", outputs["usage.csv"])
    (output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (output / "analysis.json").write_text(
        json.dumps({"input_run_uids": run_uids}, indent=2) + "\n", encoding="utf-8"
    )
    return output
