"""Export decision reasons and calculate inter-coder agreement."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from runvault.read import config_parameters, load_run_meta, sweep_children


CODES = ("sunk_cost", "rivalry", "foresight", "game_recall")


def _runs(run: Path) -> list[Path]:
    children = [Path(path) for path in sweep_children(run)]
    return children or [run]


def export_reasons(run: str | Path, output: str | Path) -> Path:
    """Write one row per non-calculator LLM decision."""
    rows: list[dict[str, object]] = []
    for child in _runs(Path(run)):
        meta = load_run_meta(child)
        params = config_parameters(child, required=False) or {}
        transcripts = child / "artifacts" / "transcripts.jsonl"
        if not transcripts.exists():
            continue
        for line in transcripts.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("raw_action") == "calc" or not item.get("reason"):
                continue
            row: dict[str, object] = {
                "run_uid": meta["run_uid"],
                "trial": item.get("trial"),
                "turn": item.get("turn"),
                "framing": params.get("framing"),
                "opponent": params.get("opponent"),
                "reason": item.get("reason"),
            }
            row.update({code: "" for code in CODES})
            rows.append(row)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_uid", "trial", "turn", "framing", "opponent", "reason", *CODES]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _kappa(left: list[str], right: list[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("coded files must contain the same non-zero number of rows")
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    labels = sorted(set(left) | set(right))
    expected = sum(
        (left.count(label) / len(left)) * (right.count(label) / len(right))
        for label in labels
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else float("nan")
    return (observed - expected) / (1.0 - expected)


def agreement(coder_a: str | Path, coder_b: str | Path) -> dict[str, float]:
    """Print and return Cohen's kappa for each registered binary code."""
    with Path(coder_a).open(encoding="utf-8", newline="") as handle:
        left = list(csv.DictReader(handle))
    with Path(coder_b).open(encoding="utf-8", newline="") as handle:
        right = list(csv.DictReader(handle))
    identity = ("run_uid", "trial", "turn")
    if [[row.get(key) for key in identity] for row in left] != [
        [row.get(key) for key in identity] for row in right
    ]:
        raise ValueError("coded files do not contain the same decisions in the same order")
    kappas = {code: _kappa([row[code] for row in left], [row[code] for row in right]) for code in CODES}
    for code, value in kappas.items():
        print(f"{code}: {value:.6g}")
    return kappas
