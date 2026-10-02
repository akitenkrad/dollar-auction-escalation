use std::collections::HashMap;

use crate::bidder::Response;

type Value = (i64, i64, Option<u32>);
type Memo = HashMap<(u32, u32), Value>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolverResult {
    pub first_bid: u32,
    pub second_reply: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixedTieResult {
    pub first_bid: u32,
    pub first_mover_value: f64,
}

#[derive(Debug, Clone, Copy)]
struct MixedNodeValue {
    mover: f64,
    opponent: f64,
    first_optimal_bid: Option<u32>,
}

type MixedMemo = HashMap<(u32, u32), MixedNodeValue>;

const TIE_EPSILON: f64 = 1e-10;

fn value(me: u32, opp: u32, s: u32, b: u32, memo: &mut Memo) -> Value {
    if let Some(result) = memo.get(&(me, opp)) {
        return *result;
    }
    let mut best = (
        -i64::from(me),
        if opp > 0 {
            i64::from(s) - i64::from(opp)
        } else {
            0
        },
        None,
    );
    if opp < b {
        for y in (opp + 1)..=b {
            let (opponent_payoff, mover_payoff, _) = value(opp, y, s, b, memo);
            if mover_payoff > best.0 {
                best = (mover_payoff, opponent_payoff, Some(y));
            }
        }
    }
    memo.insert((me, opp), best);
    best
}

pub fn best_response(s: u32, b: u32, me: u32, opp: u32) -> Response {
    let mut memo = Memo::new();
    match value(me, opp, s, b, &mut memo).2 {
        Some(amount) => Response::Bid(amount),
        None => Response::Drop,
    }
}

pub fn solve(s: u32, b: u32) -> SolverResult {
    assert!(s >= 2, "value must be at least 2");
    assert!(b >= 1, "budget must be at least 1");
    let mut memo = Memo::new();
    let first_bid = value(0, 0, s, b, &mut memo)
        .2
        .expect("the first mover always has a legal optimal bid");
    let second_reply = value(0, first_bid, s, b, &mut memo).2;
    SolverResult {
        first_bid,
        second_reply,
    }
}

pub fn proposition_first_bid(s: u32, b: u32) -> u32 {
    (b - 1) % (s - 1) + 1
}

/// Solves the subgame-perfect limit in which all payoff-maximizing actions at
/// a node are mixed uniformly, matching the limit of the agent-logit QRE.
pub fn solve_mixed_ties(s: u32, b: u32) -> MixedTieResult {
    assert!(s >= 2, "value must be at least 2");
    assert!(b >= 1, "budget must be at least 1");
    let mut memo = MixedMemo::new();
    let root = mixed_value(0, 0, s, b, &mut memo);
    MixedTieResult {
        first_bid: root
            .first_optimal_bid
            .expect("the first mover always has a legal optimal bid"),
        first_mover_value: root.mover,
    }
}

fn mixed_value(me: u32, opp: u32, s: u32, b: u32, memo: &mut MixedMemo) -> MixedNodeValue {
    if let Some(result) = memo.get(&(me, opp)).copied() {
        return result;
    }

    let mut actions = vec![(
        None,
        -f64::from(me),
        if opp > 0 {
            f64::from(s) - f64::from(opp)
        } else {
            0.0
        },
    )];
    if opp < b {
        for amount in (opp + 1)..=b {
            let child = mixed_value(opp, amount, s, b, memo);
            actions.push((Some(amount), child.opponent, child.mover));
        }
    }

    let best_payoff = actions
        .iter()
        .map(|(_, mover, _)| *mover)
        .fold(f64::NEG_INFINITY, f64::max);
    let best: Vec<_> = actions
        .iter()
        .filter(|(_, mover, _)| (*mover - best_payoff).abs() <= TIE_EPSILON)
        .collect();
    let count = best.len() as f64;
    let result = MixedNodeValue {
        mover: best.iter().map(|(_, mover, _)| mover).sum::<f64>() / count,
        opponent: best.iter().map(|(_, _, opponent)| opponent).sum::<f64>() / count,
        first_optimal_bid: best.iter().find_map(|(amount, _, _)| *amount),
    };
    memo.insert((me, opp), result);
    result
}

pub fn verify_reference_anchors(csv: &str) -> Result<(), String> {
    let mut lines = csv.lines();
    let header = lines.next().ok_or("reference.csv is empty")?;
    if header != "run_uid,step,step_unit,scope,name,value,target_id,source" {
        return Err("reference.csv has the wrong header".to_string());
    }
    let mut values = HashMap::new();
    for (index, line) in lines.enumerate() {
        let columns: Vec<_> = line.split(',').collect();
        if columns.len() != 8 {
            return Err(format!(
                "reference.csv row {} has the wrong width",
                index + 2
            ));
        }
        let value = columns[5]
            .parse::<f64>()
            .map_err(|_| format!("invalid reference value on row {}", index + 2))?;
        values.insert(columns[4], value);
    }

    let expected = [
        ("x1_star.s100.b100", f64::from(solve(100, 100).first_bid)),
        ("x1_star.s100.b250", f64::from(solve(100, 250).first_bid)),
        ("solver_selfplay.first_end_rate", 1.0),
        (
            "solver_selfplay.waste.b250",
            f64::from(solve(100, 250).first_bid) / 100.0,
        ),
        (
            "x1_star_mixtie.s100.b250",
            f64::from(solve_mixed_ties(100, 250).first_bid),
        ),
        (
            "x1_star_mixtie.s100.b100",
            f64::from(solve_mixed_ties(100, 100).first_bid),
        ),
    ];
    for (name, actual) in expected {
        let reference = values
            .get(name)
            .ok_or_else(|| format!("reference.csv is missing {name}"))?;
        if (reference - actual).abs() > 1e-12 {
            return Err(format!(
                "reference mismatch for {name}: reference={reference}, solver={actual}"
            ));
        }
    }
    Ok(())
}
