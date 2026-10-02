use dollar_auction_simulation::solver::{
    proposition_first_bid, solve, solve_mixed_ties, verify_reference_anchors,
};
use std::process::Command;

#[test]
fn exact_solver_matches_proposition_3_1_on_the_full_grid() {
    for s in [3, 5, 10] {
        for b in 1..(4 * s) {
            let result = solve(s, b);
            assert_eq!(
                result.first_bid,
                proposition_first_bid(s, b),
                "s={s}, b={b}"
            );
            assert_eq!(result.second_reply, None, "s={s}, b={b}");
        }
    }
    assert_eq!(solve(100, 100).first_bid, 1);
    assert_eq!(solve(100, 250).first_bid, 52);
    assert_eq!(solve(100, 250).second_reply, None);
}

#[test]
fn rust_solver_matches_generated_python_vectors() {
    let text = include_str!("vectors.json");
    let vectors: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
    for vector in vectors {
        let s = vector["s"].as_u64().unwrap() as u32;
        let b = vector["b"].as_u64().unwrap() as u32;
        let expected = vector["x1"].as_u64().unwrap() as u32;
        let result = solve(s, b);
        assert_eq!(result.first_bid, expected, "s={s}, b={b}");
        assert!(vector["second_reply"].is_null());
        assert_eq!(result.second_reply, None);
    }
}

#[test]
fn mixed_tie_solver_matches_confirmed_large_game_values() {
    let b100 = solve_mixed_ties(100, 100);
    assert_eq!(b100.first_bid, 1);
    assert!((b100.first_mover_value - 49.0).abs() < 1e-12);

    let b250 = solve_mixed_ties(100, 250);
    assert_eq!(b250.first_bid, 50);
    assert!((b250.first_mover_value - 50.0).abs() < 1e-12);
}

#[test]
fn repository_reference_anchors_match_both_solvers() {
    verify_reference_anchors(include_str!("../../reference.csv")).unwrap();
}

#[test]
fn solve_cli_supports_mixed_ties() {
    let output = Command::new(env!("CARGO_BIN_EXE_dollar-auction"))
        .args(["solve", "--s", "100", "--b", "250", "--tie", "mixed"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["tie"], "mixed");
    assert_eq!(json["first_bid"], 50);
    assert_eq!(json["first_mover_value"], 50.0);
}
