use dollar_auction_simulation::bidder::Response;
use dollar_auction_simulation::qre::{AgentQre, ObservedAction, QreTree};
use dollar_auction_simulation::solver::{solve, solve_mixed_ties};
use std::time::{Duration, Instant};

#[test]
fn zero_lambda_is_uniform_over_every_legal_action() {
    let qre = AgentQre::new(5, 4);
    let probs = qre.action_probabilities(0, 0, 0.0);
    assert_eq!(probs.len(), 5); // drop plus bids 1..=4
    for (_, p) in probs {
        assert!((p - 0.2).abs() < 1e-12);
    }
}

#[test]
fn probabilities_sum_to_one_and_large_lambda_selects_solver_action() {
    let qre = AgentQre::new(10, 23);
    let probs = qre.action_probabilities(0, 0, 0.7);
    let sum: f64 = probs.iter().map(|(_, p)| p).sum();
    assert!((sum - 1.0).abs() < 1e-12);

    let large_qre = AgentQre::new(3, 1);
    let large_probs = large_qre.action_probabilities(0, 0, 100.0);
    let solver_action = Response::Bid(solve(3, 1).first_bid);
    let solver_probability = large_probs
        .iter()
        .find_map(|(action, probability)| (*action == solver_action).then_some(*probability))
        .unwrap();
    assert!(
        solver_probability > 0.999,
        "probability={solver_probability}"
    );
}

#[test]
fn observed_sequence_has_finite_log_likelihood() {
    let qre = AgentQre::new(10, 23);
    let actions = [
        ObservedAction::new(0, 0, Response::Bid(5)),
        ObservedAction::new(0, 5, Response::Drop),
    ];
    assert!(qre.log_likelihood(0.7, &actions).is_finite());
}

#[test]
fn large_tree_is_reusable_and_fast() {
    let started = Instant::now();
    let tree = QreTree::new(100, 250, 0.7);
    let root = tree.action_probabilities(0, 0);
    let middle = tree.action_probabilities(49, 150);

    assert_eq!(root.len(), 251);
    assert_eq!(middle.len(), 101);
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[test]
fn large_tree_probabilities_sum_to_one_at_several_nodes() {
    let tree = QreTree::new(100, 250, 0.7);
    for (me, opp) in [(0, 0), (0, 1), (49, 150), (249, 250)] {
        let sum: f64 = tree
            .action_probabilities(me, opp)
            .iter()
            .map(|(_, probability)| probability)
            .sum();
        assert!((sum - 1.0).abs() < 1e-12, "me={me}, opp={opp}, sum={sum}");
    }
}

#[test]
fn large_lambda_converges_to_uniform_mixing_on_ties() {
    for (b, expected_bid) in [(100, 1), (250, 50)] {
        let tree = QreTree::new(100, b, 50.0);
        let probability = probability_of(
            &tree.action_probabilities(0, 0),
            &Response::Bid(expected_bid),
        );
        assert!(probability > 0.99, "b={b}, probability={probability}");
    }

    for s in [3, 5, 10] {
        for b in 1..(4 * s) {
            let expected = Response::Bid(solve_mixed_ties(s, b).first_bid);
            let tree = QreTree::new(s, b, 50.0);
            assert_eq!(
                most_likely(&tree.action_probabilities(0, 0)),
                expected,
                "s={s}, b={b}"
            );
        }
    }
    for b in [100, 250] {
        let expected = Response::Bid(solve_mixed_ties(100, b).first_bid);
        let tree = QreTree::new(100, b, 50.0);
        assert_eq!(
            most_likely(&tree.action_probabilities(0, 0)),
            expected,
            "s=100, b={b}"
        );
    }
}

#[test]
fn large_tree_log_likelihood_reuses_one_evaluation() {
    let started = Instant::now();
    let tree = QreTree::new(100, 250, 0.7);
    let observed: Vec<_> = (0..100)
        .map(|_| ObservedAction::new(0, 0, Response::Bid(50)))
        .collect();
    let probability = probability_of(&tree.action_probabilities(0, 0), &Response::Bid(50));
    let expected = 100.0 * probability.ln();
    let actual = tree.log_likelihood(&observed);

    assert!((actual - expected).abs() < 1e-10);
    assert!(started.elapsed() < Duration::from_secs(5));
}

fn probability_of(probabilities: &[(Response, f64)], expected: &Response) -> f64 {
    probabilities
        .iter()
        .find_map(|(action, probability)| (action == expected).then_some(*probability))
        .unwrap()
}

fn most_likely(probabilities: &[(Response, f64)]) -> Response {
    probabilities
        .iter()
        .reduce(|best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        })
        .unwrap()
        .0
        .clone()
}
