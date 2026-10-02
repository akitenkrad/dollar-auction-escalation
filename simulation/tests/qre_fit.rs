use dollar_auction_simulation::bidder::Response;
use dollar_auction_simulation::qre::{ObservedAction, QreTree};
use dollar_auction_simulation::qre_fit::{fit_qre, observations_from_events, FitObservation};

#[test]
fn fit_recovers_known_lambda_inside_profile_interval_and_counts_exclusions() {
    let known = 1.25;
    let tree = QreTree::new(5, 8, known);
    let probabilities = tree.action_probabilities(0, 0);
    let mut observations = Vec::new();
    for index in 0..1200 {
        let draw = ((index * 7919) % 1201) as f64 / 1201.0;
        let mut cumulative = 0.0;
        let mut selected = Response::Drop;
        for (action, probability) in &probabilities {
            cumulative += probability;
            if draw <= cumulative {
                selected = action.clone();
                break;
            }
        }
        observations.push(FitObservation::Valid(ObservedAction::new(0, 0, selected)));
    }
    observations.push(FitObservation::Invalid);
    observations.push(FitObservation::Violation);

    let fit = fit_qre(5, 8, &observations);
    assert!(
        fit.lambda_low <= known && known <= fit.lambda_high,
        "{fit:?}"
    );
    assert!((fit.lambda_hat - known).abs() < 0.35, "{fit:?}");
    assert_eq!(fit.n_included, 1200);
    assert_eq!(fit.n_excluded, 2);
    assert!(fit.log_likelihood.is_finite());
}

#[test]
fn event_extraction_keeps_valid_decisions_and_counts_invalid_surfaces() {
    let events = [
        serde_json::json!({"schema":"x.dollar-auction-escalation.decision","me":0,"opp":0,"action":"bid","amount":3}),
        serde_json::json!({"schema":"x.dollar-auction-escalation.decision","me":0,"opp":3,"action":"drop"}),
        serde_json::json!({"schema":"x.dollar-auction-escalation.decision","me":3,"opp":4,"action":"drop","is_llm":false}),
        serde_json::json!({"schema":"x.dollar-auction-escalation.violation"}),
        serde_json::json!({"schema":"terminal","outcome":"invalid"}),
    ];
    let observations = observations_from_events(&events);
    assert_eq!(observations.len(), 4);
    assert!(matches!(observations[0], FitObservation::Valid(_)));
    assert!(matches!(observations[1], FitObservation::Valid(_)));
    assert_eq!(observations[2], FitObservation::Violation);
    assert_eq!(observations[3], FitObservation::Invalid);
}
