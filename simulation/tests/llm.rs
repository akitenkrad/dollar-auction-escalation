use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use dollar_auction_simulation::bidder::{Bidder, LlmBidder, QreBidder, SolverBidder};
use dollar_auction_simulation::llm_bidder::parse;
use dollar_auction_simulation::prompt::{Framing, PromptCondition};
use dollar_auction_simulation::seeds::{llm_call_seed, trial_seed};
use dollar_auction_simulation::simulation::play_trial;
use dollar_auction_simulation::world::{Observation, Role};
use socsim_llm::mock::ScriptedClient;

fn obs() -> Observation {
    Observation {
        s: 10,
        own_b: 12,
        opponent_b: 12,
        history: vec![],
        role: Role::FirstMover,
        turn: 1,
        paid_so_far: None,
        calculator_feedback: vec![],
    }
}

#[test]
fn identical_calls_both_reach_the_scripted_backend() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let client = ScriptedClient::new("scripted", move |_| {
        seen.fetch_add(1, Ordering::SeqCst);
        r#"{"action":"drop","reason":"done"}"#.to_string()
    });
    let mut bidder = LlmBidder::new(Box::new(client), PromptCondition::default(), 7);
    bidder.act(&obs(), None);
    bidder.act(&obs(), None);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(bidder
        .transcripts()
        .borrow()
        .iter()
        .all(|t| !t.metadata.cache_hit));
}

#[test]
fn seeds_follow_the_two_level_derivation() {
    let trial = trial_seed(42, 3, 9);
    assert_eq!(trial, socsim_core::derive_seed(42, &[3, 9]));
    assert_eq!(
        llm_call_seed(trial, 2, 1),
        socsim_core::derive_seed(trial, &[2, 1])
    );
    assert_ne!(llm_call_seed(trial, 2, 1), llm_call_seed(trial, 3, 1));
    assert_ne!(llm_call_seed(trial, 2, 1), llm_call_seed(trial, 2, 2));
}

#[test]
fn any_bidder_implementation_can_occupy_either_seat() {
    let transcript_sink = Rc::new(RefCell::new(Vec::new()));
    let p1: Box<dyn Bidder> = Box::new(QreBidder::new(0.0, 1));
    let p2: Box<dyn Bidder> = Box::new(SolverBidder::new());
    let result = play_trial(10, 12, p1, p2, false, transcript_sink).unwrap();
    assert!(!result.outcome.is_empty());
}

#[test]
fn implementation_does_not_use_fallback_or_cache_wrappers() {
    let source = include_str!("../src/llm.rs");
    for forbidden in [
        "FallbackClient",
        "PromptCache",
        "wrap_client",
        "build_live_client_from_settings",
    ] {
        assert!(
            !source.contains(forbidden),
            "found forbidden API {forbidden}"
        );
    }
}

#[test]
fn llm_bidder_surface_exposes_the_protocol_parser() {
    let parsed = parse(r#"{"action":"drop","reason":"done"}"#).unwrap();
    assert_eq!(
        parsed.response,
        dollar_auction_simulation::bidder::Response::Drop
    );
}

#[test]
fn transcript_records_the_raw_disguised_action_word() {
    let client = ScriptedClient::constant(
        "scripted",
        r#"{"action":"invest","amount":4,"reason":"prepare"}"#,
    );
    let condition = PromptCondition {
        framing: Framing::Disguised,
        ..Default::default()
    };
    let mut bidder = LlmBidder::new(Box::new(client), condition, 7);
    assert_eq!(
        bidder.act(&obs(), None),
        dollar_auction_simulation::bidder::Response::Bid(4)
    );
    assert_eq!(
        bidder.transcripts().borrow()[0].raw_action.as_deref(),
        Some("invest")
    );
}

#[test]
fn transcript_preserves_a_rejected_cross_framing_action_word() {
    let client = ScriptedClient::constant(
        "scripted",
        r#"{"action":"bid","amount":4,"reason":"wrong vocabulary"}"#,
    );
    let condition = PromptCondition {
        framing: Framing::Disguised,
        ..Default::default()
    };
    let mut bidder = LlmBidder::new(Box::new(client), condition, 7);
    assert!(matches!(
        bidder.act(&obs(), None),
        dollar_auction_simulation::bidder::Response::Malformed(_)
    ));
    assert_eq!(
        bidder.transcripts().borrow()[0].raw_action.as_deref(),
        Some("bid")
    );
}
