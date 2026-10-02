use std::sync::{Arc, Mutex};

use dollar_auction_simulation::bidder::{Bidder, LlmBidder};
use std::collections::BTreeSet;

use dollar_auction_simulation::prompt::{
    render_system, render_system_for, render_user, Framing, OpponentAnnouncement, PromptCondition,
};
use dollar_auction_simulation::world::{Bid, Observation, Player, Role};
use socsim_llm::{CallMetadata, LlmClient, LlmConfig, LlmError, LlmResponse};

struct CapturingClient {
    configs: Arc<Mutex<Vec<LlmConfig>>>,
}

impl LlmClient for CapturingClient {
    fn model(&self) -> &str {
        "capture"
    }

    fn endpoint(&self) -> &str {
        "mock://capture"
    }

    fn complete(&self, _prompt: &str, config: &LlmConfig) -> Result<LlmResponse, LlmError> {
        self.configs.lock().unwrap().push(config.clone());
        Ok(LlmResponse {
            text: r#"{"action":"drop","reason":"done"}"#.to_string(),
            thinking: None,
            metadata: CallMetadata {
                model: self.model().to_string(),
                endpoint: self.endpoint().to_string(),
                temperature: config.temperature,
                seed: config.seed,
                cache_hit: false,
                usage: None,
            },
            logprobs: None,
        })
    }
}

fn observation(show_paid: bool) -> Observation {
    Observation {
        s: 100,
        own_b: 250,
        opponent_b: 250,
        history: vec![
            Bid {
                player: Player::P1,
                amount: 52,
                attempt: 1,
            },
            Bid {
                player: Player::P2,
                amount: 73,
                attempt: 1,
            },
        ],
        role: Role::FirstMover,
        turn: 3,
        paid_so_far: show_paid.then_some(52),
        calculator_feedback: Vec::new(),
    }
}

fn turn_one() -> Observation {
    Observation {
        s: 100,
        own_b: 250,
        opponent_b: 250,
        history: Vec::new(),
        role: Role::FirstMover,
        turn: 1,
        paid_so_far: None,
        calculator_feedback: Vec::new(),
    }
}

fn numbers(text: &str) -> BTreeSet<u32> {
    text.split(|character: char| !character.is_ascii_digit())
        .filter(|token| !token.is_empty())
        .map(|token| token.parse().unwrap())
        .collect()
}

fn rule_signature(text: &str, framing: Framing) -> [bool; 7] {
    let lower = text.to_ascii_lowercase();
    let (limit, exit) = match framing {
        Framing::Named => ("budget", "drop"),
        Framing::Disguised => ("cap", "withdraw"),
    };
    [
        lower.contains("first") || lower.contains("opening turn"),
        lower.contains("alternat"),
        lower.contains(limit),
        lower.contains("known") || lower.contains("know") || lower.contains("common knowledge"),
        lower.contains(exit),
        lower.contains("own") && lower.contains("pay"),
        lower.contains('0'),
    ]
}

#[test]
fn every_call_reconstructs_full_history_without_reasons() {
    let condition = PromptCondition::default();
    let first = render_user(&observation(false), &condition, None);
    let second = render_user(&observation(false), &condition, None);
    assert_eq!(first, second);
    assert!(first.contains("Player 1 bid 52"));
    assert!(first.contains("Player 2 bid 73"));
    assert!(!first.contains("private reason"));
    assert!(first.contains("Legal action: bid an integer from 74 through 250, or drop."));
}

#[test]
fn paid_so_far_appears_only_when_enabled_in_the_observation() {
    let condition = PromptCondition::default();
    assert!(!render_user(&observation(false), &condition, None).contains("paid so far"));
    assert!(render_user(&observation(true), &condition, None).contains("paid so far: 52"));
}

#[test]
fn no_prompt_enables_thinking_tokens() {
    for condition in PromptCondition::all_primary() {
        assert!(!render_system(&condition).contains("<|think|>"));
        assert!(!render_user(&observation(false), &condition, None).contains("<|think|>"));
    }
}

#[test]
fn every_llm_call_explicitly_disables_thinking() {
    let configs = Arc::new(Mutex::new(Vec::new()));
    let client = CapturingClient {
        configs: Arc::clone(&configs),
    };
    let mut bidder = LlmBidder::new(Box::new(client), PromptCondition::default(), 7);
    bidder.act(&observation(false), None);
    bidder.act(&observation(false), None);

    let captured = configs.lock().unwrap();
    assert_eq!(captured.len(), 2);
    assert!(captured.iter().all(|config| config.think == Some(false)));
    assert!(bidder
        .transcripts()
        .borrow()
        .iter()
        .all(|transcript| transcript.think == Some(false)));
}

#[test]
fn framing_specific_turn_labels_vocabulary_and_opponent_announcements_are_natural() {
    let named = PromptCondition {
        opponent: OpponentAnnouncement::Human,
        calculator: true,
        ..Default::default()
    };
    let named_system = render_system(&named);
    let named_user = render_user(&turn_one(), &named, None);
    assert!(named_system.contains("the other player is a human participant"));
    assert!(named_user.contains("Bill value: 100."));
    assert!(named_user.contains("Your budget: 250. Other player's budget: 250."));
    assert!(named_user.contains(r#"{"action":"bid""#));
    assert!(named_user.contains(r#"{"action":"drop""#));

    let disguised = PromptCondition {
        framing: Framing::Disguised,
        opponent: OpponentAnnouncement::Human,
        calculator: true,
        ..Default::default()
    };
    let disguised_system = render_system(&disguised);
    let disguised_user = render_user(&turn_one(), &disguised, None);
    assert!(disguised_system.contains("the other firm is run by a human participant"));
    assert!(disguised_user.contains("Contract value: 100."));
    assert!(disguised_user.contains("Your cap: 250. Other firm's cap: 250."));
    assert!(disguised_user.contains(r#"{"action":"invest""#));
    assert!(disguised_user.contains(r#"{"action":"withdraw""#));

    for (opponent, expected) in [
        (
            OpponentAnnouncement::RationalAi,
            "You are told that the other firm is run by a rational AI agent.",
        ),
        (
            OpponentAnnouncement::Human,
            "You are told that the other firm is run by a human participant.",
        ),
        (
            OpponentAnnouncement::SameModel,
            "You are told that the other firm is run by the same language model as you.",
        ),
    ] {
        let condition = PromptCondition {
            framing: Framing::Disguised,
            opponent,
            ..Default::default()
        };
        assert!(render_system(&condition).contains(expected));
    }
}

#[test]
fn named_paraphrases_use_the_same_game_name_once() {
    for paraphrase in 0..3 {
        let condition = PromptCondition {
            paraphrase,
            ..Default::default()
        };
        let system = render_system(&condition).to_ascii_lowercase();
        assert_eq!(system.matches("the dollar auction").count(), 1, "{system}");
    }
}

#[test]
fn optional_prompt_blocks_do_not_leave_empty_lines() {
    for framing in [Framing::Named, Framing::Disguised] {
        let condition = PromptCondition {
            framing,
            calculator: false,
            ..Default::default()
        };
        let rendered = render_user(&turn_one(), &condition, None);
        assert!(!rendered.contains("\n\n"), "{rendered}");
        assert!(!rendered.ends_with('\n'));
    }
}

#[test]
fn paraphrases_preserve_numbers_and_rule_statements_in_full_prompt_pairs() {
    for framing in [Framing::Named, Framing::Disguised] {
        let mut expected_numbers = None;
        let mut expected_rules = None;
        for paraphrase in 0..3 {
            let condition = PromptCondition {
                framing,
                opponent: OpponentAnnouncement::Human,
                paraphrase,
                calculator: true,
            };
            let system = render_system_for(100, 250, Player::P1, &condition);
            let full_pair = format!(
                "{system}\n{}\n{}",
                render_user(&turn_one(), &condition, None),
                render_user(&observation(false), &condition, None)
            );
            let current_numbers = numbers(&full_pair);
            let current_rules = rule_signature(&system, framing);
            assert!(
                current_numbers.is_superset(&BTreeSet::from([0, 1, 18, 27, 52, 73, 74, 100, 250]))
            );
            assert!(current_rules.into_iter().all(|present| present));
            if let Some(expected) = &expected_numbers {
                assert_eq!(expected, &current_numbers);
            } else {
                expected_numbers = Some(current_numbers);
            }
            if let Some(expected) = expected_rules {
                assert_eq!(expected, current_rules);
            } else {
                expected_rules = Some(current_rules);
            }
        }
    }
}
