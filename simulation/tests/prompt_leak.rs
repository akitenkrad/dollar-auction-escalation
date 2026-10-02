use dollar_auction_simulation::prompt::{
    render_system, render_user, Framing, OpponentAnnouncement, PromptCondition,
};
use dollar_auction_simulation::protocol::{Violation, ViolationKind};
use dollar_auction_simulation::solver::proposition_first_bid;
use dollar_auction_simulation::world::{Observation, Role};

const DISGUISED_BANNED: &[&str] = &[
    "dollar",
    "auction",
    "shubik",
    "escalat",
    "all-pay",
    "war of attrition",
    "sunk cost",
];
const ALL_BANNED: &[&str] = &[
    "backward induction",
    "subgame",
    "equilibrium",
    "remaining rounds",
    "must end",
    "shubik",
    "refund",
    "sunk",
    "non-refundable",
    "cannot be recovered",
];
const DISGUISED_WHOLE_WORD_BANNED: &[&str] = &["bid", "drop", "prize", "bill", "budget"];
const NAMED_WHOLE_WORD_BANNED: &[&str] = &["contract", "invest", "withdraw", "firm", "credits"];

fn empty_observation(b: u32) -> Observation {
    Observation {
        s: 100,
        own_b: b,
        opponent_b: b,
        history: Vec::new(),
        role: Role::FirstMover,
        turn: 1,
        paid_so_far: None,
        calculator_feedback: Vec::new(),
    }
}

fn standalone_number(text: &str, value: u32) -> bool {
    text.split(|c: char| c.is_ascii_alphanumeric())
        .any(|token| token == value.to_string())
}

fn contains_whole_word(text: &str, word: &str) -> bool {
    text.split(|character: char| !character.is_ascii_alphanumeric())
        .any(|token| token.eq_ignore_ascii_case(word))
}

fn feedback_with_cross_framing_words(framing: Framing) -> Violation {
    let message = match framing {
        Framing::Named => "contract invest withdraw firm credits",
        Framing::Disguised => "bid drop auction prize bill budget",
    };
    Violation {
        kind: ViolationKind::Format,
        attempt: 1,
        message: message.to_string(),
        raw_output: String::new(),
    }
}

#[test]
fn disguised_prompts_do_not_leak_the_game_identity_or_solution() {
    for paraphrase in 0..3 {
        let condition = PromptCondition {
            framing: Framing::Disguised,
            paraphrase,
            ..Default::default()
        };
        for b in [100, 250] {
            let combined = format!(
                "{}\n{}",
                render_system(&condition),
                render_user(&empty_observation(b), &condition, None)
            );
            let lower = combined.to_ascii_lowercase();
            for banned in DISGUISED_BANNED.iter().chain(ALL_BANNED) {
                assert!(!lower.contains(banned), "found {banned:?} in {combined}");
            }
            let x1 = proposition_first_bid(100, b);
            if x1 >= 10 {
                assert!(!standalone_number(&combined, x1), "leaked x1={x1}");
            }
        }
    }
}

#[test]
fn named_prompts_do_not_leak_solution_language() {
    for paraphrase in 0..3 {
        let condition = PromptCondition {
            framing: Framing::Named,
            paraphrase,
            ..Default::default()
        };
        let combined = format!(
            "{}\n{}",
            render_system(&condition),
            render_user(&empty_observation(250), &condition, None)
        );
        let lower = combined.to_ascii_lowercase();
        for banned in ALL_BANNED {
            assert!(!lower.contains(banned));
        }
        assert!(!standalone_number(&combined, 52));
    }
}

#[test]
fn every_rendered_surface_uses_only_its_framing_vocabulary() {
    for framing in [Framing::Named, Framing::Disguised] {
        for opponent in [
            OpponentAnnouncement::RationalAi,
            OpponentAnnouncement::Human,
            OpponentAnnouncement::SameModel,
        ] {
            for paraphrase in 0..3 {
                let condition = PromptCondition {
                    framing,
                    opponent,
                    paraphrase,
                    calculator: true,
                };
                let combined = format!(
                    "{}\n{}",
                    render_system(&condition),
                    render_user(
                        &empty_observation(250),
                        &condition,
                        Some(&feedback_with_cross_framing_words(framing))
                    )
                );
                let lower = combined.to_ascii_lowercase();
                for banned in ALL_BANNED {
                    assert!(!lower.contains(banned), "found {banned:?} in {combined}");
                }
                let vocabulary = match framing {
                    Framing::Named => NAMED_WHOLE_WORD_BANNED,
                    Framing::Disguised => DISGUISED_WHOLE_WORD_BANNED,
                };
                for banned in vocabulary {
                    assert!(
                        !contains_whole_word(&combined, banned),
                        "found {banned:?} in {combined}"
                    );
                }
            }
        }
    }
}
