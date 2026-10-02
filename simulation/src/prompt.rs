use crate::protocol::{Violation, ViolationKind};
use crate::world::{Observation, Player};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    Named,
    Disguised,
}

impl Framing {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Named => "named",
            Self::Disguised => "disguised",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpponentAnnouncement {
    RationalAi,
    Human,
    SameModel,
}

impl OpponentAnnouncement {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RationalAi => "rational_ai",
            Self::Human => "human",
            Self::SameModel => "same_model",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptCondition {
    pub framing: Framing,
    pub opponent: OpponentAnnouncement,
    pub paraphrase: u8,
    pub calculator: bool,
}

impl Default for PromptCondition {
    fn default() -> Self {
        Self {
            framing: Framing::Named,
            opponent: OpponentAnnouncement::RationalAi,
            paraphrase: 0,
            calculator: false,
        }
    }
}

impl PromptCondition {
    pub fn all_primary() -> Vec<Self> {
        let mut conditions = Vec::new();
        for framing in [Framing::Named, Framing::Disguised] {
            for opponent in [
                OpponentAnnouncement::RationalAi,
                OpponentAnnouncement::Human,
                OpponentAnnouncement::SameModel,
            ] {
                for paraphrase in 0..3 {
                    conditions.push(Self {
                        framing,
                        opponent,
                        paraphrase,
                        calculator: false,
                    });
                }
            }
        }
        conditions
    }
}

pub fn render_system(condition: &PromptCondition) -> String {
    render_system_for(100, 250, Player::P1, condition)
}

pub fn render_system_for(s: u32, b: u32, player: Player, condition: &PromptCondition) -> String {
    let template = match (condition.framing, condition.paraphrase) {
        (Framing::Named, 0) => include_str!("../../prompts/named_0.txt"),
        (Framing::Named, 1) => include_str!("../../prompts/named_1.txt"),
        (Framing::Named, 2) => include_str!("../../prompts/named_2.txt"),
        (Framing::Disguised, 0) => include_str!("../../prompts/disguised_0.txt"),
        (Framing::Disguised, 1) => include_str!("../../prompts/disguised_1.txt"),
        (Framing::Disguised, 2) => include_str!("../../prompts/disguised_2.txt"),
        (_, paraphrase) => panic!("unsupported paraphrase id: {paraphrase}"),
    };
    let opponent = opponent_announcement(condition);
    let mut rendered = template
        .replace("{{s}}", &s.to_string())
        .replace("{{b}}", &b.to_string())
        .replace("{{player}}", &(player.index() + 1).to_string());
    rendered.push_str(&opponent);
    if condition.calculator {
        rendered.push_str(include_str!("../../prompts/calculator.txt"));
    }
    rendered
}

fn opponent_announcement(condition: &PromptCondition) -> String {
    match (condition.framing, condition.opponent) {
        (Framing::Named, OpponentAnnouncement::RationalAi) => {
            include_str!("../../prompts/opponent_rational_ai.txt").to_string()
        }
        (Framing::Named, OpponentAnnouncement::Human) => {
            include_str!("../../prompts/opponent_human.txt").to_string()
        }
        (Framing::Named, OpponentAnnouncement::SameModel) => {
            include_str!("../../prompts/opponent_same_model.txt").to_string()
        }
        (Framing::Disguised, OpponentAnnouncement::RationalAi) => {
            "You are told that the other firm is run by a rational AI agent.\n".to_string()
        }
        (Framing::Disguised, OpponentAnnouncement::Human) => {
            "You are told that the other firm is run by a human participant.\n".to_string()
        }
        (Framing::Disguised, OpponentAnnouncement::SameModel) => {
            "You are told that the other firm is run by the same language model as you.\n"
                .to_string()
        }
    }
}

pub fn render_user(
    observation: &Observation,
    condition: &PromptCondition,
    feedback: Option<&Violation>,
) -> String {
    let history = if observation.history.is_empty() {
        match condition.framing {
            Framing::Named => "No bids yet.".to_string(),
            Framing::Disguised => "No investments yet.".to_string(),
        }
    } else {
        observation
            .history
            .iter()
            .map(|bid| match condition.framing {
                Framing::Named => format!("{} bid {}", bid.player.label(), bid.amount),
                Framing::Disguised => {
                    format!("Firm {} invested {}", bid.player.index() + 1, bid.amount)
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let highest = observation.history.last().map_or(0, |bid| bid.amount);
    let legal = match highest.checked_add(1) {
        Some(minimum) if minimum <= observation.own_b => match condition.framing {
            Framing::Named => format!(
                "Legal action: bid an integer from {minimum} through {}, or drop.",
                observation.own_b
            ),
            Framing::Disguised => format!(
                "Legal action: invest an integer from {minimum} through {}, or withdraw.",
                observation.own_b
            ),
        },
        _ => match condition.framing {
            Framing::Named => "Legal action: drop. No bid amount is legal.".to_string(),
            Framing::Disguised => {
                "Legal action: withdraw. No investment amount is legal.".to_string()
            }
        },
    };
    let paid = observation
        .paid_so_far
        .map(|amount| format!("Your paid so far: {amount}."))
        .unwrap_or_default();
    let calculator_feedback = observation
        .calculator_feedback
        .iter()
        .map(|result| format!("Calculator result: {result}"))
        .collect::<Vec<_>>()
        .join("\n");
    let violation_feedback = feedback
        .map(|violation| framing_violation_feedback(violation, condition.framing))
        .unwrap_or_default();
    let rendered = include_str!("../../prompts/turn_user.txt")
        .replace("{{turn}}", &observation.turn.to_string())
        .replace("{{s}}", &observation.s.to_string())
        .replace("{{own_b}}", &observation.own_b.to_string())
        .replace("{{opponent_b}}", &observation.opponent_b.to_string())
        .replace(
            "{{value_label}}",
            if condition.framing == Framing::Named {
                "Bill value"
            } else {
                "Contract value"
            },
        )
        .replace(
            "{{limit_label}}",
            if condition.framing == Framing::Named {
                "budget"
            } else {
                "cap"
            },
        )
        .replace(
            "{{counterparty}}",
            if condition.framing == Framing::Named {
                "player's"
            } else {
                "firm's"
            },
        )
        .replace(
            "{{history_label}}",
            if condition.framing == Framing::Named {
                "bid"
            } else {
                "investment"
            },
        )
        .replace("{{history}}", &history)
        .replace("{{paid}}", &paid)
        .replace("{{legal}}", &legal)
        .replace(
            "{{action_instructions}}",
            if condition.framing == Framing::Named {
                "To bid, use {\"action\":\"bid\",\"amount\":INTEGER,\"reason\":\"BRIEF REASON\"}. To drop, use {\"action\":\"drop\",\"reason\":\"BRIEF REASON\"}."
            } else {
                "To invest, use {\"action\":\"invest\",\"amount\":INTEGER,\"reason\":\"BRIEF REASON\"}. To withdraw, use {\"action\":\"withdraw\",\"reason\":\"BRIEF REASON\"}."
            },
        )
        .replace(
            "{{calc_format}}",
            if condition.calculator {
                " To calculate, use {\"action\":\"calc\",\"expr\":\"EXPRESSION\"}."
            } else {
                ""
            },
        )
        .replace("{{calculator_feedback}}", &calculator_feedback)
        .replace("{{violation_feedback}}", &violation_feedback);
    rendered
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn framing_violation_feedback(violation: &Violation, framing: Framing) -> String {
    let guidance = match (framing, violation.kind) {
        (Framing::Named, ViolationKind::Format) => "return one valid bid/drop JSON action",
        (Framing::Named, ViolationKind::Range) => {
            "use a bid amount within the legal range shown above"
        }
        (Framing::Disguised, ViolationKind::Format) => {
            "return one valid invest/withdraw JSON action"
        }
        (Framing::Disguised, ViolationKind::Range) => {
            "use an investment amount within the legal range shown above"
        }
        (_, ViolationKind::Contradiction) => {
            "the action word controls the decision; keep all fields consistent"
        }
    };
    format!(
        "Previous output violation ({}): {guidance}.",
        violation.kind.as_str()
    )
}
