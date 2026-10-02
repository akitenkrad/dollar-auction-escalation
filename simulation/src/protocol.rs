use serde_json::Value;

use crate::bidder::Response;
use crate::prompt::Framing;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationKind {
    Format,
    Range,
    Contradiction,
}

impl ViolationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Format => "format",
            Self::Range => "range",
            Self::Contradiction => "contradiction",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    pub kind: ViolationKind,
    pub attempt: u32,
    pub message: String,
    pub raw_output: String,
}

impl Violation {
    pub fn feedback(&self) -> String {
        format!(
            "Previous output violation ({}): {}",
            self.kind.as_str(),
            self.message
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedResponse {
    pub response: Response,
    pub reason: Option<String>,
    pub contradiction: Option<String>,
    pub fenced: bool,
    pub raw_action: String,
}

pub fn parse_model_output(raw: &str) -> Result<ParsedResponse, String> {
    parse_model_output_for_framing(raw, Framing::Named)
}

pub fn parse_model_output_for_framing(
    raw: &str,
    framing: Framing,
) -> Result<ParsedResponse, String> {
    let fenced = model_output_is_fenced(raw);
    let normalized = if fenced {
        single_fenced_payload(raw).expect("fence recognition and extraction must agree")
    } else {
        raw.trim().to_string()
    };
    let value: Value =
        serde_json::from_str(&normalized).map_err(|error| format!("invalid JSON: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "the response must be a JSON object".to_string())?;
    let action = object
        .get("action")
        .and_then(Value::as_str)
        .ok_or_else(|| "action must be a string".to_string())?;
    let reason = object
        .get("reason")
        .and_then(Value::as_str)
        .map(str::to_string);
    let bid_action = match framing {
        Framing::Named => "bid",
        Framing::Disguised => "invest",
    };
    let drop_action = match framing {
        Framing::Named => "drop",
        Framing::Disguised => "withdraw",
    };
    match action {
        action if action == bid_action => {
            if reason.is_none() {
                return Err(format!("{bid_action} action requires a string reason"));
            }
            let amount = object
                .get("amount")
                .and_then(Value::as_u64)
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| format!("{bid_action} action requires an integer amount"))?;
            Ok(ParsedResponse {
                response: Response::Bid(amount),
                reason,
                contradiction: None,
                fenced,
                raw_action: action.to_string(),
            })
        }
        action if action == drop_action => {
            if reason.is_none() {
                return Err(format!("{drop_action} action requires a string reason"));
            }
            Ok(ParsedResponse {
                response: Response::Drop,
                reason,
                contradiction: object
                    .contains_key("amount")
                    .then(|| format!("{drop_action} action included an amount")),
                fenced,
                raw_action: action.to_string(),
            })
        }
        "calc" => {
            let expr = object
                .get("expr")
                .and_then(Value::as_str)
                .ok_or_else(|| "calc action requires a string expr".to_string())?;
            Ok(ParsedResponse {
                response: Response::Calc(expr.to_string()),
                reason,
                contradiction: object
                    .contains_key("amount")
                    .then(|| "calc action included an amount".to_string()),
                fenced,
                raw_action: action.to_string(),
            })
        }
        _ => Err(format!(
            "action must be {bid_action}, {drop_action}, or calc"
        )),
    }
}

pub fn raw_action_word(raw: &str) -> Option<String> {
    let normalized = if model_output_is_fenced(raw) {
        single_fenced_payload(raw)?
    } else {
        raw.trim().to_string()
    };
    serde_json::from_str::<Value>(&normalized)
        .ok()?
        .as_object()?
        .get("action")?
        .as_str()
        .map(str::to_string)
}

pub fn model_output_is_fenced(raw: &str) -> bool {
    single_fenced_payload(raw).is_some()
}

fn single_fenced_payload(raw: &str) -> Option<String> {
    let lines: Vec<_> = raw.trim().lines().collect();
    if lines.len() < 3 {
        return None;
    }
    let opening = lines.first()?.trim();
    if opening != "```" && !opening.eq_ignore_ascii_case("```json") {
        return None;
    }
    if lines.last()?.trim() != "```" {
        return None;
    }
    let inner = &lines[1..lines.len() - 1];
    if inner.iter().any(|line| line.trim().starts_with("```")) {
        return None;
    }
    Some(inner.join("\n"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolDecision {
    Action(Response),
    Retry(Violation),
    Invalid(Violation),
}

#[derive(Debug, Clone)]
pub struct ProtocolTurn {
    minimum: u32,
    maximum: u32,
    retryable_violations: u32,
}

impl ProtocolTurn {
    pub fn new(minimum: u32, maximum: u32) -> Self {
        Self {
            minimum,
            maximum,
            retryable_violations: 0,
        }
    }

    pub fn evaluate(&mut self, response: Response, raw: &str) -> ProtocolDecision {
        let violation = match &response {
            Response::Malformed(message) => Some((ViolationKind::Format, message.clone())),
            Response::Bid(amount) if *amount < self.minimum || *amount > self.maximum => Some((
                ViolationKind::Range,
                format!(
                    "amount {amount} is outside {} through {}",
                    self.minimum, self.maximum
                ),
            )),
            _ => None,
        };
        let Some((kind, message)) = violation else {
            return ProtocolDecision::Action(response);
        };
        self.retryable_violations += 1;
        let violation = Violation {
            kind,
            attempt: self.retryable_violations,
            message,
            raw_output: raw.to_string(),
        };
        if self.retryable_violations == 1 {
            ProtocolDecision::Retry(violation)
        } else {
            ProtocolDecision::Invalid(violation)
        }
    }
}
