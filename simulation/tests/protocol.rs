use dollar_auction_simulation::bidder::Response;
use dollar_auction_simulation::calc::{eval_integer_expression, CalcOutcome, CalculatorTurn};
use dollar_auction_simulation::prompt::Framing;
use dollar_auction_simulation::protocol::{
    parse_model_output, parse_model_output_for_framing, ProtocolDecision, ProtocolTurn,
    ViolationKind,
};

#[test]
fn parser_accepts_valid_json_and_rejects_missing_fields() {
    let bid = parse_model_output(r#"{"action":"bid","amount":12,"reason":"test"}"#).unwrap();
    assert_eq!(bid.response, Response::Bid(12));
    assert_eq!(bid.reason.as_deref(), Some("test"));
    assert!(!bid.fenced);
    assert!(parse_model_output("not json").is_err());
    assert!(parse_model_output(r#"{"action":"bid","reason":"missing"}"#).is_err());
    assert!(parse_model_output(r#"{"action":"drop"}"#).is_err());
}

#[test]
fn parser_accepts_exactly_one_markdown_fence() {
    for opening in ["```", "```json", "```JSON"] {
        let raw =
            format!("{opening}\n{{\"action\":\"drop\",\"amount\":0,\"reason\":\"done\"}}\n```");
        let parsed = parse_model_output(&raw).unwrap();
        assert_eq!(parsed.response, Response::Drop);
        assert!(parsed.fenced);
    }
}

#[test]
fn parser_enforces_the_framing_vocabulary() {
    let named_bid = parse_model_output_for_framing(
        r#"{"action":"bid","amount":12,"reason":"go"}"#,
        Framing::Named,
    )
    .unwrap();
    let named_drop =
        parse_model_output_for_framing(r#"{"action":"drop","reason":"stop"}"#, Framing::Named)
            .unwrap();
    assert_eq!(named_bid.response, Response::Bid(12));
    assert_eq!(named_bid.raw_action, "bid");
    assert_eq!(named_drop.response, Response::Drop);
    assert!(parse_model_output_for_framing(
        r#"{"action":"invest","amount":12,"reason":"go"}"#,
        Framing::Named
    )
    .is_err());
    assert!(parse_model_output_for_framing(
        r#"{"action":"withdraw","reason":"stop"}"#,
        Framing::Named
    )
    .is_err());

    let disguised_invest = parse_model_output_for_framing(
        r#"{"action":"invest","amount":12,"reason":"go"}"#,
        Framing::Disguised,
    )
    .unwrap();
    let disguised_withdraw = parse_model_output_for_framing(
        r#"{"action":"withdraw","reason":"stop"}"#,
        Framing::Disguised,
    )
    .unwrap();
    assert_eq!(disguised_invest.response, Response::Bid(12));
    assert_eq!(disguised_invest.raw_action, "invest");
    assert_eq!(disguised_withdraw.response, Response::Drop);
    assert!(parse_model_output_for_framing(
        r#"{"action":"bid","amount":12,"reason":"go"}"#,
        Framing::Disguised
    )
    .is_err());
    assert!(parse_model_output_for_framing(
        r#"{"action":"drop","reason":"stop"}"#,
        Framing::Disguised
    )
    .is_err());
}

#[test]
fn framing_vocabulary_accepts_calculator_and_fenced_json() {
    for (framing, action, expected) in [
        (Framing::Named, "drop", Response::Drop),
        (Framing::Disguised, "withdraw", Response::Drop),
    ] {
        let raw = format!("```json\n{{\"action\":\"{action}\",\"reason\":\"done\"}}\n```");
        let parsed = parse_model_output_for_framing(&raw, framing).unwrap();
        assert_eq!(parsed.response, expected);
        assert!(parsed.fenced);
    }
    for framing in [Framing::Named, Framing::Disguised] {
        assert_eq!(
            parse_model_output_for_framing(r#"{"action":"calc","expr":"17 + 5"}"#, framing)
                .unwrap()
                .response,
            Response::Calc("17 + 5".to_string())
        );
    }
}

#[test]
fn parser_rejects_text_multiple_fences_and_unclosed_fences() {
    let prose = "Here is the result:\n```json\n{\"action\":\"drop\",\"reason\":\"done\"}\n```";
    let two = "```json\n{\"action\":\"drop\",\"reason\":\"done\"}\n```\n```json\n{\"action\":\"drop\",\"reason\":\"again\"}\n```";
    let unclosed = "```json\n{\"action\":\"drop\",\"reason\":\"done\"}";
    assert!(parse_model_output(prose).is_err());
    assert!(parse_model_output(two).is_err());
    assert!(parse_model_output(unclosed).is_err());
}

#[test]
fn action_wins_when_extra_fields_contradict_it() {
    let parsed = parse_model_output(r#"{"action":"drop","amount":12,"reason":"done"}"#).unwrap();
    assert_eq!(parsed.response, Response::Drop);
    assert_eq!(
        parsed.contradiction.as_deref(),
        Some("drop action included an amount")
    );
}

#[test]
fn first_retryable_violation_reprompts_and_second_ends_invalid() {
    let mut turn = ProtocolTurn::new(11, 20);
    let first = turn.evaluate(Response::Malformed("bad".into()), "bad");
    assert!(matches!(first, ProtocolDecision::Retry(v) if v.kind == ViolationKind::Format));
    let second = turn.evaluate(Response::Bid(10), r#"{"action":"bid","amount":10}"#);
    assert!(matches!(second, ProtocolDecision::Invalid(ref v) if v.kind == ViolationKind::Range));
    assert!(!matches!(second, ProtocolDecision::Action(Response::Drop)));
}

#[test]
fn calculator_supports_integer_grammar_and_euclidean_operations() {
    assert_eq!(eval_integer_expression("(17 + 5) * 3").unwrap(), 66);
    assert_eq!(eval_integer_expression("-7 / 3").unwrap(), -3);
    assert_eq!(eval_integer_expression("-7 % 3").unwrap(), 2);
    assert!(eval_integer_expression("2 ** 8").is_err());
    assert!(eval_integer_expression(&"1".repeat(201)).is_err());
}

#[test]
fn calculator_errors_are_results_but_the_fourth_request_is_a_format_violation() {
    let mut calc = CalculatorTurn::new();
    assert!(matches!(calc.evaluate("1 / 0"), CalcOutcome::Result(s) if s.starts_with("error:")));
    assert_eq!(calc.evaluate("1 + 1"), CalcOutcome::Result("2".into()));
    assert_eq!(calc.evaluate("2 + 2"), CalcOutcome::Result("4".into()));
    assert!(
        matches!(calc.evaluate("3 + 3"), CalcOutcome::Violation(v) if v.kind == ViolationKind::Format)
    );
}
