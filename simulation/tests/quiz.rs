use dollar_auction_simulation::prompt::Framing;
use dollar_auction_simulation::quiz::{parse_quiz_answer, render_quiz_prompt, score_quiz_answer};

#[test]
fn quiz_accepts_plain_and_fenced_answers_and_scores_both_anchors() {
    let plain = parse_quiz_answer(r#"{"first_bid":52,"reason":"calculated"}"#).unwrap();
    assert_eq!(plain.first_bid, 52);
    assert!(!plain.fenced);
    let fenced =
        parse_quiz_answer("```JSON\n{\"first_bid\":50,\"reason\":\"tie-aware\"}\n```").unwrap();
    assert_eq!(fenced.first_bid, 50);
    assert!(fenced.fenced);

    let exact = score_quiz_answer(100, 250, Ok(plain));
    assert!(exact.correct_drop_tie);
    assert!(!exact.correct_mixed_tie);
    let mixed = score_quiz_answer(100, 250, Ok(fenced));
    assert!(!mixed.correct_drop_tie);
    assert!(mixed.correct_mixed_tie);
    let invalid = score_quiz_answer(100, 250, parse_quiz_answer("not json"));
    assert!(invalid.invalid);
}

#[test]
fn quiz_prompts_ask_the_target_without_leaking_formula_or_answers() {
    for framing in [Framing::Named, Framing::Disguised] {
        for paraphrase in 0..3 {
            for b in [100, 250] {
                let prompt = render_quiz_prompt(100, b, framing, paraphrase);
                let combined = format!("{}\n{}", prompt.system, prompt.user).to_lowercase();
                assert!(combined.contains("subgame-perfect"));
                for forbidden in ["mod", "(b-1)", "(b - 1)"] {
                    assert!(
                        !combined.contains(forbidden),
                        "found {forbidden}: {combined}"
                    );
                }
                let answers = if b == 100 { [1, 1] } else { [52, 50] };
                for answer in answers {
                    assert!(
                        !contains_standalone_number(&combined, answer),
                        "leaked {answer}: {combined}"
                    );
                }
            }
        }
    }
}

fn contains_standalone_number(text: &str, number: u32) -> bool {
    text.split(|character: char| !character.is_ascii_digit())
        .any(|token| token == number.to_string())
}
