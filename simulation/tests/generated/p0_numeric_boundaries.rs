use dollar_auction_simulation::calc::eval_integer_expression;

#[test]
fn accepts_both_i128_endpoints_and_rejects_positive_overflow() {
    assert_eq!(
        eval_integer_expression("170141183460469231731687303715884105727").unwrap(),
        i128::MAX
    );
    assert_eq!(
        eval_integer_expression("-170141183460469231731687303715884105728").unwrap(),
        i128::MIN
    );
    assert!(eval_integer_expression("170141183460469231731687303715884105728").is_err());
}

#[test]
fn empty_and_unbalanced_expressions_return_errors_without_panicking() {
    for expression in ["", "()", "(", "1 +", ")1("] {
        assert!(
            eval_integer_expression(expression).is_err(),
            "{expression:?}"
        );
    }
}

#[test]
fn two_hundred_char_expression_is_accepted_but_two_hundred_one_is_rejected() {
    let accepted = format!("1{}", "+0".repeat(99));
    assert_eq!(accepted.chars().count(), 199);
    assert_eq!(eval_integer_expression(&accepted).unwrap(), 1);
    let rejected = format!("{accepted}  ");
    assert_eq!(rejected.chars().count(), 201);
    assert!(eval_integer_expression(&rejected).is_err());
}
