use dollar_auction_simulation::usage::UsageSummary;
use socsim_llm::{CallMetadata, TokenUsage};

fn metadata(prompt: Option<u64>, completion: Option<u64>) -> CallMetadata {
    CallMetadata {
        model: "m".to_string(),
        endpoint: "mock://usage".to_string(),
        temperature: 0.7,
        seed: 1,
        cache_hit: false,
        usage: Some(TokenUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
        }),
    }
}

#[test]
fn usage_sums_reported_fields_and_counts_missing_prompt_values() {
    let calls = [
        metadata(Some(10), Some(3)),
        metadata(None, Some(4)),
        CallMetadata {
            usage: None,
            ..metadata(Some(99), Some(99))
        },
        metadata(Some(5), None),
    ];
    let summary = UsageSummary::from_metadata(&calls);
    assert_eq!(summary.tokens_in, 15);
    assert_eq!(summary.tokens_out, 7);
    assert_eq!(summary.n_calls, 4);
    assert_eq!(summary.n_calls_missing_prompt_tokens, 2);

    let names: Vec<_> = summary
        .metrics()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    assert_eq!(
        names,
        [
            "tokens_in",
            "tokens_out",
            "n_calls",
            "n_calls_missing_prompt_tokens"
        ]
    );
    assert!(!names.contains(&"cost_usd"));
}
