use socsim_llm::CallMetadata;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UsageSummary {
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub n_calls: u64,
    pub n_calls_missing_prompt_tokens: u64,
}

impl UsageSummary {
    pub fn from_metadata(calls: &[CallMetadata]) -> Self {
        let mut summary = Self::default();
        for call in calls {
            summary.n_calls += 1;
            match call.usage.and_then(|usage| usage.prompt_tokens) {
                Some(tokens) => summary.tokens_in += tokens,
                None => summary.n_calls_missing_prompt_tokens += 1,
            }
            if let Some(tokens) = call.usage.and_then(|usage| usage.completion_tokens) {
                summary.tokens_out += tokens;
            }
        }
        summary
    }

    pub fn metrics(self) -> Vec<(&'static str, f64)> {
        vec![
            ("tokens_in", self.tokens_in as f64),
            ("tokens_out", self.tokens_out as f64),
            ("n_calls", self.n_calls as f64),
            (
                "n_calls_missing_prompt_tokens",
                self.n_calls_missing_prompt_tokens as f64,
            ),
        ]
    }
}
