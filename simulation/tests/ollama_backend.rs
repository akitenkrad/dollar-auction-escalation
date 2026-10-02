use std::sync::atomic::{AtomicUsize, Ordering};

use dollar_auction_simulation::config::ModelSpec;
use dollar_auction_simulation::llm::{
    check_model_digest, direct_ollama_client, model_config, TagsFetcher,
};
use socsim_llm::LlmClient;

struct CannedTags {
    body: String,
    fetches: AtomicUsize,
}

impl TagsFetcher for CannedTags {
    fn fetch_tags(&self, _host: &str) -> Result<String, String> {
        self.fetches.fetch_add(1, Ordering::SeqCst);
        Ok(self.body.clone())
    }
}

fn model() -> ModelSpec {
    ModelSpec {
        tag: "gemma4:26b".to_string(),
        digest_prefix: "001e5dafc3c7".to_string(),
    }
}

#[test]
fn direct_backend_is_ollama_and_request_config_is_fixed() {
    let client = direct_ollama_client("http://localhost:11434", "gemma4:26b");
    assert_eq!(client.model(), "gemma4:26b");
    assert_eq!(client.endpoint(), "http://localhost:11434/api/chat");

    let config = model_config("system".to_string(), 9182);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.seed, 9182);
    assert_eq!(config.think, Some(false));
    assert_eq!(config.system.as_deref(), Some("system"));

    let source = include_str!("../src/llm.rs");
    for forbidden in [
        "FallbackClient",
        "PromptCache",
        "build_live_client_from_settings",
    ] {
        assert!(!source.contains(forbidden));
    }
}

#[test]
fn digest_check_accepts_only_the_configured_tag_and_prefix() {
    let ok = CannedTags {
        body: r#"{"models":[{"name":"gemma4:26b","digest":"001e5dafc3c7abcdef"}]}"#.to_string(),
        fetches: AtomicUsize::new(0),
    };
    let digest = check_model_digest(&ok, "http://localhost:11434", &model()).unwrap();
    assert_eq!(digest, "001e5dafc3c7abcdef");
    assert_eq!(ok.fetches.load(Ordering::SeqCst), 1);

    let missing = CannedTags {
        body: r#"{"models":[{"name":"other:latest","digest":"001e5dafc3c7abcdef"}]}"#.to_string(),
        fetches: AtomicUsize::new(0),
    };
    let error = check_model_digest(&missing, "http://localhost:11434", &model()).unwrap_err();
    assert!(error.contains("gemma4:26b"));
    assert!(error.contains("not installed"));

    let wrong = CannedTags {
        body: r#"{"models":[{"name":"gemma4:26b","digest":"ffffffffffff0000"}]}"#.to_string(),
        fetches: AtomicUsize::new(0),
    };
    let error = check_model_digest(&wrong, "http://localhost:11434", &model()).unwrap_err();
    assert!(error.contains("digest mismatch"));
}

#[test]
fn malformed_tags_response_aborts_cleanly() {
    let tags = CannedTags {
        body: "not json".to_string(),
        fetches: AtomicUsize::new(0),
    };
    assert!(check_model_digest(&tags, "http://localhost:11434", &model()).is_err());
}
