use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use socsim_llm::mock::ScriptedClient;
use socsim_llm::{LlmClient, LlmConfig, OllamaClient};

use crate::config::ModelSpec;

pub const MODEL_THINK: bool = false;
pub const DEFAULT_OLLAMA_HOST: &str = "http://localhost:11434";

pub fn model_config(system: String, seed: u64) -> LlmConfig {
    let mut config = LlmConfig::deterministic()
        .with_temperature(crate::config::DEFAULT_TEMPERATURE)
        .with_seed(seed)
        .with_system(system);
    config.think = Some(MODEL_THINK);
    config
}

pub fn scripted_client_from_file(path: &Path) -> Result<Box<dyn LlmClient>, String> {
    let text = fs::read_to_string(path).map_err(|error| {
        format!(
            "failed to read scripted responses {}: {error}",
            path.display()
        )
    })?;
    let lines: VecDeque<String> = text.lines().map(str::to_string).collect();
    let queue = Arc::new(Mutex::new(lines));
    let answers = Arc::clone(&queue);
    Ok(Box::new(ScriptedClient::new("scripted", move |_| {
        answers
            .lock()
            .expect("scripted response queue lock was poisoned")
            .pop_front()
            .unwrap_or_else(|| "not json".to_string())
    })))
}

pub fn direct_ollama_client(host: &str, model: &str) -> Box<dyn LlmClient> {
    Box::new(OllamaClient::new(host, model))
}

pub fn ollama_host() -> String {
    std::env::var("OLLAMA_HOST").unwrap_or_else(|_| DEFAULT_OLLAMA_HOST.to_string())
}

pub trait TagsFetcher {
    fn fetch_tags(&self, host: &str) -> Result<String, String>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct HttpTagsFetcher;

impl TagsFetcher for HttpTagsFetcher {
    fn fetch_tags(&self, host: &str) -> Result<String, String> {
        let endpoint = format!("{}/api/tags", host.trim_end_matches('/'));
        let response = ureq::get(&endpoint)
            .call()
            .map_err(|error| format!("failed to query Ollama tags at {endpoint}: {error}"))?;
        response
            .into_string()
            .map_err(|error| format!("failed to read Ollama tags at {endpoint}: {error}"))
    }
}

pub fn check_model_digest(
    fetcher: &dyn TagsFetcher,
    host: &str,
    model: &ModelSpec,
) -> Result<String, String> {
    let raw = fetcher.fetch_tags(host)?;
    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|error| format!("invalid JSON from Ollama /api/tags: {error}"))?;
    let models = value
        .get("models")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "Ollama /api/tags response has no models array".to_string())?;
    let Some(found) = models.iter().find(|entry| {
        entry.get("name").and_then(serde_json::Value::as_str) == Some(model.tag.as_str())
    }) else {
        return Err(format!("Ollama model {} is not installed", model.tag));
    };
    let digest = found
        .get("digest")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("Ollama model {} has no digest", model.tag))?;
    if !digest.starts_with(&model.digest_prefix) {
        return Err(format!(
            "Ollama digest mismatch for {}: expected prefix {}, found {}",
            model.tag, model.digest_prefix, digest
        ));
    }
    Ok(digest.to_string())
}
