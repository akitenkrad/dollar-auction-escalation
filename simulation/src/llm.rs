use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use socsim_llm::mock::ScriptedClient;
use socsim_llm::{LlmClient, LlmConfig, OllamaClient};

pub const MODEL_THINK: bool = false;

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
