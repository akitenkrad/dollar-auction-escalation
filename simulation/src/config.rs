use serde::{Deserialize, Serialize};

pub const DEFAULT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_ROOT_SEED: u64 = 42;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameConfig {
    pub s: u32,
    pub b: u32,
    pub temperature: f32,
    pub display_paid_so_far: bool,
}

impl GameConfig {
    pub fn new(s: u32, b: u32) -> Self {
        Self {
            s,
            b,
            temperature: DEFAULT_TEMPERATURE,
            display_paid_so_far: false,
        }
    }
}
