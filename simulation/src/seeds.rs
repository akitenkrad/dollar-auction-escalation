use socsim_core::derive_seed;

pub fn trial_seed(root: u64, cell_index: u64, trial_index: u64) -> u64 {
    derive_seed(root, &[cell_index, trial_index])
}

pub fn llm_call_seed(trial_seed: u64, turn: u64, attempt: u32) -> u64 {
    derive_seed(trial_seed, &[turn, u64::from(attempt)])
}
