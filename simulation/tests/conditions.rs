use dollar_auction_simulation::conditions::{auxiliary_cells, paraphrase_schedule, primary_cells};
use dollar_auction_simulation::config::{GameConfig, DEFAULT_TEMPERATURE};

#[test]
fn primary_design_has_twelve_cells() {
    let cells = primary_cells();
    assert_eq!(cells.len(), 12);
    assert_eq!(cells.iter().filter(|c| c.b == 100).count(), 6);
    assert_eq!(cells.iter().filter(|c| c.b == 250).count(), 6);
}

#[test]
fn paraphrases_are_balanced_ten_ten_ten() {
    let schedule = paraphrase_schedule(30);
    assert_eq!(schedule.iter().filter(|&&p| p == 0).count(), 10);
    assert_eq!(schedule.iter().filter(|&&p| p == 1).count(), 10);
    assert_eq!(schedule.iter().filter(|&&p| p == 2).count(), 10);
}

#[test]
fn auxiliary_design_contains_solver_and_calculator_cells() {
    let cells = auxiliary_cells();
    assert!(cells.iter().any(|c| c.llm_vs_solver));
    assert!(cells.iter().any(|c| c.calculator));
}

#[test]
fn pilot_game_defaults_are_common_budget_and_temperature_point_seven() {
    let config = GameConfig::new(100, 250);
    assert_eq!(config.s, 100);
    assert_eq!(config.b, 250);
    assert_eq!(config.temperature, DEFAULT_TEMPERATURE);
    assert!(!config.display_paid_so_far);
}
