use crate::prompt::{Framing, OpponentAnnouncement};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExperimentCell {
    pub framing: Framing,
    pub opponent: OpponentAnnouncement,
    pub b: u32,
    pub calculator: bool,
    pub llm_vs_solver: bool,
}

pub fn primary_cells() -> Vec<ExperimentCell> {
    let mut cells = Vec::new();
    for framing in [Framing::Named, Framing::Disguised] {
        for opponent in [
            OpponentAnnouncement::RationalAi,
            OpponentAnnouncement::Human,
            OpponentAnnouncement::SameModel,
        ] {
            for b in [100, 250] {
                cells.push(ExperimentCell {
                    framing,
                    opponent,
                    b,
                    calculator: false,
                    llm_vs_solver: false,
                });
            }
        }
    }
    cells
}

pub fn auxiliary_cells() -> Vec<ExperimentCell> {
    let mut cells = Vec::new();
    for b in [100, 250] {
        cells.push(ExperimentCell {
            framing: Framing::Disguised,
            opponent: OpponentAnnouncement::RationalAi,
            b,
            calculator: true,
            llm_vs_solver: false,
        });
        cells.push(ExperimentCell {
            framing: Framing::Disguised,
            opponent: OpponentAnnouncement::RationalAi,
            b,
            calculator: false,
            llm_vs_solver: true,
        });
    }
    cells
}

pub fn paraphrase_schedule(trials: usize) -> Vec<u8> {
    (0..trials).map(|index| (index % 3) as u8).collect()
}
