use socsim_core::{AgentId, SimClock, WorldState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    P1,
    P2,
}

impl Player {
    pub fn index(self) -> usize {
        match self {
            Self::P1 => 0,
            Self::P2 => 1,
        }
    }

    pub fn other(self) -> Self {
        match self {
            Self::P1 => Self::P2,
            Self::P2 => Self::P1,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::P1 => "Player 1",
            Self::P2 => "Player 2",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    FirstMover,
    SecondMover,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bid {
    pub player: Player,
    pub amount: u32,
    pub attempt: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    Drop,
    CapReached,
    Invalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ongoing,
    Ended {
        winner: Option<Player>,
        reason: EndReason,
    },
}

#[derive(Debug, Clone)]
pub struct Observation {
    pub s: u32,
    pub own_b: u32,
    pub opponent_b: u32,
    pub history: Vec<Bid>,
    pub role: Role,
    pub turn: u64,
    pub paid_so_far: Option<u32>,
    pub calculator_feedback: Vec<String>,
}

#[derive(Debug)]
pub struct AuctionWorld {
    pub s: u32,
    pub b: [u32; 2],
    pub bids: Vec<Bid>,
    pub to_move: Player,
    pub status: Status,
    pub clock: SimClock,
    pub agent_ids: [AgentId; 2],
    pub display_paid_so_far: bool,
    pub invalid_turn: Option<u64>,
    pub paraphrase_id: u8,
    pub thinking_calls: u64,
    pub fenced_calls: u64,
}

impl AuctionWorld {
    pub fn new(s: u32, b: [u32; 2], display_paid_so_far: bool) -> Self {
        let max_turns = u64::from(b[0].max(b[1])) * 2 + 4;
        Self {
            s,
            b,
            bids: Vec::new(),
            to_move: Player::P1,
            status: Status::Ongoing,
            clock: SimClock::new(max_turns),
            agent_ids: [AgentId(0), AgentId(1)],
            display_paid_so_far,
            invalid_turn: None,
            paraphrase_id: 0,
            thinking_calls: 0,
            fenced_calls: 0,
        }
    }

    pub fn highest_bid(&self) -> u32 {
        self.bids.last().map_or(0, |bid| bid.amount)
    }

    pub fn last_bid(&self, player: Player) -> u32 {
        self.bids
            .iter()
            .rev()
            .find(|bid| bid.player == player)
            .map_or(0, |bid| bid.amount)
    }

    pub fn observation_for(&self, player: Player) -> Observation {
        Observation {
            s: self.s,
            own_b: self.b[player.index()],
            opponent_b: self.b[player.other().index()],
            history: self.bids.clone(),
            role: if player == Player::P1 {
                Role::FirstMover
            } else {
                Role::SecondMover
            },
            turn: self.clock.t(),
            paid_so_far: self.display_paid_so_far.then(|| self.last_bid(player)),
            calculator_feedback: Vec::new(),
        }
    }

    pub fn winner(&self) -> Option<Player> {
        match self.status {
            Status::Ended { winner, .. } => winner,
            Status::Ongoing => None,
        }
    }

    pub fn outcome(&self) -> &'static str {
        match self.status {
            Status::Ended {
                reason: EndReason::Invalid,
                ..
            } => "invalid",
            Status::Ended { winner: None, .. } => "no_bid",
            Status::Ended {
                winner: Some(Player::P1),
                ..
            } => "p1_win",
            Status::Ended {
                winner: Some(Player::P2),
                ..
            } => "p2_win",
            Status::Ongoing => "ongoing",
        }
    }

    pub fn payoffs(&self) -> [i64; 2] {
        crate::rules::payoffs(
            self.s,
            [self.last_bid(Player::P1), self.last_bid(Player::P2)],
            self.winner(),
        )
    }
}

impl WorldState for AuctionWorld {
    fn agent_ids(&self) -> Vec<AgentId> {
        self.agent_ids.to_vec()
    }

    fn clock(&self) -> &SimClock {
        &self.clock
    }

    fn clock_mut(&mut self) -> &mut SimClock {
        &mut self.clock
    }
}
