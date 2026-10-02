use std::cell::RefCell;
use std::rc::Rc;

use rand::RngCore;
use serde::Serialize;
use socsim_core::SimRng;
use socsim_llm::{CallMetadata, LlmClient, TokenUsage};

use crate::llm::model_config;
use crate::prompt::{render_system_for, render_user, PromptCondition};
use crate::protocol::{
    model_output_is_fenced, parse_model_output_for_framing, raw_action_word, Violation,
};
use crate::qre::QreTree;
use crate::seeds::llm_call_seed;
use crate::solver::best_response;
use crate::world::{Observation, Player, Role};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Response {
    Bid(u32),
    Drop,
    Calc(String),
    Malformed(String),
}

pub trait Bidder {
    fn act(&mut self, obs: &Observation, feedback: Option<&Violation>) -> Response;

    fn take_fatal_error(&mut self) -> Option<String> {
        None
    }

    fn last_raw_output(&self) -> Option<String> {
        None
    }

    fn calculator_enabled(&self) -> bool {
        false
    }

    fn take_contradiction(&mut self) -> Option<String> {
        None
    }

    fn set_engine_draw(&mut self, _draw: u64) {}

    fn take_thinking_observed(&mut self) -> bool {
        false
    }

    fn take_fenced_observed(&mut self) -> bool {
        false
    }

    fn is_llm(&self) -> bool {
        false
    }
}

#[derive(Debug, Default)]
pub struct SolverBidder;

impl SolverBidder {
    pub fn new() -> Self {
        Self
    }
}

impl Bidder for SolverBidder {
    fn act(&mut self, obs: &Observation, _feedback: Option<&Violation>) -> Response {
        let me = obs
            .history
            .iter()
            .rev()
            .find(|bid| {
                bid.player
                    == if obs.role == Role::FirstMover {
                        Player::P1
                    } else {
                        Player::P2
                    }
            })
            .map_or(0, |bid| bid.amount);
        let opp = obs
            .history
            .iter()
            .rev()
            .find(|bid| {
                bid.player
                    != if obs.role == Role::FirstMover {
                        Player::P1
                    } else {
                        Player::P2
                    }
            })
            .map_or(0, |bid| bid.amount);
        best_response(obs.s, obs.own_b, me, opp)
    }
}

pub struct QreBidder {
    lambda: f64,
    rng: SimRng,
    engine_draw: Option<u64>,
    tree: Option<QreTree>,
}

impl QreBidder {
    pub fn new(lambda: f64, seed: u64) -> Self {
        Self {
            lambda,
            rng: SimRng::from_seed(seed),
            engine_draw: None,
            tree: None,
        }
    }
}

impl Bidder for QreBidder {
    fn act(&mut self, obs: &Observation, _feedback: Option<&Violation>) -> Response {
        let own_player = if obs.role == Role::FirstMover {
            Player::P1
        } else {
            Player::P2
        };
        let me = obs
            .history
            .iter()
            .rev()
            .find(|bid| bid.player == own_player)
            .map_or(0, |bid| bid.amount);
        let opp = obs
            .history
            .iter()
            .rev()
            .find(|bid| bid.player != own_player)
            .map_or(0, |bid| bid.amount);
        let tree = self
            .tree
            .get_or_insert_with(|| QreTree::new(obs.s, obs.own_b, self.lambda));
        let probabilities = tree.action_probabilities(me, opp);
        let draw = self
            .engine_draw
            .take()
            .unwrap_or_else(|| self.rng.next_u64()) as f64
            / u64::MAX as f64;
        let mut cumulative = 0.0;
        for (action, probability) in &probabilities {
            cumulative += probability;
            if draw <= cumulative {
                return action.clone();
            }
        }
        probabilities
            .last()
            .expect("QRE always includes drop")
            .0
            .clone()
    }

    fn set_engine_draw(&mut self, draw: u64) {
        self.engine_draw = Some(draw);
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Transcript {
    pub trial: Option<usize>,
    pub turn: u64,
    pub attempt: u32,
    pub player: usize,
    pub system: String,
    pub user: String,
    pub raw_response: String,
    pub thinking: Option<String>,
    pub seed: u64,
    pub token_usage: Option<TokenUsage>,
    pub metadata: CallMetadata,
    pub think: Option<bool>,
    pub fenced: bool,
    pub raw_action: Option<String>,
    pub reason: Option<String>,
}

pub type TranscriptSink = Rc<RefCell<Vec<Transcript>>>;

pub struct LlmBidder {
    client: Box<dyn LlmClient>,
    condition: PromptCondition,
    trial_seed: u64,
    last_turn: Option<u64>,
    attempt: u32,
    transcripts: TranscriptSink,
    fatal_error: Option<String>,
    last_raw: Option<String>,
    contradiction: Option<String>,
    thinking_observed: bool,
    fenced_observed: bool,
    trial_index: Option<usize>,
}

impl LlmBidder {
    pub fn new(client: Box<dyn LlmClient>, condition: PromptCondition, trial_seed: u64) -> Self {
        Self::with_sink(
            client,
            condition,
            trial_seed,
            Rc::new(RefCell::new(Vec::new())),
        )
    }

    pub fn with_sink(
        client: Box<dyn LlmClient>,
        condition: PromptCondition,
        trial_seed: u64,
        transcripts: TranscriptSink,
    ) -> Self {
        Self {
            client,
            condition,
            trial_seed,
            last_turn: None,
            attempt: 0,
            transcripts,
            fatal_error: None,
            last_raw: None,
            contradiction: None,
            thinking_observed: false,
            fenced_observed: false,
            trial_index: None,
        }
    }

    pub fn with_trial_index(mut self, trial_index: usize) -> Self {
        self.trial_index = Some(trial_index);
        self
    }

    pub fn transcripts(&self) -> TranscriptSink {
        Rc::clone(&self.transcripts)
    }
}

impl Bidder for LlmBidder {
    fn is_llm(&self) -> bool {
        true
    }

    fn act(&mut self, obs: &Observation, feedback: Option<&Violation>) -> Response {
        if self.last_turn == Some(obs.turn) {
            self.attempt += 1;
        } else {
            self.last_turn = Some(obs.turn);
            self.attempt = 1;
        }
        let player = if obs.role == Role::FirstMover {
            Player::P1
        } else {
            Player::P2
        };
        let system = render_system_for(obs.s, obs.own_b, player, &self.condition);
        let user = render_user(obs, &self.condition, feedback);
        let seed = llm_call_seed(self.trial_seed, obs.turn, self.attempt);
        let config = model_config(system.clone(), seed);
        let sent_think = config.think;
        let response = match self.client.complete(&user, &config) {
            Ok(response) => response,
            Err(error) => {
                let message = error.to_string();
                self.fatal_error = Some(message.clone());
                return Response::Malformed(message);
            }
        };
        self.last_raw = Some(response.text.clone());
        self.thinking_observed = response.thinking.is_some();
        self.fenced_observed = model_output_is_fenced(&response.text);
        let raw_action = raw_action_word(&response.text);
        let parsed = parse_model_output_for_framing(&response.text, self.condition.framing);
        let reason = parsed
            .as_ref()
            .ok()
            .and_then(|parsed| parsed.reason.clone());
        self.transcripts.borrow_mut().push(Transcript {
            trial: self.trial_index,
            turn: obs.turn,
            attempt: self.attempt,
            player: player.index() + 1,
            system,
            user,
            raw_response: response.text.clone(),
            thinking: response.thinking.clone(),
            seed,
            token_usage: response.metadata.usage,
            metadata: response.metadata,
            think: sent_think,
            fenced: self.fenced_observed,
            raw_action,
            reason,
        });
        match parsed {
            Ok(parsed) => {
                self.contradiction = parsed.contradiction;
                parsed.response
            }
            Err(error) => Response::Malformed(error),
        }
    }

    fn take_fatal_error(&mut self) -> Option<String> {
        self.fatal_error.take()
    }

    fn last_raw_output(&self) -> Option<String> {
        self.last_raw.clone()
    }

    fn calculator_enabled(&self) -> bool {
        self.condition.calculator
    }

    fn take_contradiction(&mut self) -> Option<String> {
        self.contradiction.take()
    }

    fn take_thinking_observed(&mut self) -> bool {
        std::mem::take(&mut self.thinking_observed)
    }

    fn take_fenced_observed(&mut self) -> bool {
        std::mem::take(&mut self.fenced_observed)
    }
}
