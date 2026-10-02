use std::cell::RefCell;
use std::rc::Rc;

use rand::RngCore;
use serde_json::json;
use socsim_core::{Mechanism, Phase, Recorder, Result as SocsimResult, SocsimError, StepContext};
use socsim_engine::{SequentialScheduler, SimulationBuilder};

use crate::bidder::{Bidder, Response, Transcript, TranscriptSink};
use crate::calc::{CalcOutcome, CalculatorTurn};
use crate::protocol::{ProtocolDecision, ProtocolTurn, Violation, ViolationKind};
use crate::rules::apply_response;
use crate::solver::proposition_first_bid;
use crate::world::{AuctionWorld, EndReason, Player, Status};

#[derive(Debug, Clone)]
pub struct TrialResult {
    pub outcome: String,
    pub payoffs: [i64; 2],
    pub bids: Vec<crate::world::Bid>,
    pub turns: u64,
    pub censored: bool,
    pub first_bid_deviation: i64,
    pub first_end: bool,
    pub waste: f64,
    pub invalid_turn: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub struct TrialRunConfig {
    pub s: u32,
    pub b: u32,
    pub display_paid_so_far: bool,
    pub engine_seed: u64,
    pub paraphrase_id: u8,
}

struct SharedBidders(Rc<RefCell<[Box<dyn Bidder>; 2]>>);

pub struct TakeTurn {
    bidders: SharedBidders,
}

impl TakeTurn {
    fn new(bidders: [Box<dyn Bidder>; 2]) -> Self {
        Self {
            bidders: SharedBidders(Rc::new(RefCell::new(bidders))),
        }
    }

    fn record_violation(
        ctx: &mut StepContext<'_, AuctionWorld>,
        player: Player,
        violation: &Violation,
    ) {
        ctx.recorder.record_event(
            ctx.clock.t(),
            "violation",
            json!({
                "unit_id": "trial-0",
                "player": player.index() + 1,
                "kind": violation.kind.as_str(),
                "attempt": violation.attempt,
                "raw_output": violation.raw_output,
                "message": violation.message,
            }),
        );
    }
}

impl Mechanism<AuctionWorld> for TakeTurn {
    fn name(&self) -> &str {
        "take_turn"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }

    fn apply(
        &mut self,
        _phase: Phase,
        ctx: &mut StepContext<'_, AuctionWorld>,
    ) -> SocsimResult<()> {
        if ctx.world.status != Status::Ongoing {
            return Ok(());
        }
        let player = ctx.world.to_move;
        let index = player.index();
        let minimum = ctx.world.highest_bid().saturating_add(1);
        let maximum = ctx.world.b[index];
        let mut protocol = ProtocolTurn::new(minimum, maximum);
        let mut calculator = CalculatorTurn::new();
        let mut observation = ctx.world.observation_for(player);
        let mut feedback: Option<Violation> = None;

        loop {
            let mut bidders = self.bidders.0.borrow_mut();
            let bidder = &mut bidders[index];
            bidder.set_engine_draw(ctx.rng.next_u64());
            let mut response = bidder.act(&observation, feedback.as_ref());
            if bidder.take_thinking_observed() {
                ctx.world.thinking_calls += 1;
            }
            if bidder.take_fenced_observed() {
                ctx.world.fenced_calls += 1;
            }
            if let Some(error) = bidder.take_fatal_error() {
                return Err(SocsimError::Mechanism(format!(
                    "LLM backend failed: {error}"
                )));
            }
            let raw = bidder
                .last_raw_output()
                .unwrap_or_else(|| format!("{response:?}"));
            if let Some(message) = bidder.take_contradiction() {
                let violation = Violation {
                    kind: ViolationKind::Contradiction,
                    attempt: 1,
                    message,
                    raw_output: raw.clone(),
                };
                Self::record_violation(ctx, player, &violation);
            }
            if let Response::Calc(expression) = response {
                if !bidder.calculator_enabled() {
                    response = Response::Malformed(
                        "calc action is unavailable in this condition".to_string(),
                    );
                } else {
                    match calculator.evaluate(&expression) {
                        CalcOutcome::Result(result) => {
                            observation.calculator_feedback.push(result);
                            feedback = None;
                            continue;
                        }
                        CalcOutcome::Violation(violation) => {
                            response = Response::Malformed(violation.message);
                        }
                    }
                }
            }
            drop(bidders);

            match protocol.evaluate(response, &raw) {
                ProtocolDecision::Action(action) => {
                    apply_response(ctx.world, action, feedback.as_ref().map_or(1, |_| 2));
                    let t = ctx.world.bids.len() as u64;
                    ctx.recorder.record_event(
                        t,
                        "observation",
                        json!({
                            "unit_id": "trial-0",
                            "player": player.index() + 1,
                            "highest_bid": ctx.world.highest_bid(),
                        }),
                    );
                    return Ok(());
                }
                ProtocolDecision::Retry(violation) => {
                    Self::record_violation(ctx, player, &violation);
                    feedback = Some(violation);
                }
                ProtocolDecision::Invalid(violation) => {
                    Self::record_violation(ctx, player, &violation);
                    ctx.recorder.record_event(
                        ctx.world.bids.len() as u64,
                        "observation",
                        json!({
                            "unit_id": "trial-0",
                            "player": player.index() + 1,
                            "highest_bid": ctx.world.highest_bid(),
                            "invalid": true,
                        }),
                    );
                    ctx.world.status = Status::Ended {
                        winner: None,
                        reason: EndReason::Invalid,
                    };
                    ctx.world.invalid_turn = Some(ctx.clock.t());
                    return Ok(());
                }
            }
        }
    }
}

pub struct Settle;

impl Mechanism<AuctionWorld> for Settle {
    fn name(&self) -> &str {
        "settle"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }

    fn apply(
        &mut self,
        _phase: Phase,
        ctx: &mut StepContext<'_, AuctionWorld>,
    ) -> SocsimResult<()> {
        let Status::Ended { reason, .. } = ctx.world.status else {
            return Ok(());
        };
        let t = ctx.world.bids.len() as u64;
        let censored = reason == EndReason::CapReached;
        // A bidder may jump directly to the cap, so cap reached can have t < b.
        // Runvault requires t == budget for censored terminal events.
        let budget = if censored {
            t
        } else {
            u64::from(ctx.world.b[0])
        };
        let x1 = ctx.world.bids.first().map_or(0, |bid| bid.amount);
        let x1_star = proposition_first_bid(ctx.world.s, ctx.world.b[0]);
        let last_total = ctx.world.last_bid(Player::P1) + ctx.world.last_bid(Player::P2);
        let waste = f64::from(last_total) / f64::from(ctx.world.s);
        ctx.recorder.record_event(
            t,
            "trial",
            json!({
                "unit_id": "trial-0",
                "first_bid_deviation": i64::from(x1) - i64::from(x1_star),
                "first_end": ctx.world.bids.len() <= 1,
                "T": t,
                "waste": waste,
                "paraphrase_id": ctx.world.paraphrase_id,
                "thinking_calls": ctx.world.thinking_calls,
                "fenced_calls": ctx.world.fenced_calls,
            }),
        );
        ctx.recorder.record_event(
            t,
            "terminal",
            json!({
                "unit_id": "trial-0",
                "outcome": ctx.world.outcome(),
                "censored": censored,
                "budget": budget,
                "h": ctx.world.highest_bid(),
                "turn": ctx.world.invalid_turn.unwrap_or(ctx.clock.t()),
            }),
        );
        ctx.request_stop();
        Ok(())
    }
}

pub fn play_trial(
    s: u32,
    b: u32,
    p1: Box<dyn Bidder>,
    p2: Box<dyn Bidder>,
    display_paid_so_far: bool,
    _transcripts: TranscriptSink,
) -> Result<TrialResult, String> {
    run_trial_with_recorder(
        TrialRunConfig {
            s,
            b,
            display_paid_so_far,
            engine_seed: 0,
            paraphrase_id: 0,
        },
        p1,
        p2,
        Box::new(socsim_core::NullRecorder),
    )
}

pub fn run_trial_with_recorder(
    config: TrialRunConfig,
    p1: Box<dyn Bidder>,
    p2: Box<dyn Bidder>,
    recorder: Box<dyn Recorder>,
) -> Result<TrialResult, String> {
    let mut world = AuctionWorld::new(config.s, [config.b, config.b], config.display_paid_so_far);
    world.paraphrase_id = config.paraphrase_id;
    let mut simulation = SimulationBuilder::new(world)
        .scheduler(Box::new(SequentialScheduler))
        .seed(config.engine_seed)
        .add_mechanism(Box::new(TakeTurn::new([p1, p2])))
        .add_mechanism(Box::new(Settle))
        .recorder(recorder)
        .build();
    simulation.run().map_err(|error| error.to_string())?;
    let world = simulation.world();
    let x1 = world.bids.first().map_or(0, |bid| bid.amount);
    let x1_star = proposition_first_bid(config.s, config.b);
    let last_total = world.last_bid(Player::P1) + world.last_bid(Player::P2);
    Ok(TrialResult {
        outcome: world.outcome().to_string(),
        payoffs: world.payoffs(),
        bids: world.bids.clone(),
        turns: world.clock.t(),
        censored: matches!(
            world.status,
            Status::Ended {
                reason: EndReason::CapReached,
                ..
            }
        ),
        first_bid_deviation: i64::from(x1) - i64::from(x1_star),
        first_end: world.bids.len() <= 1,
        waste: f64::from(last_total) / f64::from(config.s),
        invalid_turn: world.invalid_turn,
    })
}

pub fn empty_transcript_sink() -> Rc<RefCell<Vec<Transcript>>> {
    Rc::new(RefCell::new(Vec::new()))
}
