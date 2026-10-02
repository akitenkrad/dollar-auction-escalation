use std::ops::RangeInclusive;

use crate::bidder::Response;
use crate::world::{AuctionWorld, Bid, EndReason, Player, Status};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleResult {
    Accepted,
    RangeViolation { minimum: u32, maximum: u32 },
}

impl RuleResult {
    pub fn is_range_violation(&self) -> bool {
        matches!(self, Self::RangeViolation { .. })
    }
}

pub fn legal_bid_range(world: &AuctionWorld) -> Option<RangeInclusive<u32>> {
    let minimum = world.highest_bid().checked_add(1)?;
    let maximum = world.b[world.to_move.index()];
    (minimum <= maximum).then_some(minimum..=maximum)
}

pub fn apply_response(world: &mut AuctionWorld, response: Response, attempt: u32) -> RuleResult {
    match response {
        Response::Drop => {
            let winner = world.bids.last().map(|bid| bid.player);
            let reason =
                if world.highest_bid() == world.b[world.to_move.index()] && winner.is_some() {
                    EndReason::CapReached
                } else {
                    EndReason::Drop
                };
            world.status = Status::Ended { winner, reason };
            RuleResult::Accepted
        }
        Response::Bid(amount) => {
            let minimum = world.highest_bid().saturating_add(1);
            let maximum = world.b[world.to_move.index()];
            if amount < minimum || amount > maximum {
                return RuleResult::RangeViolation { minimum, maximum };
            }
            world.bids.push(Bid {
                player: world.to_move,
                amount,
                attempt,
            });
            world.to_move = world.to_move.other();
            RuleResult::Accepted
        }
        Response::Calc(_) | Response::Malformed(_) => RuleResult::RangeViolation {
            minimum: world.highest_bid().saturating_add(1),
            maximum: world.b[world.to_move.index()],
        },
    }
}

pub fn payoffs(s: u32, last_bids: [u32; 2], winner: Option<Player>) -> [i64; 2] {
    let mut result = [-(i64::from(last_bids[0])), -(i64::from(last_bids[1]))];
    if let Some(player) = winner {
        result[player.index()] = i64::from(s) - i64::from(last_bids[player.index()]);
    }
    result
}
