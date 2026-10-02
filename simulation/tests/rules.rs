use dollar_auction_simulation::bidder::Response;
use dollar_auction_simulation::rules::{apply_response, legal_bid_range, payoffs, RuleResult};
use dollar_auction_simulation::world::{AuctionWorld, EndReason, Player, Status};

#[test]
fn legal_range_starts_above_the_high_bid_and_ends_at_the_cap() {
    let mut world = AuctionWorld::new(100, [250, 250], false);
    assert_eq!(legal_bid_range(&world), Some(1..=250));
    assert_eq!(
        apply_response(&mut world, Response::Bid(52), 1),
        RuleResult::Accepted
    );
    assert_eq!(legal_bid_range(&world), Some(53..=250));
    assert!(apply_response(&mut world, Response::Bid(52), 1).is_range_violation());
    assert!(apply_response(&mut world, Response::Bid(251), 1).is_range_violation());
}

#[test]
fn a_bid_at_the_cap_forces_the_other_player_to_drop() {
    let mut world = AuctionWorld::new(10, [12, 12], false);
    assert_eq!(
        apply_response(&mut world, Response::Bid(12), 1),
        RuleResult::Accepted
    );
    assert_eq!(legal_bid_range(&world), None);
    assert_eq!(
        apply_response(&mut world, Response::Drop, 1),
        RuleResult::Accepted
    );
    assert!(matches!(
        world.status,
        Status::Ended {
            winner: Some(Player::P1),
            reason: EndReason::CapReached,
        }
    ));
}

#[test]
fn bid_limit_stops_after_the_bid_and_preserves_budget_cap_precedence() {
    let mut turn_capped = AuctionWorld::new(10, [12, 12], false).with_max_bids(Some(2));
    apply_response(&mut turn_capped, Response::Bid(1), 1);
    assert_eq!(turn_capped.status, Status::Ongoing);
    apply_response(&mut turn_capped, Response::Bid(2), 1);
    assert!(matches!(
        turn_capped.status,
        Status::Ended {
            winner: Some(Player::P2),
            reason: EndReason::TurnCap,
        }
    ));
    assert_eq!(turn_capped.outcome(), "turn_cap");

    let mut budget_capped = AuctionWorld::new(10, [12, 12], false).with_max_bids(Some(1));
    apply_response(&mut budget_capped, Response::Bid(12), 1);
    assert!(matches!(
        budget_capped.status,
        Status::Ended {
            winner: Some(Player::P1),
            reason: EndReason::CapReached,
        }
    ));
    assert_eq!(budget_capped.outcome(), "p1_win");
}

#[test]
fn both_players_pay_their_last_bid_and_never_bid_means_zero() {
    assert_eq!(payoffs(100, [52, 73], Some(Player::P2)), [-52, 27]);
    assert_eq!(payoffs(100, [52, 0], Some(Player::P1)), [48, 0]);
}

#[test]
fn first_mover_drop_is_no_bid_with_zero_payoffs() {
    let mut world = AuctionWorld::new(100, [250, 250], false);
    assert_eq!(
        apply_response(&mut world, Response::Drop, 1),
        RuleResult::Accepted
    );
    assert_eq!(world.outcome(), "no_bid");
    assert_eq!(world.payoffs(), [0, 0]);
}
