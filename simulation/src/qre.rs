use std::collections::HashMap;

use crate::bidder::Response;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedAction {
    pub me: u32,
    pub opp: u32,
    pub action: Response,
}

impl ObservedAction {
    pub fn new(me: u32, opp: u32, action: Response) -> Self {
        Self { me, opp, action }
    }
}

#[derive(Debug, Clone, Copy)]
struct NodeValue {
    mover: f64,
    opponent: f64,
}

/// One complete agent-logit evaluation for a fixed game and lambda.
///
/// Node values are computed once. Probability queries only read the cached
/// child values and therefore never walk a subtree again.
pub struct QreTree {
    s: u32,
    b: u32,
    lambda: f64,
    values: HashMap<(u32, u32), NodeValue>,
}

impl QreTree {
    pub fn new(s: u32, b: u32, lambda: f64) -> Self {
        assert!(lambda >= 0.0 && lambda.is_finite());
        let mut values = HashMap::new();
        Self::node_value(s, b, lambda, 0, 0, &mut values);
        Self {
            s,
            b,
            lambda,
            values,
        }
    }

    pub fn action_probabilities(&self, me: u32, opp: u32) -> Vec<(Response, f64)> {
        softmax(&self.utilities(me, opp), self.lambda)
    }

    pub fn log_likelihood(&self, observed_actions: &[ObservedAction]) -> f64 {
        observed_actions
            .iter()
            .map(|observed| {
                self.action_probabilities(observed.me, observed.opp)
                    .into_iter()
                    .find_map(|(action, probability)| {
                        (action == observed.action).then_some(probability.ln())
                    })
                    .unwrap_or(f64::NEG_INFINITY)
            })
            .sum()
    }

    fn node_value(
        s: u32,
        b: u32,
        lambda: f64,
        me: u32,
        opp: u32,
        values: &mut HashMap<(u32, u32), NodeValue>,
    ) -> NodeValue {
        if let Some(value) = values.get(&(me, opp)).copied() {
            return value;
        }

        let mut actions = vec![(
            Response::Drop,
            -f64::from(me),
            if opp > 0 {
                f64::from(s) - f64::from(opp)
            } else {
                0.0
            },
        )];
        if opp < b {
            for amount in (opp + 1)..=b {
                let child = Self::node_value(s, b, lambda, opp, amount, values);
                actions.push((Response::Bid(amount), child.opponent, child.mover));
            }
        }
        let probabilities = softmax(&actions, lambda);
        let mut value = NodeValue {
            mover: 0.0,
            opponent: 0.0,
        };
        for ((_, mover, opponent), (_, probability)) in actions.iter().zip(probabilities) {
            value.mover += mover * probability;
            value.opponent += opponent * probability;
        }
        values.insert((me, opp), value);
        value
    }

    fn utilities(&self, me: u32, opp: u32) -> Vec<(Response, f64, f64)> {
        let mut actions = vec![(
            Response::Drop,
            -f64::from(me),
            if opp > 0 {
                f64::from(self.s) - f64::from(opp)
            } else {
                0.0
            },
        )];
        if opp < self.b {
            for amount in (opp + 1)..=self.b {
                let child = self
                    .values
                    .get(&(opp, amount))
                    .expect("every child value is computed when the QRE tree is built");
                actions.push((Response::Bid(amount), child.opponent, child.mover));
            }
        }
        actions
    }
}

pub struct AgentQre {
    s: u32,
    b: u32,
}

impl AgentQre {
    pub fn new(s: u32, b: u32) -> Self {
        Self { s, b }
    }

    pub fn tree(&self, lambda: f64) -> QreTree {
        QreTree::new(self.s, self.b, lambda)
    }

    pub fn action_probabilities(&self, me: u32, opp: u32, lambda: f64) -> Vec<(Response, f64)> {
        self.tree(lambda).action_probabilities(me, opp)
    }

    pub fn log_likelihood(&self, lambda: f64, observed_actions: &[ObservedAction]) -> f64 {
        self.tree(lambda).log_likelihood(observed_actions)
    }
}

fn softmax(actions: &[(Response, f64, f64)], lambda: f64) -> Vec<(Response, f64)> {
    if lambda == 0.0 {
        let probability = 1.0 / actions.len() as f64;
        return actions
            .iter()
            .map(|(action, _, _)| (action.clone(), probability))
            .collect();
    }
    let max_scaled = actions
        .iter()
        .map(|(_, utility, _)| lambda * utility)
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = actions
        .iter()
        .map(|(_, utility, _)| (lambda * utility - max_scaled).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    actions
        .iter()
        .zip(weights)
        .map(|((action, _, _), weight)| (action.clone(), weight / total))
        .collect()
}
