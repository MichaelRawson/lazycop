use crate::options::Options;
use crate::prelude::*;
#[cfg(feature = "nn")]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct UCTValue(f32);

impl UCTValue {
    fn new(value: f32) -> Self {
        debug_assert!(value.is_finite());
        debug_assert!(value >= 0.0);
        Self(value)
    }
}

impl Eq for UCTValue {}

impl Ord for UCTValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        some(self.partial_cmp(other))
    }
}

pub(crate) struct UCTNode {
    parent: Id<UCTNode>,
    children: Range<UCTNode>,
    rule: Rule,
    atomic_visits: AtomicU32,
    score: u32,
    prior_bits: AtomicU32,
    closed: bool,
    #[cfg(feature = "nn")]
    evaluated: AtomicBool,
}

impl UCTNode {
    fn new(parent: Id<UCTNode>, rule: Rule, score: u32, prior: f32) -> Self {
        let children = Range::new(Id::default(), Id::default());
        let atomic_visits = AtomicU32::new(1);
        let prior_bits = AtomicU32::new(prior.to_bits());
        let closed = false;
        #[cfg(feature = "nn")]
        let evaluated = AtomicBool::new(false);
        Self {
            parent,
            children,
            rule,
            atomic_visits,
            score,
            prior_bits,
            closed,
            #[cfg(feature = "nn")]
            evaluated,
        }
    }

    fn visits(&self) -> u32 {
        self.atomic_visits.load(Ordering::Relaxed)
    }

    fn prior(&self) -> f32 {
        f32::from_bits(self.prior_bits.load(Ordering::Relaxed))
    }

    #[cfg(feature = "nn")]
    fn set_prior(&self, prior: f32) {
        self.prior_bits.store(prior.to_bits(), Ordering::Relaxed);
    }

    fn is_leaf(&self) -> bool {
        !self.closed && Range::is_empty(self.children)
    }

    fn puct(&self, max_score: u32, sqrt_pv: f32) -> UCTValue {
        let score = (max_score - self.score) as f32;
        let max_score = max_score as f32;
        let exploitation = score / max_score;

        let visits = self.visits() as f32;
        let exploration = self.prior() * sqrt_pv / visits;

        UCTValue::new(exploitation + exploration)
    }
}

pub(crate) struct UCTree {
    nodes: Block<UCTNode>,
}

impl Default for UCTree {
    fn default() -> Self {
        let mut nodes = Block::default();
        nodes.push(UCTNode::new(Id::default(), Rule::Reflexivity, 0, 1.0));
        Self { nodes }
    }
}

impl UCTree {
    pub(crate) fn is_closed(&self) -> bool {
        self.nodes[Id::default()].closed
    }

    fn choose_child(&self, current: Id<UCTNode>) -> Option<Id<UCTNode>> {
        let node = &self.nodes[current];
        let sqrt_pv = (node.visits() as f32).sqrt();
        let eligible = node
            .children
            .into_iter()
            .filter(|child| !self.nodes[*child].closed);

        let max = eligible
            .clone()
            .map(|child| self.nodes[child].score)
            .max()?;
        eligible.max_by_key(|child| self.nodes[*child].puct(max, sqrt_pv))
    }

    pub(crate) fn select_for_expansion(
        &self,
        list: &mut Vec<Rule>,
    ) -> Option<Id<UCTNode>> {
        debug_assert!(!self.is_closed());
        let mut current = Id::default();
        while !self.nodes[current].is_leaf() {
            let next = self.choose_child(current)?;
            list.push(self.nodes[next].rule);
            self.nodes[current]
                .atomic_visits
                .fetch_add(1, Ordering::Relaxed);
            current = next;
        }
        Some(current)
    }

    #[cfg(feature = "nn")]
    pub(crate) fn select_for_evaluation(
        &self,
        list: &mut Vec<Rule>,
    ) -> Option<Id<UCTNode>> {
        if self.is_closed() {
            return None;
        }

        let mut current = Id::default();
        while self.nodes[current].evaluated.load(Ordering::Relaxed) {
            let next = self.choose_child(current)?;
            list.push(self.nodes[next].rule);
            current = next;
        }
        if self.nodes[current].is_leaf() {
            None
        } else {
            Some(current)
        }
    }

    pub(crate) fn expand(
        &mut self,
        parent: Id<UCTNode>,
        data: &[(Rule, u32)],
    ) {
        let prior = 1.0 / data.len() as f32;
        let start = self.nodes.len();
        for (rule, score) in data {
            self.nodes.push(UCTNode::new(parent, *rule, *score, prior));
        }
        let end = self.nodes.len();
        self.nodes[parent].children = Range::new(start, end);

        let mut current = parent;
        loop {
            let children = self.nodes[current].children;
            let closed =
                children.into_iter().all(|child| self.nodes[child].closed);
            if !closed {
                self.nodes[current].score = some(
                    children
                        .into_iter()
                        .filter(|child| !self.nodes[*child].closed)
                        .map(|child| self.nodes[child].score)
                        .min(),
                );
            }
            self.nodes[current].closed = closed;

            if current == Id::default() {
                break;
            }
            current = self.nodes[current].parent;
        }
    }

    #[cfg(feature = "nn")]
    pub(crate) fn evaluate(&self, parent: Id<UCTNode>, scores: &[f32]) {
        for (node, score) in
            self.nodes[parent].children.into_iter().zip(scores.iter())
        {
            self.nodes[node].set_prior(*score);
        }
        self.nodes[parent].evaluated.store(true, Ordering::Relaxed);
    }

    pub(crate) fn eligible_training_nodes(
        &self,
        options: &Options,
    ) -> Vec<Id<UCTNode>> {
        let mut nodes: Vec<_> = self
            .nodes
            .range()
            .into_iter()
            .filter(move |id| !self.nodes[*id].is_leaf())
            .filter(move |id| !self.nodes[*id].closed)
            .filter(move |id| {
                self.nodes[*id]
                    .children
                    .into_iter()
                    .any(|child| self.nodes[child].score == 0)
                    || self.nodes[*id].visits() as usize
                        >= options.min_training_visits
            })
            .collect();
        nodes.sort_unstable_by_key(|id| -(self.nodes[*id].visits() as i64));
        nodes.truncate(options.max_training_data);
        nodes
    }

    pub(crate) fn rules_for_node(&self, node: Id<UCTNode>) -> Vec<Rule> {
        let mut result = vec![];
        let mut current = node;
        while current != Id::default() {
            result.push(self.nodes[current].rule);
            current = self.nodes[current].parent;
        }
        result.reverse();
        result
    }

    #[cfg(feature = "nn")]
    pub(crate) fn child_inferences(
        &self,
        node: Id<UCTNode>,
        rules: &mut Vec<Rule>,
    ) {
        rules.extend(
            self.nodes[node]
                .children
                .into_iter()
                .map(|node| self.nodes[node].rule),
        );
    }

    pub(crate) fn child_rule_scores(
        &self,
        node: Id<UCTNode>,
    ) -> impl Iterator<Item = (Rule, u32)> + '_ {
        self.nodes[node]
            .children
            .into_iter()
            .filter(move |id| !self.nodes[*id].closed)
            .map(move |id| (self.nodes[id].rule, self.nodes[id].score))
    }
}
