use crate::options::Options;
use crate::prelude::*;
use crate::util::heuristic::Heuristic;
#[cfg(feature = "nn")]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU32, Ordering};

const EXPLORATION: f32 = 2.0;

pub(crate) struct Node {
    parent: Id<Node>,
    children: Range<Node>,
    rule: Rule,
    score: u32,
    closed: bool,
    atomic_visits: AtomicU32,
    atomic_prior: AtomicU32,
    #[cfg(feature = "nn")]
    atomic_evaluated: AtomicBool,
}

impl Node {
    fn new(parent: Id<Node>, rule: Rule, score: u32, prior: f32) -> Self {
        let children = Range::new(Id::default(), Id::default());
        let closed = false;
        let atomic_visits = AtomicU32::new(1);
        let atomic_prior = AtomicU32::new(f32::to_bits(prior));
        #[cfg(feature = "nn")]
        let atomic_evaluated = AtomicBool::new(false);
        Self {
            parent,
            children,
            rule,
            score,
            closed,
            atomic_visits,
            atomic_prior,
            #[cfg(feature = "nn")]
            atomic_evaluated,
        }
    }

    fn is_leaf(&self) -> bool {
        !self.closed && Range::is_empty(self.children)
    }

    fn visits(&self) -> u32 {
        self.atomic_visits.load(Ordering::Relaxed)
    }

    fn increment_visits(&self) {
        self.atomic_visits.fetch_add(1, Ordering::Relaxed);
    }

    fn puct(&self, min: u32, max: u32, sqrt_pv: f32) -> Heuristic {
        let numerator = (self.score - min) as f32;
        let denominator = (max - min) as f32;
        let exploitation = 1.0 - numerator / denominator;

        let visits = self.visits() as f32;
        let exploration = EXPLORATION * self.prior() * sqrt_pv / visits;

        Heuristic::new(exploitation + exploration)
    }

    fn prior(&self) -> f32 {
        f32::from_bits(self.atomic_prior.load(Ordering::Relaxed))
    }

    #[cfg(feature = "nn")]
    fn set_prior(&self, prior: f32) {
        self.atomic_prior.store(prior.to_bits(), Ordering::Relaxed);
    }

    #[cfg(feature = "nn")]
    fn is_evaluated(&self) -> bool {
        self.atomic_evaluated.load(Ordering::Relaxed)
    }

    #[cfg(feature = "nn")]
    fn set_evaluated(&self) {
        self.atomic_evaluated.store(true, Ordering::Relaxed);
    }
}

pub(crate) struct Tree {
    nodes: Block<Node>,
}

impl Default for Tree {
    fn default() -> Self {
        let mut nodes = Block::default();
        nodes.push(Node::new(Id::default(), Rule::Reflexivity, 0, 1.0));
        Self { nodes }
    }
}

impl Tree {
    pub(crate) fn is_closed(&self) -> bool {
        self.nodes[Id::default()].closed
    }

    pub(crate) fn select_for_expansion<E: Extend<Rule>>(
        &self,
        rules: &mut E,
    ) -> Option<Id<Node>> {
        debug_assert!(!self.is_closed());
        let mut current = Id::default();
        while !self.nodes[current].is_leaf() {
            let next = self.choose_child(current)?;
            rules.extend(std::iter::once(self.nodes[next].rule));

            self.nodes[current].increment_visits();
            current = next;
        }
        Some(current)
    }

    #[cfg(feature = "nn")]
    pub(crate) fn select_for_evaluation<E: Extend<Rule>>(
        &self,
        rules: &mut E,
    ) -> Option<Id<Node>> {
        if self.is_closed() {
            return None;
        }

        let mut current = Id::default();
        while self.nodes[current].is_evaluated() {
            let next = self.choose_child(current)?;
            rules.extend(std::iter::once(self.nodes[next].rule));
            current = next;
        }
        if self.nodes[current].is_leaf() {
            None
        } else {
            Some(current)
        }
    }

    pub(crate) fn expand(&mut self, parent: Id<Node>, data: &[(Rule, u32)]) {
        let start = self.nodes.len();
        let prior = 1.0 / std::cmp::max(1, data.len()) as f32;
        for (rule, score) in data {
            self.nodes.push(Node::new(parent, *rule, *score, prior));
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
    pub(crate) fn evaluate(&self, parent: Id<Node>, scores: &[f32]) {
        for (node, score) in
            self.nodes[parent].children.into_iter().zip(scores.iter())
        {
            self.nodes[node].set_prior(*score);
        }
        self.nodes[parent].set_evaluated();
    }

    pub(crate) fn backward_derivation(
        &self,
        node: Id<Node>,
    ) -> impl Iterator<Item = Rule> + '_ {
        let mut current = node;
        std::iter::from_fn(move || {
            if current == Id::default() {
                None
            } else {
                let rule = self.nodes[current].rule;
                current = self.nodes[current].parent;
                Some(rule)
            }
        })
    }

    pub(crate) fn child_rule_scores(
        &self,
        node: Id<Node>,
    ) -> impl Iterator<Item = (Rule, u32)> + '_ {
        self.nodes[node]
            .children
            .into_iter()
            .filter(move |id| !self.nodes[*id].closed)
            .map(move |id| (self.nodes[id].rule, self.nodes[id].score))
    }

    pub(crate) fn eligible_training_nodes(
        &self,
        options: &Options,
    ) -> Vec<Id<Node>> {
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

    fn choose_child(&self, current: Id<Node>) -> Option<Id<Node>> {
        let node = &self.nodes[current];
        let eligible = node
            .children
            .into_iter()
            .filter(|child| !self.nodes[*child].closed);

        let sqrt_pv = (node.visits() as f32).sqrt();
        let min = node.score;
        let max = eligible
            .clone()
            .map(|child| self.nodes[child].score)
            .max()?;
        let max = std::cmp::max(min + 1, max);
        eligible.max_by_key(|child| self.nodes[*child].puct(min, max, sqrt_pv))
    }
}
