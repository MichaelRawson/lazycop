use crate::prelude::*;

const SCORE_BASE: f32 = 0.95;
const EXPLORATION: f32 = 1.0;

#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct UCTValue(f32);

impl UCTValue {
    fn new(value: f32) -> Self {
        debug_assert!(value.is_normal());
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
    visits: u32,
    score: i32,
    prior: f32,
    closed: bool,
}

impl UCTNode {
    fn new(parent: Id<UCTNode>, rule: Rule, score: i32, prior: f32) -> Self {
        let children = Range::new(Id::default(), Id::default());
        let visits = 1;
        let closed = false;
        Self {
            parent,
            children,
            rule,
            visits,
            score,
            prior,
            closed,
        }
    }

    fn is_leaf(&self) -> bool {
        !self.closed && Range::is_empty(self.children)
    }

    fn uct(&self, lpv: f32) -> UCTValue {
        let visits = self.visits as f32;
        let exploitation = SCORE_BASE.powi(self.score);
        let exploration = EXPLORATION * self.prior * (lpv / visits).sqrt();
        let uct = exploitation + exploration;
        UCTValue::new(uct)
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

    pub(crate) fn take(&mut self, list: &mut Vec<Rule>) -> Id<UCTNode> {
        debug_assert!(!self.is_closed());
        let mut current = Id::default();
        while !self.nodes[current].is_leaf() {
            let node = &self.nodes[current];
            let lpv = (node.visits as f32).ln();
            let eligible = node
                .children
                .into_iter()
                .filter(|child| !self.nodes[*child].closed);
            let next =
                some(eligible.max_by_key(|child| self.nodes[*child].uct(lpv)));
            list.push(self.nodes[next].rule);
            self.nodes[current].visits += 1;
            current = next;
        }
        current
    }

    pub(crate) fn give(&mut self, parent: Id<UCTNode>, data: &[(Rule, i32)]) {
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
}
