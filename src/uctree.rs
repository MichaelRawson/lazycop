use crate::prelude::*;

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
    visits: u32,
    score: u32,
    prior: f32,
    closed: bool,
}

impl UCTNode {
    fn new(parent: Id<UCTNode>, rule: Rule, score: u32, prior: f32) -> Self {
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

    fn puct(&self, max_score: u32, sqrt_pv: f32) -> UCTValue {
        let score = (max_score - self.score) as f32;
        let max_score = max_score as f32;
        let exploitation = score / max_score;

        let visits = self.visits as f32;
        let exploration = self.prior * sqrt_pv / visits;

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

    pub(crate) fn take(&mut self, list: &mut Vec<Rule>) -> Id<UCTNode> {
        debug_assert!(!self.is_closed());
        let mut current = Id::default();
        while !self.nodes[current].is_leaf() {
            /*
            for node in self.nodes[current].children {
                print!(" {}", self.nodes[node].score);
                if self.nodes[node].closed {
                    print!("*");
                }
            }
            println!();
            */

            let node = &self.nodes[current];
            let sqrt_pv = (node.visits as f32).sqrt();
            let eligible = node
                .children
                .into_iter()
                .filter(|child| !self.nodes[*child].closed);
            let max_score = some(
                eligible.clone().map(|child| self.nodes[child].score).max(),
            );
            let next = some(eligible.max_by_key(|child| {
                self.nodes[*child].puct(max_score, sqrt_pv)
            }));
            list.push(self.nodes[next].rule);
            self.nodes[current].visits += 1;
            current = next;
        }
        current
    }

    pub(crate) fn give(&mut self, parent: Id<UCTNode>, data: &[(Rule, u32)]) {
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

    pub(crate) fn eligible_training_nodes(
        &self,
        threshold: u32,
    ) -> impl Iterator<Item = Id<UCTNode>> + '_ {
        self.nodes
            .range()
            .into_iter()
            .filter(move |id| !self.nodes[*id].closed)
            .filter(move |id| self.nodes[*id].visits > threshold)
    }

    pub(crate) fn rules_for_node(
        &self,
        node: Id<UCTNode>,
        rules: &mut Vec<Rule>,
    ) {
        rules.clear();
        let mut current = node;
        while current != Id::default() {
            rules.push(self.nodes[current].rule);
            current = self.nodes[current].parent;
        }
        rules.reverse();
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
