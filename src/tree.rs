use crate::prelude::*;

#[cfg(feature = "cudann")]
const LAMBDA: f32 = 1.0;
#[cfg(feature = "cudann")]
const INVALID_SCORE: f32 = f32::NEG_INFINITY;
#[cfg(not(feature = "cudann"))]
const INVALID_SCORE: i32 = i32::MIN;

pub(crate) struct Node {
    parent: Id<Node>,
    children: Range<Node>,
    rule: Rule,
    #[cfg(not(feature = "cudann"))]
    score: i32,
    #[cfg(feature = "cudann")]
    score: f32,
    #[cfg(feature = "cudann")]
    log_prior: f32,
    #[cfg(feature = "cudann")]
    evaluated: bool,
}

impl Node {
    fn leaf(parent: Id<Node>, rule: Rule, estimate: u32) -> Self {
        let children = Range::new(Id::default(), Id::default());

        #[cfg(feature = "cudann")]
        let score = -LAMBDA * estimate as f32;
        #[cfg(not(feature = "cudann"))]
        let score = -(estimate as i32);

        Self {
            parent,
            children,
            rule,
            #[cfg(feature = "cudann")]
            log_prior: 0.0,
            score,
            #[cfg(feature = "cudann")]
            evaluated: false,
        }
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    fn is_closed(&self) -> bool {
        self.children.is_invalid()
    }
}

pub(crate) struct Tree {
    nodes: Block<Node>,
}

impl Default for Tree {
    fn default() -> Self {
        let mut nodes = Block::default();
        nodes.push(Node::leaf(Id::default(), Rule::Reflexivity, 0));
        Self { nodes }
    }
}

impl Tree {
    pub(crate) fn is_closed(&self) -> bool {
        self.nodes[Id::default()].is_closed()
    }

    pub(crate) fn select_for_expansion<E: Extend<Rule>>(
        &mut self,
        rules: &mut E,
    ) -> Option<Id<Node>> {
        debug_assert!(!self.is_closed());
        let mut current = Id::default();
        while !self.nodes[current].is_leaf() {
            current = self.choose_child(current);
            rules.extend(std::iter::once(self.nodes[current].rule));
        }
        if self.nodes[current].score == INVALID_SCORE {
            None
        } else {
            self.nodes[current].score = INVALID_SCORE;
            self.propagate_score(self.nodes[current].parent);
            Some(current)
        }
    }

    pub(crate) fn expand(&mut self, leaf: Id<Node>, data: &[(Rule, u32)]) {
        let start = self.nodes.end();
        for (rule, estimate) in data {
            self.nodes.push(Node::leaf(leaf, *rule, *estimate));
        }
        let end = self.nodes.end();
        self.nodes[leaf].children = Range::new(start, end);
        self.propagate_expansion(leaf);
    }

    #[cfg(feature = "smt")]
    pub(crate) fn select_for_smt<E: Extend<Rule>>(
        &self,
        rules: &mut E,
        start: Id<Node>,
    ) -> Option<Id<Node>> {
        let node = Range::new(start, self.nodes.end())
            .into_iter()
            .find(|id| !self.nodes[*id].is_closed())?;
        let mut ancestor = node;
        while ancestor != Id::default() {
            rules.extend(std::iter::once(self.nodes[ancestor].rule));
            ancestor = self.nodes[ancestor].parent;
        }
        Some(node)
    }

    #[cfg(feature = "cudann")]
    pub(crate) fn select_for_evaluation<E: Extend<Rule>>(
        &self,
        rules: &mut E,
    ) -> Option<Id<Node>> {
        if self.is_closed() {
            return None;
        }

        let mut current = Id::default();
        while self.nodes[current].evaluated {
            current = self.choose_child(current);
            rules.extend(std::iter::once(self.nodes[current].rule));
        }
        if self.nodes[current].is_leaf() {
            None
        } else {
            Some(current)
        }
    }

    #[cfg(feature = "cudann")]
    pub(crate) fn evaluate(&mut self, node: Id<Node>, log_priors: &[f32]) {
        for (child, log_prior) in
            self.nodes[node].children.into_iter().zip(log_priors.iter())
        {
            self.nodes[child].log_prior = *log_prior
        }
        self.nodes[node].evaluated = true;
        self.propagate_evaluation(node);
    }

    #[cfg(feature = "cudann")]
    pub(crate) fn child_rules(
        &self,
        node: Id<Node>,
    ) -> impl Iterator<Item = Rule> + '_ {
        self.nodes[node]
            .children
            .into_iter()
            .map(move |child| self.nodes[child].rule)
    }

    fn propagate_expansion(&mut self, leaf: Id<Node>) {
        let mut current = leaf;
        while self.nodes[current]
            .children
            .into_iter()
            .all(|child| self.nodes[child].is_closed())
        {
            self.nodes[current].children = Range::new_invalid();
            if current == Id::default() {
                return;
            }
            current = self.nodes[current].parent;
        }
        self.propagate_score(current);
    }

    fn propagate_score(&mut self, mut current: Id<Node>) {
        while current != Id::default() {
            self.update_score(current);
            current = self.nodes[current].parent;
        }
    }

    #[cfg(feature = "cudann")]
    fn propagate_evaluation(&mut self, evaluated: Id<Node>) {
        let mut current = evaluated;
        loop {
            self.update_score(current);
            if current == Id::default() {
                return;
            }
            current = self.nodes[current].parent;
        }
    }

    fn choose_child(&self, current: Id<Node>) -> Id<Node> {
        unwrap(self.open_children(current).max_by(|x, y| {
            unwrap(self.nodes[*x].score.partial_cmp(&self.nodes[*y].score))
        }))
    }

    fn open_children(
        &self,
        parent: Id<Node>,
    ) -> impl Iterator<Item = Id<Node>> + '_ {
        self.nodes[parent]
            .children
            .into_iter()
            .filter(move |child| !self.nodes[*child].is_closed())
    }

    #[cfg(feature = "cudann")]
    fn update_score(&mut self, node: Id<Node>) {
        self.nodes[node].score = self
            .open_children(node)
            .map(|child| &self.nodes[child])
            .map(|child| child.score + child.log_prior)
            .max_by(|x, y| unwrap(x.partial_cmp(y)))
            .unwrap_or_default();
    }

    #[cfg(not(feature = "cudann"))]
    fn update_score(&mut self, node: Id<Node>) {
        self.nodes[node].score = self
            .open_children(node)
            .map(|child| self.nodes[child].score)
            .max()
            .unwrap_or_default();
    }
}
