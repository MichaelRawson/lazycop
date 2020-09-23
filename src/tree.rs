use crate::options::Options;
use crate::prelude::*;

pub(crate) struct Node {
    parent: Id<Node>,
    children: Range<Node>,
    rule: Rule,
    log_prior: f32,
    score: f32,
    closed: bool,
    #[cfg(feature = "nn")]
    evaluated: bool,
}

impl Node {
    fn leaf(
        parent: Id<Node>,
        rule: Rule,
        log_siblings: f32,
        depth: u32,
        estimate: u32,
    ) -> Self {
        let children = Range::new(Id::default(), Id::default());
        let log_prior = log_siblings;
        let score = -(estimate as f32 + (depth as f32).ln());
        let closed = false;
        Self {
            parent,
            children,
            rule,
            log_prior,
            score,
            closed,
            #[cfg(feature = "nn")]
            evaluated: false,
        }
    }

    fn is_leaf(&self) -> bool {
        !self.closed && Range::is_empty(self.children)
    }
}

pub(crate) struct Tree {
    nodes: Block<Node>,
}

impl Default for Tree {
    fn default() -> Self {
        let mut nodes = Block::default();
        nodes.push(Node::leaf(Id::default(), Rule::Reflexivity, 0.0, 1, 0));
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
            current = self.choose_child(current);
            rules.extend(std::iter::once(self.nodes[current].rule));
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

    pub(crate) fn expand(
        &mut self,
        leaf: Id<Node>,
        depth: u32,
        data: &[(Rule, u32)],
    ) {
        let log_siblings = -(data.len() as f32).ln();
        let start = self.nodes.len();
        for (rule, estimate) in data {
            self.nodes.push(Node::leaf(
                leaf,
                *rule,
                log_siblings,
                depth,
                *estimate,
            ));
        }
        let end = self.nodes.len();
        self.nodes[leaf].children = Range::new(start, end);
        self.propagate_expansion(leaf);
    }

    #[cfg(feature = "nn")]
    pub(crate) fn evaluate(&mut self, node: Id<Node>, log_priors: &[f32]) {
        for (child, log_prior) in
            self.nodes[node].children.into_iter().zip(log_priors.iter())
        {
            self.nodes[child].log_prior = *log_prior
        }
        self.nodes[node].evaluated = true;
        self.propagate_evaluation(node);
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

    pub(crate) fn child_rules(
        &self,
        node: Id<Node>,
    ) -> impl Iterator<Item = Rule> + '_ {
        self.nodes[node]
            .children
            .into_iter()
            .map(move |child| self.nodes[child].rule)
    }

    pub(crate) fn eligible_training_nodes(
        &self,
        _options: &Options,
    ) -> Vec<Id<Node>> {
        /*
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
        */
        todo!()
    }

    fn propagate_expansion(&mut self, leaf: Id<Node>) {
        let mut current = leaf;
        while self.nodes[current]
            .children
            .into_iter()
            .all(|child| self.nodes[child].closed)
        {
            self.nodes[current].closed = true;
            if current == Id::default() {
                return;
            }
            current = self.nodes[current].parent;
        }

        loop {
            self.update_score(current);
            if current == Id::default() {
                return;
            }
            current = self.nodes[current].parent;
        }
    }

    #[cfg(feature = "nn")]
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
        some(self.open_children(current).max_by(|x, y| {
            some(self.nodes[*x].score.partial_cmp(&self.nodes[*y].score))
        }))
    }

    fn open_children(
        &self,
        parent: Id<Node>,
    ) -> impl Iterator<Item = Id<Node>> + '_ {
        self.nodes[parent]
            .children
            .into_iter()
            .filter(move |child| !self.nodes[*child].closed)
    }

    fn update_score(&mut self, node: Id<Node>) {
        self.nodes[node].score = self
            .open_children(node)
            .map(|child| &self.nodes[child])
            .map(|child| child.score + child.log_prior)
            .max_by(|x, y| some(x.partial_cmp(y)))
            .unwrap_or_default();
    }
}
