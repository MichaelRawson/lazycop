use crate::prelude::*;

#[cfg(feature = "nn")]
const LAMBDA: f32 = 1.0;

pub(crate) struct Node {
    parent: Id<Node>,
    children: Range<Node>,
    rule: Rule,
    #[cfg(not(feature = "nn"))]
    score: i32,
    #[cfg(feature = "nn")]
    score: f32,
    #[cfg(feature = "nn")]
    log_prior: f32,
    #[cfg(feature = "nn")]
    evaluated: bool,
}

impl Node {
    fn leaf(parent: Id<Node>, rule: Rule, estimate: u32) -> Self {
        let children = Range::new(Id::default(), Id::default());

        #[cfg(feature = "nn")]
        let score = -LAMBDA * estimate as f32;
        #[cfg(not(feature = "nn"))]
        let score = -(estimate as i32);

        Self {
            parent,
            children,
            rule,
            #[cfg(feature = "nn")]
            log_prior: 0.0,
            score,
            #[cfg(feature = "nn")]
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

    pub(crate) fn select_for_expansion(
        &self,
        rules: &mut Vec<Rule>,
    ) -> Id<Node> {
        debug_assert!(!self.is_closed());
        let mut current = Id::default();
        while !self.nodes[current].is_leaf() {
            current = self.choose_child(current);
            rules.push(self.nodes[current].rule);
        }
        current
    }

    pub(crate) fn expand<I: Iterator<Item = (Rule, u32)>>(
        &mut self,
        leaf: Id<Node>,
        children: I,
    ) {
        let start = self.nodes.end();
        for (rule, estimate) in children {
            self.nodes.push(Node::leaf(leaf, rule, estimate));
        }
        let end = self.nodes.end();
        self.nodes[leaf].children = Range::new(start, end);
        self.propagate_expansion(leaf);
    }

    pub(crate) fn derivation(&self, mut index: Id<Node>) -> Vec<Rule> {
        let mut rules = vec![];
        while index != Id::default() {
            rules.push(self.nodes[index].rule);
            index = self.nodes[index].parent;
        }
        rules.reverse();
        rules
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

    #[cfg(feature = "nn")]
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
        loop {
            self.update_score(current);
            if current == Id::default() {
                break;
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

    #[cfg(feature = "nn")]
    fn update_score(&mut self, node: Id<Node>) {
        self.nodes[node].score = self
            .open_children(node)
            .map(|child| &self.nodes[child])
            .map(|child| child.score + child.log_prior)
            .max_by(|x, y| unwrap(x.partial_cmp(y)))
            .unwrap_or_default();
    }

    #[cfg(not(feature = "nn"))]
    fn update_score(&mut self, node: Id<Node>) {
        self.nodes[node].score = self
            .open_children(node)
            .map(|child| self.nodes[child].score)
            .max()
            .unwrap_or_default();
    }
}
