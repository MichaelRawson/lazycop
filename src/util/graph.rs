use crate::prelude::*;

#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub(crate) enum Node {
    Symbol,
    Variable,
    Argument,
    Application,
    Equality,
    Negation,
    Clause,
}

#[derive(Default)]
pub(crate) struct Graph {
    pub(crate) subgraphs: u32,
    pub(crate) nodes: Block<Node>,
    pub(crate) from: Vec<u32>,
    pub(crate) to: Vec<u32>,
    pub(crate) node_batch: Vec<u32>,
    pub(crate) edge_batch: Vec<u32>,
    symbols: LUT<Symbol, Option<Id<Node>>>,
    terms: LUT<Term, Option<Id<Node>>>,
}

impl Graph {
    pub(crate) fn clear(&mut self) {
        self.subgraphs = 0;
        self.nodes.clear();
        self.from.clear();
        self.to.clear();
        self.node_batch.clear();
        self.edge_batch.clear();
        self.symbols.resize(Id::default());
        self.terms.resize(Id::default());
    }

    pub(crate) fn finish_subgraph(&mut self) {
        self.subgraphs += 1;
        self.node_batch.push(self.nodes.len().index());
        self.edge_batch.push(self.from.len() as u32);
        self.symbols.resize(Id::default());
        self.terms.resize(Id::default());
    }

    pub(crate) fn initialise(&mut self, symbols: &Symbols, terms: &Terms) {
        self.symbols.resize(symbols.len());
        self.terms.resize(terms.len());
    }

    pub(crate) fn connect(&mut self, from: Id<Node>, to: Id<Node>) {
        self.from.push(from.index());
        self.to.push(to.index());
    }

    pub(crate) fn symbol(&mut self, symbol: Id<Symbol>) -> Id<Node> {
        if let Some(node) = self.symbols[symbol] {
            node
        } else {
            let node = self.node(Node::Symbol);
            self.symbols[symbol] = Some(node);
            node
        }
    }

    pub(crate) fn variable(&mut self) -> Id<Node> {
        self.node(Node::Variable)
    }

    pub(crate) fn argument(
        &mut self,
        application: Id<Node>,
        previous: Id<Node>,
        term: Id<Node>,
    ) -> Id<Node> {
        let argument = self.node(Node::Argument);
        self.connect(argument, term);
        self.connect(application, argument);
        self.connect(previous, argument);
        argument
    }

    pub(crate) fn application(&mut self, symbol: Id<Node>) -> Id<Node> {
        let application = self.node(Node::Application);
        self.connect(application, symbol);
        application
    }

    pub(crate) fn get_cached_term(&self, term: Id<Term>) -> Option<Id<Node>> {
        self.terms[term]
    }

    pub(crate) fn cache_term(&mut self, term: Id<Term>, node: Id<Node>) {
        self.terms[term] = Some(node);
    }

    pub(crate) fn equality(
        &mut self,
        left: Id<Node>,
        right: Id<Node>,
    ) -> Id<Node> {
        let equality = self.node(Node::Equality);
        self.connect(equality, left);
        self.connect(equality, right);
        equality
    }

    pub(crate) fn negation(&mut self, atom: Id<Node>) -> Id<Node> {
        let negation = self.node(Node::Negation);
        self.connect(negation, atom);
        negation
    }

    pub(crate) fn clause(&mut self) -> Id<Node> {
        self.node(Node::Clause)
    }

    fn node(&mut self, node: Node) -> Id<Node> {
        self.nodes.push(node)
    }
}
