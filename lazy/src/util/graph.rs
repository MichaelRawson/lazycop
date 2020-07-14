use crate::prelude::*;

#[derive(Clone, Copy)]
#[repr(u32)]
pub enum Node {
    Symbol,
    Variable,
    Argument,
    Application,
    Equality,
    Negation,
    Clause,
}

#[derive(Default)]
pub struct Graph {
    pub subgraphs: u32,
    pub nodes: Block<Node>,
    pub from: Vec<u32>,
    pub to: Vec<u32>,
    pub batch: Vec<u32>,
    symbols: LUT<Symbol, Option<Id<Node>>>,
    terms: LUT<Term, Option<Id<Node>>>,
}

impl Graph {
    pub fn clear(&mut self) {
        self.subgraphs = 0;
        self.nodes.clear();
        self.from.clear();
        self.to.clear();
        self.batch.clear();
        self.symbols.resize(Id::default());
        self.terms.resize(Id::default());
    }

    pub fn finish_subgraph(&mut self) {
        self.subgraphs += 1;
        self.symbols.resize(Id::default());
        self.terms.resize(Id::default());
    }

    pub fn initialise(&mut self, symbols: &Symbols, terms: &Terms) {
        self.symbols.resize(symbols.len());
        self.terms.resize(terms.len());
    }

    pub fn connect(&mut self, from: Id<Node>, to: Id<Node>) {
        self.from.push(from.as_u32() - 1);
        self.to.push(to.as_u32() - 1);
    }

    pub fn symbol(&mut self, symbol: Id<Symbol>) -> Id<Node> {
        if let Some(node) = self.symbols[symbol] {
            node
        } else {
            let node = self.node(Node::Symbol);
            self.symbols[symbol] = Some(node);
            node
        }
    }

    pub fn variable(&mut self) -> Id<Node> {
        self.node(Node::Variable)
    }

    pub fn argument(
        &mut self,
        application: Id<Node>,
        previous: Id<Node>,
        term: Id<Node>,
    ) -> Id<Node> {
        let argument = self.node(Node::Application);
        self.connect(argument, term);
        self.connect(application, argument);
        self.connect(previous, argument);
        argument
    }

    pub fn application(&mut self, symbol: Id<Node>) -> Id<Node> {
        let application = self.node(Node::Application);
        self.connect(application, symbol);
        application
    }

    pub fn get_term(&self, term: Id<Term>) -> Option<Id<Node>> {
        self.terms[term]
    }

    pub fn set_term(&mut self, term: Id<Term>, node: Id<Node>) {
        self.terms[term] = Some(node);
    }

    pub fn equality(&mut self, left: Id<Node>, right: Id<Node>) -> Id<Node> {
        let equality = self.node(Node::Equality);
        self.connect(equality, left);
        self.connect(equality, right);
        equality
    }

    pub fn negation(&mut self, atom: Id<Node>) -> Id<Node> {
        let negation = self.node(Node::Negation);
        self.connect(negation, atom);
        negation
    }

    pub fn clause(&mut self) -> Id<Node> {
        self.node(Node::Clause)
    }

    fn node(&mut self, node: Node) -> Id<Node> {
        self.batch.push(self.subgraphs);
        self.nodes.push(node)
    }
}
