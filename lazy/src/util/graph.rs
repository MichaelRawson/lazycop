use crate::prelude::*;

#[derive(Clone, Copy)]
pub enum Node {
    Symbol,
    Variable,
    Argument,
    Application,
    Predicate,
    Equality,
    Negation,
}

#[derive(Default)]
pub struct Graph {
    pub nodes: Block<Node>,
    pub from: Vec<Id<Node>>,
    pub to: Vec<Id<Node>>,
    symbols: LUT<Symbol, Option<Id<Node>>>,
    terms: LUT<Term, Option<Id<Node>>>,
}

impl Graph {
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.from.clear();
        self.to.clear();
        self.symbols.resize(Id::default());
        self.terms.resize(Id::default());
    }

    pub fn initialise(&mut self, symbols: &Symbols, terms: &Terms) {
        self.symbols.resize(symbols.len());
        self.terms.resize(terms.len());
    }

    pub fn connect(&mut self, from: Id<Node>, to: Id<Node>) {
        self.from.push(from);
        self.to.push(to);
    }

    pub fn symbol(&mut self, symbol: Id<Symbol>) -> Id<Node> {
        if let Some(node) = self.symbols[symbol] {
            node
        } else {
            let node = self.nodes.push(Node::Symbol);
            self.symbols[symbol] = Some(node);
            node
        }
    }

    pub fn variable(&mut self) -> Id<Node> {
        self.nodes.push(Node::Variable)
    }

    pub fn argument(
        &mut self,
        application: Id<Node>,
        previous: Id<Node>,
        term: Id<Node>,
    ) -> Id<Node> {
        let argument = self.nodes.push(Node::Application);
        self.connect(argument, term);
        self.connect(application, argument);
        self.connect(previous, argument);
        argument
    }

    pub fn application(&mut self, symbol: Id<Node>) -> Id<Node> {
        let application = self.nodes.push(Node::Application);
        self.connect(application, symbol);
        application
    }

    pub fn get_term(&self, term: Id<Term>) -> Option<Id<Node>> {
        self.terms[term]
    }

    pub fn set_term(&mut self, term: Id<Term>, node: Id<Node>) {
        self.terms[term] = Some(node);
    }

    pub fn predicate(&mut self, term: Id<Node>) -> Id<Node> {
        let predicate = self.nodes.push(Node::Predicate);
        self.connect(predicate, term);
        predicate
    }

    pub fn equality(&mut self, left: Id<Node>, right: Id<Node>) -> Id<Node> {
        let equality = self.nodes.push(Node::Equality);
        self.connect(equality, left);
        self.connect(equality, right);
        equality
    }

    pub fn negation(&mut self, atom: Id<Node>) -> Id<Node> {
        let negation = self.nodes.push(Node::Negation);
        self.connect(negation, atom);
        negation
    }
}