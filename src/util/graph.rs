use crate::prelude::*;

#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub(crate) enum Node {
    Symbol,
    Skolem,
    Definition,
    Variable,
    Argument,
    Application,
    Predicate,
    Equality,
    Negation,
    Root,
}

#[derive(Default)]
pub(crate) struct Graph {
    pub(crate) num_graphs: u32,
    pub(crate) nodes: Block<u32>,
    pub(crate) sources: Vec<u32>,
    pub(crate) targets: Vec<u32>,
    pub(crate) batch: Vec<u32>,
    symbols: LUT<Symbol, Option<Id<Node>>>,
    terms: LUT<Term, Option<Id<Node>>>,
}

impl Graph {
    pub(crate) fn node_labels(&self) -> &[u32] {
        self.nodes.as_ref()
    }

    pub(crate) fn clear(&mut self) {
        self.num_graphs = 0;
        self.nodes.clear();
        self.sources.clear();
        self.targets.clear();
        self.batch.clear();
        self.symbols.resize(Id::default());
        self.terms.resize(Id::default());
    }

    pub(crate) fn finish_subgraph(&mut self) {
        self.num_graphs += 1;
        self.symbols.resize(Id::default());
        self.terms.resize(Id::default());
    }

    pub(crate) fn initialise(&mut self, symbols: &Symbols, terms: &Terms) {
        self.symbols.resize(symbols.len());
        self.terms.resize(terms.len());
    }

    pub(crate) fn connect(&mut self, sources: Id<Node>, targets: Id<Node>) {
        self.sources.push(sources.index());
        self.targets.push(targets.index());
    }

    pub(crate) fn symbol(
        &mut self,
        symbols: &Symbols,
        symbol: Id<Symbol>,
    ) -> Id<Node> {
        if let Some(node) = self.symbols[symbol] {
            node
        } else {
            let node = match symbols[symbol].name {
                Name::Regular(_) | Name::Quoted(_) => Node::Symbol,
                Name::Skolem(_) => Node::Skolem,
                Name::Definition(_) => Node::Definition,
            };
            let node = self.node(node);
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

    pub(crate) fn predicate(&mut self, term: Id<Node>) -> Id<Node> {
        let predicate = self.node(Node::Predicate);
        self.connect(predicate, term);
        predicate
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

    pub(crate) fn negation(&mut self, target: Id<Node>) -> Id<Node> {
        let negation = self.node(Node::Negation);
        self.connect(negation, target);
        negation
    }

    pub(crate) fn root(&mut self) -> Id<Node> {
        self.node(Node::Root)
    }

    fn node(&mut self, node: Node) -> Id<Node> {
        self.batch.push(self.num_graphs);
        self.nodes.push(node as u32).transmute()
    }
}
