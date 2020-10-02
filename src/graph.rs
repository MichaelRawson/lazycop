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
    Lemma,
    Clause,
    Start,
    Reflexivity,
    Reduction,
    StrictExtension,
    LazyExtension,
    Demodulation,
    StrictParamodulation,
    LazyParamodulation,
}

#[derive(Default)]
pub(crate) struct Graph {
    pub(crate) nodes: Block<u32>,
    pub(crate) sources: Vec<u32>,
    pub(crate) targets: Vec<u32>,
    pub(crate) rules: Vec<u32>,
    symbols: LUT<Symbol, Option<Id<Node>>>,
    terms: LUT<Term, Option<Id<Node>>>,
    literals: LUT<Literal, Option<Id<Node>>>,
}

impl Graph {
    pub(crate) fn node_labels(&self) -> &[u32] {
        self.nodes.as_ref()
    }

    pub(crate) fn clear(&mut self) {
        self.nodes.clear();
        self.sources.clear();
        self.targets.clear();
        self.rules.clear();
        self.symbols.resize(Id::default());
        self.terms.resize(Id::default());
        self.literals.resize(Id::default());
    }

    pub(crate) fn signature(&mut self, symbols: &Symbols) {
        self.symbols.resize(symbols.len());
    }

    pub(crate) fn resize_for(&mut self, terms: &Terms, literals: &Literals) {
        self.terms.resize(terms.len());
        self.literals.resize(literals.len());
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

    pub(crate) fn store_term(&mut self, term: Id<Term>, node: Id<Node>) {
        self.terms[term] = Some(node);
    }

    pub(crate) fn get_term(&self, term: Id<Term>) -> Id<Node> {
        unwrap(self.terms[term])
    }

    pub(crate) fn get_possible_term(
        &self,
        term: Id<Term>,
    ) -> Option<Id<Node>> {
        self.terms[term]
    }

    pub(crate) fn store_literal(
        &mut self,
        literal: Id<Literal>,
        node: Id<Node>,
    ) {
        self.literals[literal] = Some(node);
    }

    pub(crate) fn get_literal(&self, literal: Id<Literal>) -> Id<Node> {
        unwrap(self.literals[literal])
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

    pub(crate) fn lemma(&mut self, literal: Id<Node>) -> Id<Node> {
        let node = self.node(Node::Lemma);
        self.connect(node, literal);
        node
    }

    pub(crate) fn clause(&mut self) -> Id<Node> {
        self.node(Node::Clause)
    }

    pub(crate) fn start(&mut self, clause: Id<Node>) -> Id<Node> {
        let node = self.rule(Node::Start);
        self.connect(node, clause);
        node
    }

    pub(crate) fn reflexivity(&mut self, current: Id<Node>) -> Id<Node> {
        let node = self.rule(Node::Reflexivity);
        self.connect(node, current);
        node
    }

    pub(crate) fn reduction(
        &mut self,
        mate: Id<Node>,
        current: Id<Node>,
    ) -> Id<Node> {
        let node = self.rule(Node::Reduction);
        self.connect(mate, node);
        self.connect(node, current);
        node
    }

    pub(crate) fn extension(
        &mut self,
        strict: bool,
        current: Id<Node>,
        mate: Id<Node>,
    ) -> Id<Node> {
        let node = if strict {
            Node::StrictExtension
        } else {
            Node::LazyExtension
        };
        let node = self.rule(node);
        self.connect(current, node);
        self.connect(node, mate);
        node
    }

    pub(crate) fn demodulation(
        &mut self,
        from: Id<Node>,
        target: Id<Node>,
    ) -> Id<Node> {
        let node = self.rule(Node::Demodulation);
        self.connect(from, node);
        self.connect(node, target);
        node
    }

    pub(crate) fn paramodulation(
        &mut self,
        strict: bool,
        from: Id<Node>,
        target: Id<Node>,
    ) -> Id<Node> {
        let node = if strict {
            Node::StrictParamodulation
        } else {
            Node::LazyParamodulation
        };
        let node = self.rule(node);
        self.connect(from, node);
        self.connect(node, target);
        node
    }

    fn rule(&mut self, node: Node) -> Id<Node> {
        let node = self.node(node);
        self.rules.push(node.index());
        node
    }

    fn node(&mut self, node: Node) -> Id<Node> {
        self.nodes.push(node as u32).transmute()
    }
}
