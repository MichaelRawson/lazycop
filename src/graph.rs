use crate::prelude::*;

#[derive(Clone, Copy)]
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
    DistinctObjects,
    Reduction,
    StrictExtension,
    LazyExtension,
    Demodulation,
    StrictParamodulation,
    LazyParamodulation,
}

#[derive(Clone, Copy)]
pub(crate) struct ProblemClauseRecord {
    pub(crate) node: Id<Node>,
    pub(crate) term_offset: Offset<Term>,
    pub(crate) literal_offset: Offset<Literal>,
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
    problem_clauses: LUT<ProblemClause, Option<ProblemClauseRecord>>,
}

impl Graph {
    pub(crate) fn new(problem: &Problem) -> Self {
        let mut new = Self::default();
        new.symbols.resize(problem.symbols.len());
        new.problem_clauses.resize(problem.clauses.len());
        new
    }

    pub(crate) fn node_labels(&self) -> &[u32] {
        self.nodes.as_ref()
    }

    pub(crate) fn clear(&mut self) {
        for id in self.symbols.range() {
            self.symbols[id] = None;
        }
        for id in self.problem_clauses.range() {
            self.problem_clauses[id] = None;
        }
        self.nodes.clear();
        self.sources.clear();
        self.targets.clear();
        self.rules.clear();
        self.terms.resize(Length::default());
        self.literals.resize(Length::default());
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
                Name::Regular(_) | Name::Quoted(_) | Name::Distinct(_) => {
                    Node::Symbol
                }
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

    #[must_use]
    pub(crate) fn problem_clause(
        &mut self,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        bindings: &mut Bindings,
        problem_clause: Id<ProblemClause>,
    ) -> ProblemClauseRecord {
        if let Some(record) = self.problem_clauses[problem_clause] {
            return record;
        }

        let term_offset = terms.offset();
        let literal_offset = literals.offset();
        let new =
            Clause::new_from_axiom(problem, terms, literals, problem_clause);
        bindings.resize(terms.len());
        self.resize_for(terms, literals);
        let node =
            new.graph(self, &problem.symbols, terms, literals, bindings);

        let record = ProblemClauseRecord {
            node,
            term_offset,
            literal_offset,
        };
        self.problem_clauses[problem_clause] = Some(record);
        record
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

    pub(crate) fn distinct_objects(&mut self, current: Id<Node>) -> Id<Node> {
        let node = self.rule(Node::DistinctObjects);
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
