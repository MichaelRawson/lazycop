use crate::atom::Atom;
use crate::constraint::Constraints;
use crate::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct Literal {
    pub(crate) polarity: bool,
    atom: Atom,
}

impl Literal {
    pub(crate) fn new(polarity: bool, atom: Atom) -> Self {
        Self { polarity, atom }
    }

    pub(crate) fn disequation(left: Id<Term>, right: Id<Term>) -> Self {
        Self::new(false, Atom::Equality(left, right))
    }

    pub(crate) fn offset(&mut self, offset: Offset<Term>) {
        self.atom.offset(offset);
    }

    pub(crate) fn is_predicate(&self) -> bool {
        self.atom.is_predicate()
    }

    pub(crate) fn is_equality(&self) -> bool {
        self.atom.is_equality()
    }

    pub(crate) fn get_predicate(&self) -> Id<Term> {
        self.atom.get_predicate()
    }

    pub(crate) fn get_predicate_symbol(&self, terms: &Terms) -> Id<Symbol> {
        self.atom.get_predicate_symbol(terms)
    }

    pub(crate) fn get_predicate_arguments(
        &self,
        symbols: &Symbols,
        terms: &Terms,
    ) -> Range<Argument> {
        self.atom.get_predicate_arguments(symbols, terms)
    }

    pub(crate) fn get_equality(&self) -> (Id<Term>, Id<Term>) {
        self.atom.get_equality()
    }

    pub(crate) fn graph(
        &self,
        graph: &mut Graph,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
    ) -> Id<Node> {
        let atom = self.atom.graph(graph, symbols, terms, bindings);
        if !self.polarity {
            graph.negation(atom)
        } else {
            atom
        }
    }

    pub(crate) fn subterms<F: FnMut(Id<Term>)>(
        &self,
        symbols: &Symbols,
        terms: &Terms,
        f: &mut F,
    ) {
        self.atom.subterms(symbols, terms, f);
    }

    pub(crate) fn subst(
        &self,
        symbols: &Symbols,
        terms: &mut Terms,
        constraints: &mut Constraints,
        from: Id<Term>,
        to: Id<Term>,
    ) -> Self {
        let polarity = self.polarity;
        let atom = self.atom.subst(symbols, terms, constraints, from, to);
        Self { polarity, atom }
    }

    pub(crate) fn add_reflexivity_constraints(
        &self,
        constraints: &mut Constraints,
    ) {
        self.atom.add_reflexivity_constraints(constraints);
    }

    pub(crate) fn add_disequation_constraints(
        &self,
        constraints: &mut Constraints,
        terms: &Terms,
        other: &Self,
    ) {
        self.atom
            .add_disequation_constraints(constraints, terms, &other.atom)
    }
}

pub(crate) type Literals = Block<Literal>;
