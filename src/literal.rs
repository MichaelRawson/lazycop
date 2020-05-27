use crate::atom::Atom;
use crate::prelude::*;
use crate::solver::Solver;

#[derive(Clone, Copy)]
pub(crate) struct Literal {
    pub(crate) polarity: bool,
    atom: Atom,
}

impl Literal {
    pub(crate) fn new(polarity: bool, atom: Atom) -> Self {
        Self { polarity, atom }
    }

    pub(crate) fn offset(&mut self, offset: Offset<Term>) {
        self.atom.offset(offset);
    }

    pub(crate) fn invert(&mut self) {
        self.polarity = !self.polarity;
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

    pub(crate) fn get_equality(&self) -> (Id<Term>, Id<Term>) {
        self.atom.get_equality()
    }

    pub(crate) fn add_unit_constraints(&self, solver: &mut Solver) {
        if self.polarity {
            self.atom.add_positive_constraints(solver);
        }
    }

    pub(crate) fn add_disequation_constraints(
        &self,
        solver: &mut Solver,
        terms: &Terms,
        other: &Self,
    ) {
        self.atom
            .add_disequation_constraints(solver, terms, &other.atom)
    }
}
