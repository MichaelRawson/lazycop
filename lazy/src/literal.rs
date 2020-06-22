use crate::atom::Atom;
use crate::constraint::Constraints;
use crate::prelude::*;

#[derive(Clone, Copy)]
pub struct Literal {
    pub polarity: bool,
    atom: Atom,
}

impl Literal {
    pub fn new(polarity: bool, atom: Atom) -> Self {
        Self { polarity, atom }
    }

    pub fn disequation(left: Id<Term>, right: Id<Term>) -> Self {
        Self::new(false, Atom::Equality(left, right))
    }

    pub fn offset(&mut self, offset: Offset<Term>) {
        self.atom.offset(offset);
    }

    pub fn is_predicate(&self) -> bool {
        self.atom.is_predicate()
    }

    pub fn is_equality(&self) -> bool {
        self.atom.is_equality()
    }

    pub fn get_predicate(&self) -> Id<Term> {
        self.atom.get_predicate()
    }

    pub fn get_predicate_symbol(&self, terms: &Terms) -> Id<Symbol> {
        self.atom.get_predicate_symbol(terms)
    }

    pub fn get_predicate_arguments(
        &self,
        symbols: &Symbols,
        terms: &Terms,
    ) -> Range<Argument> {
        self.atom.get_predicate_arguments(symbols, terms)
    }

    pub fn get_equality(&self) -> (Id<Term>, Id<Term>) {
        self.atom.get_equality()
    }

    pub fn subterms<F: FnMut(Id<Term>)>(
        &self,
        symbols: &Symbols,
        terms: &Terms,
        f: &mut F,
    ) {
        self.atom.subterms(symbols, terms, f);
    }

    pub fn subst(
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

    pub fn add_reflexivity_constraints(&self, constraints: &mut Constraints) {
        self.atom.add_reflexivity_constraints(constraints);
    }

    pub fn add_disequation_constraints(
        &self,
        constraints: &mut Constraints,
        terms: &Terms,
        other: &Self,
    ) {
        self.atom
            .add_disequation_constraints(constraints, terms, &other.atom)
    }
}

pub type Literals = Block<Literal>;
