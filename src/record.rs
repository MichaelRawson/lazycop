use crate::clause::Clause;
use crate::prelude::*;

pub(crate) trait Record {
    fn copy(
        &mut self,
        _symbols: &Block<Symbol>,
        _terms: &Terms,
        _literals: &Block<Literal>,
        _clause: &Clause,
    ) {
    }

    fn start(&mut self) {}

    fn predicate_reduction(
        &mut self,
        _symbols: &Block<Symbol>,
        _terms: &Terms,
        _literals: &Block<Literal>,
        _clause: &Clause,
        _left: Id<Term>,
        _right: Id<Term>,
    ) {
    }

    fn predicate_extension(
        &mut self,
        _symbols: &Block<Symbol>,
        _terms: &Terms,
        _literals: &Block<Literal>,
        _clause: &Clause,
        _new_clause: &Clause,
    ) {
    }

    fn reflexivity(
        &mut self,
        _symbols: &Block<Symbol>,
        _terms: &Terms,
        _literals: &Block<Literal>,
        _clause: &Clause,
        _left: Id<Term>,
        _right: Id<Term>,
    ) {
    }

    fn unification<I: Iterator<Item = (Id<Variable>, Id<Term>)>>(
        &mut self,
        _symbols: &Block<Symbol>,
        _terms: &Terms,
        _bindings: I,
    ) {
    }
}

pub(crate) struct Silent;
impl Record for Silent {}
