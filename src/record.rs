use crate::prelude::*;

pub(crate) trait Record {
    fn copy(
        &mut self,
        _symbols: &Symbols,
        _terms: &Terms,
        _literals: &Block<Literal>,
        _clause: &Clause,
    ) {
    }

    fn predicate_reduction(
        &mut self,
        _symbols: &Symbols,
        _terms: &Terms,
        _literals: &Block<Literal>,
        _clause: &Clause,
        _left: Id<Term>,
        _right: Id<Term>,
    ) {
    }

    fn predicate_extension(
        &mut self,
        _symbols: &Symbols,
        _terms: &Terms,
        _literals: &Block<Literal>,
        _clause: &Clause,
        _new_clause: &Clause,
    ) {
    }

    fn reflexivity(
        &mut self,
        _symbols: &Symbols,
        _terms: &Terms,
        _literals: &Block<Literal>,
        _clause: &Clause,
        _left: Id<Term>,
        _right: Id<Term>,
    ) {
    }

    /*
    fn unification(
        &mut self,
        _literals: &Symbols,
        _terms: &Terms,
        _bindings: &IdMap<Variable, Option<Id<Term>>>,
    ) {
    }
    */
}

pub(crate) struct Silent;
impl Record for Silent {}
