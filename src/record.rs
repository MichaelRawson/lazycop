use crate::clause::Clause;
use crate::prelude::*;

pub(crate) trait Record {
    fn axiom(
        &mut self,
        _symbols: &Symbols,
        _terms: &Terms,
        _literals: &Literals,
        _axiom: &Clause,
    ) {
    }

    fn inference(
        &mut self,
        _symbols: &Symbols,
        _terms: &Terms,
        _literals: &Literals,
        _inference: &'static str,
        _equations: &[(Id<Term>, Id<Term>)],
        _deductions: &[&Clause],
    ) {
    }

    fn unification<I: Iterator<Item = (Id<Variable>, Id<Term>)>>(
        &mut self,
        _symbols: &Symbols,
        _terms: &Terms,
        _bindings: I,
    ) {
    }
}

pub(crate) struct Silent;
impl Record for Silent {}
