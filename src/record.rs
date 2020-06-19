use crate::clause::Clause;
use crate::prelude::*;
use std::fmt::Display;

pub(crate) trait Record {
    fn axiom(
        &mut self,
        _symbols: &Symbols,
        _terms: &Terms,
        _literals: &Literals,
        _axiom: Clause,
    ) {
    }

    #[allow(clippy::too_many_arguments)]
    fn inference<I: IntoIterator<Item = (Id<Term>, Id<Term>)>>(
        &mut self,
        _symbols: &Symbols,
        _terms: &Terms,
        _literals: &Literals,
        _inference: &'static str,
        _equations: I,
        _literal: Option<&Literal>,
        _deductions: &[Clause],
    ) {
    }

    fn unification<I: Iterator<Item = (Id<Variable>, Id<Term>)>>(
        &mut self,
        _symbols: &Symbols,
        _terms: &Terms,
        _bindings: I,
    ) {
    }

    fn statistic<T: Display>(&mut self, _key: &'static str, _value: T) {}
}

pub(crate) struct Silent;
impl Record for Silent {}
