use crate::clause::Clause;
use crate::prelude::*;
use std::fmt::Display;

pub(crate) trait Inference: Sized {
    fn new(name: &'static str) -> Self;

    fn equation(&mut self, _left: Id<Term>, _right: Id<Term>) -> &mut Self {
        self
    }

    fn literal(&mut self, _literal: Id<Literal>) -> &mut Self {
        self
    }

    fn deduction(&mut self, _literals: Range<Literal>) -> &mut Self {
        self
    }
}

pub(crate) trait Record {
    type Inference: Inference;

    fn axiom(
        &mut self,
        _symbols: &Symbols,
        _terms: &Terms,
        _literals: &Literals,
        _axiom: Clause,
    ) {
    }

    fn inference(
        &mut self,
        _symbols: &Symbols,
        _terms: &Terms,
        _literals: &Literals,
        _inference: &mut Self::Inference,
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

pub(crate) struct SilentInference;

impl Inference for SilentInference {
    fn new(_name: &'static str) -> Self {
        Self
    }
}

pub(crate) struct Silent;

impl Record for Silent {
    type Inference = SilentInference;
}
