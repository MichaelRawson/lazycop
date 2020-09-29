use crate::prelude::*;
use std::fmt::Display;

pub(crate) trait Inference: Sized {
    fn new(name: &'static str) -> Self;

    fn axiom(
        &mut self,
        _id: Id<ProblemClause>,
        _literals: Range<Literal>,
    ) -> &mut Self {
        self
    }

    fn lemma(&mut self, _lemma: Id<Literal>) -> &mut Self {
        self
    }

    fn equation(&mut self, _left: Id<Term>, _right: Id<Term>) -> &mut Self {
        self
    }

    fn deduction(&mut self, _literals: Range<Literal>) -> &mut Self {
        self
    }
}

pub(crate) trait Record {
    type Inference: Inference;

    fn inference(
        &mut self,
        _problem: &Problem,
        _terms: &Terms,
        _literals: &Literals,
        _inference: &Self::Inference,
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
