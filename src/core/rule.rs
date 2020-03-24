use crate::prelude::*;

#[derive(Clone, Copy)]
pub enum Rule {
    Start(Id<Clause>),
    Extension(Id<(Clause, Literal)>),
    Reduction(Id<Literal>),
    Merge(Id<Literal>),
    Lemma(Id<Literal>),
    Symmetry,
}
