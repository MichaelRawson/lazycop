use crate::prelude::*;

pub(crate) enum Name {
    Regular(String),
    Quoted(String),
    Distinct(String),
    Skolem(usize),
    Definition(usize),
}

pub(crate) struct Symbol {
    pub(crate) arity: u32,
    pub(crate) name: Name,
}

impl Symbol {
    pub(crate) fn is_distinct_object(&self) -> bool {
        matches!(self.name, Name::Distinct(_))
    }
}

pub(crate) type Symbols = Block<Symbol>;
