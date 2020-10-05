use crate::prelude::*;

pub(crate) enum Name {
    Regular(String),
    Quoted(String),
    Skolem(usize),
    Definition(usize),
}

pub(crate) struct Symbol {
    pub(crate) arity: u32,
    pub(crate) name: Name,
}

pub(crate) type Symbols = Block<Symbol>;
