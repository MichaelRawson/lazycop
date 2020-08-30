use crate::prelude::*;

#[derive(Debug)]
pub(crate) enum Name {
    Regular(String),
    Quoted(String),
    Skolem(u32),
    Definition(u32),
}

#[derive(Debug)]
pub(crate) struct Symbol {
    pub(crate) arity: u32,
    pub(crate) name: Name,
}

pub(crate) type Symbols = Block<Symbol>;
