use crate::prelude::*;

pub(crate) struct Symbol {
    pub(crate) arity: u32,
    pub(crate) name: String,
}

pub(crate) type Symbols = Block<Symbol>;
