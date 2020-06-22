use crate::prelude::*;

pub struct Symbol {
    pub arity: u32,
    pub name: String,
}

pub type Symbols = Block<Symbol>;
