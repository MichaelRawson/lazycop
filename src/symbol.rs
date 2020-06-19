use crate::prelude::*;

pub struct Symbol;

#[derive(Default)]
pub struct Symbols {
    arities: Block<u32>,
    names: Block<String>,
}

impl Symbols {
    pub fn is_empty(&self) -> bool {
        self.arities.is_empty()
    }

    pub fn len(&self) -> Id<Symbol> {
        self.arities.len().transmute()
    }

    pub fn add(&mut self, arity: u32, name: String) -> Id<Symbol> {
        self.arities.push(arity);
        self.names.push(name).transmute()
    }

    pub fn arity(&self, symbol: Id<Symbol>) -> u32 {
        self.arities[symbol.transmute()]
    }

    pub fn name(&self, symbol: Id<Symbol>) -> &str {
        &self.names[symbol.transmute()]
    }
}
