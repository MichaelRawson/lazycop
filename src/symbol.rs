use crate::prelude::*;

pub(crate) struct Symbol;

#[derive(Default)]
pub(crate) struct Symbols {
    arities: Block<u32>,
    names: Block<String>,
}

impl Symbols {
    pub(crate) fn add(&mut self, arity: u32, name: String) -> Id<Symbol> {
        self.arities.push(arity);
        self.names.push(name).transmute()
    }

    pub(crate) fn len(&self) -> Id<Symbol> {
        self.arities.len().transmute()
    }

    pub(crate) fn arity(&self, symbol: Id<Symbol>) -> u32 {
        self.arities[symbol.transmute()]
    }

    pub(crate) fn name(&self, symbol: Id<Symbol>) -> &str {
        &self.names[symbol.transmute()]
    }
}
