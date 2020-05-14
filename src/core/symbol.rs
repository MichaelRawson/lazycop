use crate::prelude::*;

pub(crate) struct Symbol {
    name: String,
}

#[derive(Default)]
pub(crate) struct SymbolTable {
    names: Block<Symbol>,
}

impl SymbolTable {
    pub(crate) fn append(&mut self, name: String) -> Id<Symbol> {
        self.names.push(Symbol { name }).transmute()
    }

    pub(crate) fn name(&self, id: Id<Symbol>) -> &str {
        &self.names[id].name
    }

    pub(crate) fn len(&self) -> Id<Symbol> {
        self.names.len()
    }
}
