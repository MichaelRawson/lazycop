use crate::prelude::*;

pub(crate) struct Symbol;

#[derive(Default)]
pub(crate) struct SymbolTable {
    names: Arena<String>,
}

impl SymbolTable {
    pub(crate) fn append(&mut self, name: String) -> Id<Symbol> {
        self.names.push(name).transmute()
    }

    pub(crate) fn name(&self, id: Id<Symbol>) -> &str {
        &self.names[id.transmute()]
    }
}
