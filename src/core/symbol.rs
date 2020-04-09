use crate::prelude::*;

pub struct Symbol;

#[derive(Default)]
pub struct SymbolTable {
    names: Arena<String>,
}

impl SymbolTable {
    pub fn append(&mut self, name: String) -> Id<Symbol> {
        self.names.push(name).transmute()
    }

    pub fn name(&self, id: Id<Symbol>) -> &str {
        &self.names[id.transmute()]
    }
}
