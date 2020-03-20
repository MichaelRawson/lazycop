use crate::prelude::*;

pub struct Symbol;

#[derive(Default)]
pub struct SymbolTable {
    arities: Vec<u32>,
    names: Vec<String>,
}

impl SymbolTable {
    pub fn add(&mut self, name: String, arity: u32) -> Id<Symbol> {
        let id = self.arities.len().into();
        self.arities.push(arity);
        self.names.push(name);
        id
    }

    pub fn arity(&self, id: Id<Symbol>) -> u32 {
        self.arities[id.index()]
    }

    pub fn name(&self, id: Id<Symbol>) -> &str {
        &self.names[id.index()]
    }
}
