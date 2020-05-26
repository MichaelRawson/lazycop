use crate::prelude::*;

pub(crate) struct Symbol {
    name: String,
}

#[derive(Default)]
pub(crate) struct Symbols {
    names: Block<Symbol>,
}

impl Symbols {
    pub(crate) fn append(&mut self, name: String) -> Id<Symbol> {
        self.names.push(Symbol { name })
    }

    pub(crate) fn name(&self, id: Id<Symbol>) -> &str {
        &self.names[id].name
    }
}
