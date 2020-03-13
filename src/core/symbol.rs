use crate::prelude::*;
use std::fmt;

pub struct Symbol;

pub struct Symbols {
    arities: Vec<u32>,
    names: Vec<String>,
    pub equality: Id<Symbol>
}

impl Symbols {
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

impl Default for Symbols {
    fn default() -> Self {
        let arities = vec![];
        let names = vec![];
        let equality = 0.into();
        let mut symbols = Self { arities, names, equality };
        symbols.add("=".into(), 2);
        symbols
    }
}

impl fmt::Display for Pair<&Symbols, Id<Symbol>> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Pair(symbols, id) = self;
        write!(f, "{}", symbols.name(*id))
    }
}
