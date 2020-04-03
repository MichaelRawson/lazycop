use crate::prelude::*;

pub struct Term;

#[derive(Clone, Copy)]
pub enum TermView {
    Variable(Id<Term>),
    Function(Id<Symbol>, IdRange<Term>),
}

#[derive(Clone, Copy)]
enum Item {
    Symbol(Id<Symbol>),
    Reference(Offset<Term>),
}

#[derive(Default)]
pub struct TermGraph {
    items: Vec<Item>,
}

impl TermGraph {
    pub fn clear(&mut self) {
        self.items.clear();
    }

    pub fn add_variable(&mut self) -> Id<Term> {
        let id = self.items.len().into();
        self.add_reference(id)
    }

    pub fn add_function(
        &mut self,
        symbol: Id<Symbol>,
        args: &[Id<Term>],
    ) -> Id<Term> {
        let id = self.items.len().into();
        self.items.push(Item::Symbol(symbol));
        for arg in args {
            self.add_reference(*arg);
        }
        id
    }

    pub fn current_offset(&self) -> Offset<Term> {
        let base: Id<Term> = 0.into();
        let current: Id<Term> = self.items.len().into();
        current - base
    }

    pub fn copy_from(&mut self, other: &Self) {
        self.items.extend_from_slice(&other.items);
    }

    pub fn resolve_reference(&self, id: Id<Term>) -> Id<Term> {
        match self.items[id.index()] {
            Item::Reference(offset) => id + offset,
            _ => id,
        }
    }

    pub fn view(&self, symbol_table: &SymbolTable, id: Id<Term>) -> TermView {
        let id = self.resolve_reference(id);
        match self.items[id.index()] {
            Item::Symbol(symbol) => {
                let arity = symbol_table.arity(symbol);
                let args = IdRange::after(id, arity);
                TermView::Function(symbol, args)
            }
            Item::Reference(_) => TermView::Variable(id),
        }
    }

    fn add_reference(&mut self, referred: Id<Term>) -> Id<Term> {
        let id = self.items.len().into();
        let offset = referred - id;
        self.items.push(Item::Reference(offset));
        id
    }
}
