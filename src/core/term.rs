use crate::prelude::*;

pub struct Term;

#[derive(Clone, Copy)]
pub enum TermView {
    Variable,
    Function(Id<Symbol>, IdRange<Term>),
}

enum Item {
    Symbol(Id<Symbol>),
    Reference(Offset<Term>),
}

#[derive(Default)]
pub struct TermList {
    items: Vec<Item>,
}

impl TermList {
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn add_variable(&mut self) -> Id<Term> {
        let id = self.items.len().into();
        self.add_reference(id)
    }

    pub fn add_reference(&mut self, referred: Id<Term>) -> Id<Term> {
        let id = self.items.len().into();
        let offset = referred - id;
        self.items.push(Item::Reference(offset));
        id
    }

    pub fn add_symbol(&mut self, symbol: Id<Symbol>) -> Id<Term> {
        let id = self.items.len().into();
        self.items.push(Item::Symbol(symbol));
        id
    }

    pub fn view(&self, symbol_list: &SymbolList, id: Id<Term>) -> TermView {
        let mut current = id;
        loop {
            match self.items[current.index()] {
                Item::Symbol(symbol) => {
                    let arity = symbol_list.arity(symbol);
                    let args = IdRange::after(id, arity);
                    return TermView::Function(symbol, args);
                }
                Item::Reference(offset) => {
                    let new = current + offset;
                    if current == new {
                        return TermView::Variable;
                    }
                    current = new;
                }
            }
        }
    }
}
