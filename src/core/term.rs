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
pub struct TermList {
    items: Vec<Item>,
}

impl TermList {
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

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
                        return TermView::Variable(current);
                    }
                    current = new;
                }
            }
        }
    }

    pub fn current_offset(&self) -> Offset<Term> {
        let base: Id<Term> = 0.into();
        let current: Id<Term> = self.items.len().into();
        current - base
    }

    pub fn copy_from(&mut self, other: &Self) {
        self.items.extend_from_slice(&other.items);
    }

    fn add_reference(&mut self, referred: Id<Term>) -> Id<Term> {
        let id = self.items.len().into();
        let offset = referred - id;
        self.items.push(Item::Reference(offset));
        id
    }
}
