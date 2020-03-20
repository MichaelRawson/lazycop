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

    pub fn current_offset(&self) -> Offset<Term> {
        let base: Id<Term> = 0.into();
        let current: Id<Term> = self.items.len().into();
        current - base
    }

    pub fn copy_from(&mut self, other: &Self) {
        self.items.extend_from_slice(&other.items);
    }

    pub fn view(&self, symbol_table: &SymbolTable, id: Id<Term>) -> TermView {
        let id = self.chase_references(id);
        match self.items[id.index()] {
            Item::Symbol(symbol) => {
                let arity = symbol_table.arity(symbol);
                let args = IdRange::after(id, arity);
                TermView::Function(symbol, args)
            }
            Item::Reference(refloop) => {
                assert!(refloop.is_zero());
                TermView::Variable(id)
            }
        }
    }

    pub fn bind(&mut self, variable: Id<Term>, term: Id<Term>) {
        let term = self.chase_references(term);
        let offset = term - variable;
        let refloop = match &mut self.items[variable.index()] {
            Item::Reference(refloop) => refloop,
            _ => unreachable!(),
        };
        assert!(refloop.is_zero());
        *refloop = offset;
    }

    fn chase_references(&self, id: Id<Term>) -> Id<Term> {
        let mut current = id;
        loop {
            match self.items[current.index()] {
                Item::Symbol(_) => {
                    return current;
                }
                Item::Reference(offset) if offset.is_zero() => {
                    return current;
                }
                Item::Reference(offset) => {
                    current = current + offset;
                }
            }
        }
    }

    fn add_reference(&mut self, referred: Id<Term>) -> Id<Term> {
        let id = self.items.len().into();
        let offset = referred - id;
        self.items.push(Item::Reference(offset));
        id
    }
}
