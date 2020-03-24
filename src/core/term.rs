use crate::prelude::*;
use std::hint::unreachable_unchecked;

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

    pub fn view(&self, symbol_table: &SymbolTable, id: Id<Term>) -> TermView {
        let id = self.chase_references(id);
        match *self.get_item(id) {
            Item::Symbol(symbol) => {
                let arity = symbol_table.arity(symbol);
                let args = IdRange::after(id, arity);
                TermView::Function(symbol, args)
            }
            Item::Reference(_) => TermView::Variable(id),
        }
    }

    pub fn bind(&mut self, variable: Id<Term>, term: Id<Term>) {
        let term = self.chase_references(term);
        let offset = term - variable;
        let refloop = match self.get_item_mut(variable) {
            Item::Reference(refloop) => refloop,
            _ => unsafe { unreachable_unchecked() },
        };
        *refloop = offset;
    }

    pub fn bind_vars(&mut self, v1: Id<Term>, v2: Id<Term>) {
        let (variable, term) = if v1 > v2 { (v1, v2) } else { (v2, v1) };
        let offset = term - variable;
        let refloop = match self.get_item_mut(variable) {
            Item::Reference(refloop) => refloop,
            _ => unsafe { unreachable_unchecked() },
        };
        *refloop = offset;
    }

    fn chase_references(&self, id: Id<Term>) -> Id<Term> {
        let mut current = id;
        loop {
            let next = match self.get_item(current) {
                Item::Reference(offset) => current + *offset,
                _ => current,
            };
            if current == next {
                return current;
            }
            current = next;
        }
    }

    fn add_reference(&mut self, referred: Id<Term>) -> Id<Term> {
        let id = self.items.len().into();
        let offset = referred - id;
        self.items.push(Item::Reference(offset));
        id
    }

    fn get_item(&self, id: Id<Term>) -> &Item {
        unsafe { self.items.get_unchecked(id.index()) }
    }

    fn get_item_mut(&mut self, id: Id<Term>) -> &mut Item {
        unsafe { self.items.get_unchecked_mut(id.index()) }
    }
}
