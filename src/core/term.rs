use crate::prelude::*;

pub struct Term;

#[derive(Clone, Copy)]
pub enum TermView {
    Variable,
    Function(Id<Symbol>, IdRange<Term>)
}

#[derive(Clone, Copy)]
enum Flavour {
    Reference,
    Symbol
}

#[repr(C)]
union Item {
    offset: Offset<Term>,
    symbol: Id<Symbol>
}

#[derive(Default)]
pub struct TermList {
    flavours: Vec<Flavour>,
    items: Vec<Item>
}

impl TermList {
    pub fn add_variable(&mut self) -> Id<Term> {
        let id = self.items.len().into();
        self.add_reference(id)
    }

    pub fn add_reference(&mut self, referred: Id<Term>) -> Id<Term> {
        let id = self.items.len().into();
        let offset = referred - id;
        let item = Item { offset };
        self.flavours.push(Flavour::Reference);
        self.items.push(item);
        id
    }

    pub fn add_symbol(&mut self, symbol: Id<Symbol>) -> Id<Term> {
        let id = self.items.len().into();
        let item = Item { symbol };
        self.flavours.push(Flavour::Symbol);
        self.items.push(item);
        id
    }

    pub fn view(&self, symbols: &Symbols, mut id: Id<Term>) -> TermView {
        while let Flavour::Reference = self.flavours[id.index()] {
            let offset = unsafe { self.items[id.index()].offset };
            let new_id = id + offset;
            if new_id == id {
                return TermView::Variable
            }
            id = new_id;
        }

        let symbol = unsafe { self.items[id.index()].symbol };
        let arity = symbols.arity(symbol);
        let args = IdRange::after(id, arity);
        TermView::Function(symbol, args)
    }
}
