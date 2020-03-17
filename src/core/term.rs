use crate::prelude::*;

pub struct Term;

#[derive(Clone, Copy)]
pub enum TermView {
    Variable,
    Function(Id<Symbol>, IdRange<Term>),
}

bitflags! {
    struct TermFlags: u8 {
        const REFERENCE = 0b1;
    }
}

#[repr(C)]
union Item {
    offset: Offset<Term>,
    symbol: Id<Symbol>,
}

#[derive(Default)]
pub struct TermList {
    flags: Vec<TermFlags>,
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
        let item = Item { offset };
        let flags = TermFlags::REFERENCE;
        self.flags.push(flags);
        self.items.push(item);
        id
    }

    pub fn add_symbol(&mut self, symbol: Id<Symbol>) -> Id<Term> {
        let id = self.items.len().into();
        let item = Item { symbol };
        let flags = TermFlags::empty();
        self.flags.push(flags);
        self.items.push(item);
        id
    }

    pub fn view(
        &self,
        symbol_list: &SymbolList,
        mut id: Id<Term>,
    ) -> TermView {
        while self.flags[id.index()].contains(TermFlags::REFERENCE) {
            let offset = unsafe { self.items[id.index()].offset };
            let new_id = id + offset;
            if new_id == id {
                return TermView::Variable;
            }
            id = new_id;
        }

        let symbol = unsafe { self.items[id.index()].symbol };
        let arity = symbol_list.arity(symbol);
        let args = IdRange::after(id, arity);
        TermView::Function(symbol, args)
    }
}
