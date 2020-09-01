use crate::prelude::*;

pub(crate) struct Argument;
pub(crate) struct Variable;
pub(crate) struct Term;

#[derive(Clone, Copy)]
pub(crate) enum TermView {
    Variable(Id<Variable>),
    Function(Id<Symbol>, Range<Argument>),
}

#[derive(Clone, Copy)]
enum Item {
    Symbol(Id<Symbol>),
    Offset(Offset<Item>),
}

#[derive(Default)]
pub(crate) struct Terms {
    items: Block<Item>,
    save: Id<Item>,
}

impl Terms {
    pub(crate) fn len(&self) -> Id<Term> {
        self.items.len().transmute()
    }

    pub(crate) fn current_offset(&self) -> Offset<Term> {
        self.items.offset().transmute()
    }

    pub(crate) fn clear(&mut self) {
        self.items.clear();
    }

    pub(crate) fn save(&mut self) {
        self.save = self.items.len();
    }

    pub(crate) fn restore(&mut self) {
        self.items.truncate(self.save);
    }

    pub(crate) fn extend(&mut self, other: &Self) {
        self.items.extend(&other.items);
    }

    pub(crate) fn add_variable(&mut self) -> Id<Term> {
        let item = Item::Offset(Offset::new(0));
        self.items.push(item).transmute()
    }

    pub(crate) fn add_function(
        &mut self,
        symbol: Id<Symbol>,
        args: Vec<Id<Term>>,
    ) -> Id<Term> {
        let id = self.items.push(Item::Symbol(symbol));
        for arg in args {
            self.add_reference(arg.transmute());
        }
        id.transmute()
    }

    pub(crate) fn fresh_function(
        &mut self,
        symbols: &Symbols,
        symbol: Id<Symbol>,
    ) -> Id<Term> {
        let arity = symbols[symbol].arity;
        let start = self.items.len();
        for _ in 0..arity {
            self.add_variable();
        }
        let id = self.items.push(Item::Symbol(symbol));
        for arg in Range::new_with_len(start, arity) {
            self.add_reference(arg);
        }
        id.transmute()
    }

    pub(crate) fn subst(
        &mut self,
        symbols: &Symbols,
        term: Id<Term>,
        from: Id<Term>,
        to: Id<Term>,
    ) -> Id<Term> {
        if term == from {
            return to;
        }
        match self.view(symbols, term) {
            TermView::Variable(_) => term,
            TermView::Function(symbol, args) => {
                let mut changed = None;
                for arg in args {
                    let subterm = self.resolve(arg);
                    let result = self.subst(symbols, subterm, from, to);
                    if result != subterm {
                        changed = Some((arg, result));
                    }
                }
                let (changed, result) = if let Some((arg, result)) = changed {
                    (arg, result)
                } else {
                    return term;
                };

                let start = self.items.push(Item::Symbol(symbol));
                for arg in args {
                    if arg == changed {
                        self.add_reference(result.transmute());
                    } else {
                        self.add_reference(self.resolve(arg).transmute());
                    }
                }
                start.transmute()
            }
        }
    }

    pub(crate) fn subterms<F: FnMut(Id<Term>)>(
        &self,
        symbols: &Symbols,
        term: Id<Term>,
        f: &mut F,
    ) {
        if let TermView::Function(_, args) = self.view(symbols, term) {
            f(term);
            for subterm in args.into_iter().map(|arg| self.resolve(arg)) {
                self.subterms(symbols, subterm, f);
            }
        }
    }

    pub(crate) fn proper_subterms<F: FnMut(Id<Term>)>(
        &self,
        symbols: &Symbols,
        term: Id<Term>,
        f: &mut F,
    ) {
        if let TermView::Function(_, args) = self.view(symbols, term) {
            for subterm in args.into_iter().map(|arg| self.resolve(arg)) {
                self.subterms(symbols, subterm, f);
            }
        }
    }

    pub(crate) fn resolve(&self, argument: Id<Argument>) -> Id<Term> {
        let item = argument.transmute();
        let resolved = item + self.offset(item);
        resolved.transmute()
    }

    pub(crate) fn view(&self, symbols: &Symbols, id: Id<Term>) -> TermView {
        match self.items[id.transmute()] {
            Item::Symbol(symbol) => {
                let arity = symbols[symbol].arity;
                let start = (id + Offset::new(1)).transmute();
                let args = Range::new_with_len(start, arity);
                TermView::Function(symbol, args)
            }
            Item::Offset(offset) => {
                debug_assert!(offset.is_zero());
                TermView::Variable(id.transmute())
            }
        }
    }

    pub(crate) fn symbol(&self, id: Id<Term>) -> Id<Symbol> {
        match self.items[id.transmute()] {
            Item::Symbol(symbol) => symbol,
            _ => unreachable(),
        }
    }

    pub(crate) fn arguments(
        &self,
        symbols: &Symbols,
        id: Id<Term>,
    ) -> Range<Argument> {
        let symbol = self.symbol(id);
        let arity = symbols[symbol].arity;
        let start = (id + Offset::new(1)).transmute();
        Range::new_with_len(start, arity)
    }

    pub(crate) fn is_variable(&self, id: Id<Term>) -> bool {
        match self.items[id.transmute()] {
            Item::Offset(offset) => offset.is_zero(),
            _ => false,
        }
    }

    fn offset(&self, id: Id<Item>) -> Offset<Item> {
        match self.items[id.transmute()] {
            Item::Offset(offset) => offset,
            _ => unreachable(),
        }
    }

    fn add_reference(&mut self, referred: Id<Item>) {
        let id = self.items.len();
        let offset = referred - id;
        let item = Item::Offset(offset);
        self.items.push(item);
    }
}
