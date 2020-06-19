use crate::prelude::*;

pub struct Argument;
pub struct Variable;

#[derive(Clone, Copy)]
pub enum TermView {
    Variable(Id<Variable>),
    Function(Id<Symbol>, Range<Argument>),
}

#[derive(Clone, Copy)]
pub union Term {
    symbol: Option<Id<Symbol>>,
    offset: Offset<Term>,
}

impl Term {
    fn as_symbol(self) -> Option<Id<Symbol>> {
        unsafe { self.symbol }
    }

    fn as_offset(self) -> Offset<Term> {
        unsafe { self.offset }
    }
}

#[derive(Default)]
pub struct Terms {
    terms: Block<Term>,
    save: Id<Term>,
}

impl Terms {
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn len(&self) -> Id<Term> {
        self.terms.len()
    }

    pub fn offset(&self) -> Offset<Term> {
        self.terms.offset()
    }

    pub fn clear(&mut self) {
        self.terms.clear();
    }

    pub fn save(&mut self) {
        self.save = self.terms.len();
    }

    pub fn restore(&mut self) {
        self.terms.truncate(self.save);
    }

    pub fn extend(&mut self, other: &Self) {
        self.terms.extend(&other.terms);
    }

    pub fn add_variable(&mut self) -> Id<Term> {
        let symbol = None;
        let term = Term { symbol };
        self.terms.push(term)
    }

    pub fn add_function(
        &mut self,
        symbol: Id<Symbol>,
        args: &[Id<Term>],
    ) -> Id<Term> {
        let symbol = Some(symbol);
        let id = self.terms.push(Term { symbol });
        for arg in args {
            self.add_reference(*arg);
        }
        id
    }

    pub fn fresh_function(
        &mut self,
        symbols: &Symbols,
        symbol: Id<Symbol>,
    ) -> Id<Term> {
        let arity = symbols.arity(symbol);
        let symbol = Some(symbol);
        let start = self.terms.len();
        for _ in 0..arity {
            self.add_variable();
        }
        let id = self.terms.push(Term { symbol });
        for arg in Range::new_with_len(start, arity) {
            self.add_reference(arg);
        }
        id
    }

    pub fn subst(
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

                let symbol = Some(symbol);
                let start = self.terms.len();
                self.terms.push(Term { symbol });
                for arg in args {
                    if arg == changed {
                        self.add_reference(result);
                    } else {
                        self.add_reference(self.resolve(arg));
                    }
                }
                start
            }
        }
    }

    pub fn subterms<F: FnMut(Id<Term>)>(
        &self,
        symbols: &Symbols,
        term: Id<Term>,
        f: &mut F,
    ) {
        if let TermView::Function(_, args) = self.view(symbols, term) {
            f(term);
            for subterm in args.map(|arg| self.resolve(arg)) {
                self.subterms(symbols, subterm, f);
            }
        }
    }

    pub fn proper_subterms<F: FnMut(Id<Term>)>(
        &self,
        symbols: &Symbols,
        term: Id<Term>,
        f: &mut F,
    ) {
        if let TermView::Function(_, args) = self.view(symbols, term) {
            for subterm in args.map(|arg| self.resolve(arg)) {
                self.subterms(symbols, subterm, f);
            }
        }
    }

    pub fn resolve(&self, argument: Id<Argument>) -> Id<Term> {
        let id = argument.transmute();
        id + self.terms[id].as_offset()
    }

    pub fn view(&self, symbols: &Symbols, id: Id<Term>) -> TermView {
        match self.terms[id].as_symbol() {
            Some(symbol) => {
                let arity = symbols.arity(symbol);
                let start = (id + Offset::new(1)).transmute();
                let args = Range::new_with_len(start, arity);
                TermView::Function(symbol, args)
            }
            None => TermView::Variable(id.transmute()),
        }
    }

    pub fn arguments(
        &self,
        symbols: &Symbols,
        id: Id<Term>,
    ) -> Range<Argument> {
        let symbol = self.symbol(id);
        let arity = symbols.arity(symbol);
        let start = (id + Offset::new(1)).transmute();
        Range::new_with_len(start, arity)
    }

    pub fn is_variable(&self, id: Id<Term>) -> bool {
        self.terms[id].as_symbol().is_none()
    }

    pub fn symbol(&self, id: Id<Term>) -> Id<Symbol> {
        some(self.terms[id].as_symbol())
    }

    fn add_reference(&mut self, referred: Id<Term>) -> Id<Term> {
        let id = self.terms.len();
        let offset = referred - id;
        let term = Term { offset };
        self.terms.push(term)
    }
}
