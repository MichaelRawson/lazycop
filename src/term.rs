use crate::prelude::*;

pub(crate) struct Argument;
pub(crate) struct Variable;

#[derive(Clone, Copy)]
pub(crate) enum TermView {
    Variable(Id<Variable>),
    Function(Id<Symbol>, Range<Argument>),
}

#[derive(Clone, Copy)]
pub(crate) union Term {
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
pub(crate) struct Terms {
    terms: Block<Term>,
    save: Id<Term>,
}

impl Terms {
    pub(crate) fn len(&self) -> Id<Term> {
        self.terms.len()
    }

    pub(crate) fn offset(&self) -> Offset<Term> {
        self.terms.offset()
    }

    pub(crate) fn clear(&mut self) {
        self.terms.clear();
    }

    pub(crate) fn save(&mut self) {
        self.save = self.terms.len();
    }

    pub(crate) fn restore(&mut self) {
        self.terms.truncate(self.save);
    }

    pub(crate) fn extend(&mut self, other: &Self) {
        self.terms.extend(&other.terms);
    }

    pub(crate) fn add_variable(&mut self) -> Id<Term> {
        let symbol = None;
        let term = Term { symbol };
        self.terms.push(term)
    }

    pub(crate) fn add_function(
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

    pub(crate) fn fresh_function(
        &mut self,
        symbols: &Symbols,
        symbol: Id<Symbol>,
    ) -> Id<Term> {
        let arity = symbols[symbol].arity;
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

    pub(crate) fn subterms<F: FnMut(Id<Term>)>(
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

    pub(crate) fn proper_subterms<F: FnMut(Id<Term>)>(
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

    pub(crate) fn resolve(&self, argument: Id<Argument>) -> Id<Term> {
        let id = argument.transmute();
        id + self.terms[id].as_offset()
    }

    pub(crate) fn view(&self, symbols: &Symbols, id: Id<Term>) -> TermView {
        match self.terms[id].as_symbol() {
            Some(symbol) => {
                let arity = symbols[symbol].arity;
                let start = (id + Offset::new(1)).transmute();
                let args = Range::new_with_len(start, arity);
                TermView::Function(symbol, args)
            }
            None => TermView::Variable(id.transmute()),
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
        self.terms[id].as_symbol().is_none()
    }

    pub(crate) fn symbol(&self, id: Id<Term>) -> Id<Symbol> {
        some(self.terms[id].as_symbol())
    }

    fn add_reference(&mut self, referred: Id<Term>) -> Id<Term> {
        let id = self.terms.len();
        let offset = referred - id;
        let term = Term { offset };
        self.terms.push(term)
    }
}
