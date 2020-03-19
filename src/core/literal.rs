use crate::prelude::*;

#[derive(Clone, Copy)]
pub struct Literal {
    pub polarity: bool,
    pub atom: Atom,
}

impl Literal {
    pub fn new(polarity: bool, atom: Atom) -> Self {
        Self { polarity, atom }
    }

    pub fn is_predicate(&self) -> bool {
        self.atom.is_predicate()
    }

    pub fn predicate_term(&self) -> Id<Term> {
        self.atom.predicate_term()
    }

    pub fn offset(&mut self, offset: Offset<Term>) {
        self.atom.offset(offset);
    }

    pub fn might_resolve(
        &self,
        symbol_list: &SymbolList,
        term_list: &TermList,
        other: &Self,
    ) -> bool {
        self.polarity != other.polarity
            && self.atom.might_unify(symbol_list, term_list, &other.atom)
    }

    pub fn resolve(
        &self,
        symbol_list: &SymbolList,
        term_list: &mut TermList,
        other: &Self,
    ) -> bool {
        assert_ne!(self.polarity, other.polarity);
        self.atom.unify(symbol_list, term_list, &other.atom)
    }

    pub fn might_equality_reduce(
        &self,
        symbol_list: &SymbolList,
        term_list: &TermList,
    ) -> bool {
        !self.polarity && self.atom.might_self_unify(symbol_list, term_list)
    }

    pub fn equality_reduce(
        &self,
        symbol_list: &SymbolList,
        term_list: &mut TermList,
    ) -> bool {
        assert!(!self.polarity);
        self.atom.self_unify(symbol_list, term_list)
    }

    pub fn lazy_disequalities<'symbol, 'term, 'iterator>(
        &self,
        symbol_list: &'symbol SymbolList,
        term_list: &'term TermList,
        other: &Self,
    ) -> impl Iterator<Item = Self> + 'iterator
    where
        'symbol: 'iterator,
        'term: 'iterator,
    {
        assert_ne!(self.polarity, other.polarity);
        self.atom
            .lazy_constraints(symbol_list, term_list, &other.atom)
            .map(|eq| Self::new(false, eq))
    }
}
