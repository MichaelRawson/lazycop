use crate::core::unification::UnificationPolicy;
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
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        other: &Self,
    ) -> bool {
        self.polarity != other.polarity
            && self.atom.might_unify(symbol_table, term_graph, &other.atom)
    }

    pub fn resolve<U: UnificationPolicy>(
        &self,
        symbol_table: &SymbolTable,
        term_graph: &mut TermGraph,
        other: &Self,
    ) -> bool {
        assert_ne!(self.polarity, other.polarity);
        self.atom.unify::<U>(symbol_table, term_graph, &other.atom)
    }

    pub fn might_equality_reduce(
        &self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
    ) -> bool {
        !self.polarity && self.atom.might_self_unify(symbol_table, term_graph)
    }

    pub fn equality_reduce<U: UnificationPolicy>(
        &self,
        symbol_table: &SymbolTable,
        term_graph: &mut TermGraph,
    ) -> bool {
        assert!(!self.polarity);
        self.atom.self_unify::<U>(symbol_table, term_graph)
    }

    pub fn lazy_disequalities<'symbol, 'term, 'iterator>(
        &self,
        symbol_table: &'symbol SymbolTable,
        term_graph: &'term mut TermGraph,
        other: &Self,
    ) -> impl Iterator<Item = Self> + 'iterator
    where
        'symbol: 'iterator,
        'term: 'iterator,
    {
        assert_ne!(self.polarity, other.polarity);
        self.atom
            .lazy_constraints(symbol_table, term_graph, &other.atom)
            .map(|eq| Self::new(false, eq))
    }
}
