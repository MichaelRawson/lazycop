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

    pub fn equal(
        &self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        other: &Self,
    ) -> bool {
        self.polarity == other.polarity
            && self.atom.equal(symbol_table, term_graph, &other.atom)
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

    pub fn resolve<P: Policy>(
        &self,
        symbol_table: &SymbolTable,
        term_graph: &mut TermGraph,
        other: &Self,
    ) -> bool {
        assert_ne!(self.polarity, other.polarity);
        self.atom.unify::<P>(symbol_table, term_graph, &other.atom)
    }

    pub fn might_equality_unify(
        &self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
    ) -> bool {
        !self.polarity
            && self.atom.might_equality_unify(symbol_table, term_graph)
    }

    pub fn equality_unify<P: Policy>(
        &self,
        symbol_table: &SymbolTable,
        term_graph: &mut TermGraph,
    ) -> bool {
        assert!(!self.polarity);
        self.atom.self_unify::<P>(symbol_table, term_graph)
    }

    pub fn resolve_or_disequations<'symbol, 'term, 'iterator>(
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
            .unify_or_disequations(symbol_table, term_graph, &other.atom)
            .map(|eq| Self::new(false, eq))
    }
}
