use crate::prelude::*;

#[derive(Clone, Copy)]
pub enum Atom {
    Predicate(Id<Term>),
    Equality(Id<Term>, Id<Term>),
}

impl Atom {
    pub fn is_predicate(&self) -> bool {
        match self {
            Atom::Predicate(_) => true,
            _ => false,
        }
    }

    pub fn predicate_term(&self) -> Id<Term> {
        match self {
            Atom::Predicate(p) => *p,
            _ => unreachable!(),
        }
    }

    pub fn offset(&mut self, offset: Offset<Term>) {
        match self {
            Atom::Predicate(p) => {
                *p = *p + offset;
            }
            Atom::Equality(left, right) => {
                *left = *left + offset;
                *right = *right + offset;
            }
        }
    }

    pub fn might_unify(
        &self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        other: &Self,
    ) -> bool {
        match (self, other) {
            (Atom::Predicate(p), Atom::Predicate(q)) => {
                might_unify(symbol_table, term_graph, *p, *q)
            }
            _ => false,
        }
    }

    pub fn unify<P: Policy>(
        &self,
        symbol_table: &SymbolTable,
        term_graph: &mut TermGraph,
        other: &Self,
    ) -> bool {
        match (self, other) {
            (Atom::Predicate(p), Atom::Predicate(q)) => {
                P::unify(symbol_table, term_graph, *p, *q)
            }
            _ => unreachable!(),
        }
    }

    pub fn might_equality_unify(
        &self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
    ) -> bool {
        match self {
            Atom::Equality(left, right) => {
                might_unify(symbol_table, term_graph, *left, *right)
            }
            _ => false,
        }
    }

    pub fn self_unify<P: Policy>(
        &self,
        symbol_table: &SymbolTable,
        term_graph: &mut TermGraph,
    ) -> bool {
        match self {
            Atom::Equality(left, right) => {
                P::unify(symbol_table, term_graph, *left, *right)
            }
            _ => unreachable!(),
        }
    }

    pub fn unify_or_disequations<'symbol, 'term, 'iterator>(
        &self,
        symbol_table: &'symbol SymbolTable,
        term_graph: &'term mut TermGraph,
        other: &Self,
    ) -> impl Iterator<Item = Self> + 'iterator
    where
        'symbol: 'iterator,
        'term: 'iterator,
    {
        match (self, other) {
            (Atom::Predicate(p), Atom::Predicate(q)) => {
                unify_or_disequations(symbol_table, term_graph, *p, *q)
                    .map(|(left, right)| Atom::Equality(left, right))
            }
            _ => unreachable!(),
        }
    }
}
