use crate::prelude::*;

pub trait Policy {
    fn unify(
        symbol_table: &SymbolTable,
        term_graph: &mut TermGraph,
        left: Id<Term>,
        right: Id<Term>,
    ) -> bool;
}

pub struct Unchecked;

impl Policy for Unchecked {
    fn unify(
        symbol_table: &SymbolTable,
        term_graph: &mut TermGraph,
        left: Id<Term>,
        right: Id<Term>,
    ) -> bool {
        unify_unchecked(symbol_table, term_graph, left, right)
    }
}

pub struct Checked;

impl Policy for Checked {
    fn unify(
        symbol_table: &SymbolTable,
        term_graph: &mut TermGraph,
        left: Id<Term>,
        right: Id<Term>,
    ) -> bool {
        unify_checked(symbol_table, term_graph, left, right)
    }
}
