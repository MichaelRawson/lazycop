use crate::prelude::*;

pub(crate) trait Record {
    fn start_inference(&mut self, _inference: &'static str) {}
    fn therefore(&mut self) {}
    fn clause(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause_storage: &ClauseStorage,
        _clause: Clause,
    ) {
    }
    fn lemma(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _literal: Literal,
    ) {
    }
    fn equality_constraint(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _left: Id<Term>,
        _right: Id<Term>,
    ) {
    }
    fn binding(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _variable: Id<Variable>,
        _term: Id<Term>,
    ) {
    }

    fn end_inference(&mut self) {}
}

pub(crate) struct Silent;
impl Record for Silent {}
