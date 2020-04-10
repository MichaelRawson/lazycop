use crate::prelude::*;
use crate::util::id_map::IdMap;

pub(crate) trait Record {
    fn start(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause_storage: &ClauseStorage,
        _clause: Clause,
    ) {
    }

    fn reduction(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause_storage: &ClauseStorage,
        _clause: Clause,
        _left: Id<Term>,
        _right: Id<Term>,
    ) {
    }

    fn lemma(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause_storage: &ClauseStorage,
        _clause: Clause,
        _left: Id<Term>,
        _right: Id<Term>,
    ) {
    }

    fn equality_reduction(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause_storage: &ClauseStorage,
        _clause: Clause,
        _left: Id<Term>,
        _right: Id<Term>,
    ) {
    }

    fn extension(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause_storage: &ClauseStorage,
        _clause: Clause,
        _extension_clause: Clause,
    ) {
    }

    fn constraint_solving(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _bindings: &IdMap<Variable, Option<Id<Term>>>,
    ) {
    }
}

pub(crate) struct Silent;
impl Record for Silent {}
