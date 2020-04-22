use crate::prelude::*;
use crate::util::id_map::IdMap;

pub(crate) trait Record {
    fn start<T: Iterator<Item = Id<Literal>>>(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause_storage: &ClauseStorage,
        _literals: T,
    ) {
    }

    fn reduction<T: Iterator<Item = Id<Literal>>>(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause_storage: &ClauseStorage,
        _literals: T,
        _left: Id<Term>,
        _right: Id<Term>,
    ) {
    }

    fn lemma<T: Iterator<Item = Id<Literal>>>(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause_storage: &ClauseStorage,
        _literals: T,
        _left: Id<Term>,
        _right: Id<Term>,
    ) {
    }

    fn equality_reduction<T: Iterator<Item = Id<Literal>>>(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause_storage: &ClauseStorage,
        _literals: T,
        _left: Id<Term>,
        _right: Id<Term>,
    ) {
    }

    fn extension<
        T: Iterator<Item = Id<Literal>>,
        S: Iterator<Item = Id<Literal>>,
    >(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause_storage: &ClauseStorage,
        _literals: T,
        _extension_literals: S,
    ) {
    }

    fn unification(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _bindings: &IdMap<Variable, Option<Id<Term>>>,
    ) {
    }
}

pub(crate) struct Silent;
impl Record for Silent {}
