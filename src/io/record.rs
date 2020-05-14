use crate::prelude::*;
use crate::util::id_map::IdMap;

pub(crate) trait Record {
    fn start(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause_storage: &ClauseStorage,
        _literals: Range<Literal>,
    ) {
    }

    fn reduction(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause_storage: &ClauseStorage,
        _literals: Range<Literal>,
        _left: Id<Term>,
        _right: Id<Term>,
    ) {
    }

    fn extension(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause_storage: &ClauseStorage,
        _literals: Range<Literal>,
        _extension_literals: Range<Literal>,
        _left: Id<Term>,
        _right: Id<Term>,
    ) {
    }

    fn lemma(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause_storage: &ClauseStorage,
        _literals: Range<Literal>,
        _left: Id<Term>,
        _right: Id<Term>,
    ) {
    }

    fn lazy_extension(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause_storage: &ClauseStorage,
        _literals: Range<Literal>,
        _extension_literals: Range<Literal>,
    ) {
    }

    fn reflexivity(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause_storage: &ClauseStorage,
        _literals: Range<Literal>,
        _left: Id<Term>,
        _right: Id<Term>,
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
