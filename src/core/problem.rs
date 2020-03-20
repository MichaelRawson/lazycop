use crate::index::Index;
use crate::prelude::*;

pub struct Problem {
    pub symbol_table: SymbolTable,
    pub clauses: Vec<(Clause, TermGraph)>,
    pub start_clauses: Vec<Id<Clause>>,
    pub index: Index,
}

impl Problem {
    pub fn new(
        symbol_table: SymbolTable,
        clauses: Vec<(Clause, TermGraph)>,
        start_clauses: Vec<Id<Clause>>,
        index: Index,
    ) -> Self {
        Self {
            symbol_table,
            clauses,
            start_clauses,
            index,
        }
    }

    pub fn start_rules(&self) -> impl Iterator<Item = Rule> + '_ {
        self.start_clauses.iter().copied().map(Rule::Start)
    }

    pub fn copy_clause_into(
        &self,
        term_graph: &mut TermGraph,
        id: Id<Clause>,
    ) -> Clause {
        let (clause, clause_term_graph) = &self.clauses[id.index()];
        let mut clause = clause.clone();
        clause.offset(term_graph.current_offset());
        term_graph.copy_from(clause_term_graph);
        clause
    }
}
