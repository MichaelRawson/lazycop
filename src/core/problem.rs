use crate::prelude::*;

#[derive(Default)]
pub struct Problem {
    pub clauses: Vec<(Clause, TermGraph)>,
    pub start_clauses: Vec<Id<Clause>>,
    pub predicate_occurrences: [IdMap<Symbol, Vec<Position>>; 2],
    pub symbol_table: SymbolTable,
}

/*
impl Problem {
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
*/
