use crate::index::PredicateIndex;
use crate::prelude::*;

pub struct Problem {
    pub symbol_list: SymbolList,
    pub clauses: Vec<(Clause, TermList)>,
    pub start_clauses: Vec<Id<Clause>>,
    pub predicate_index: PredicateIndex,
}

impl Problem {
    pub fn new(
        symbol_list: SymbolList,
        clauses: Vec<(Clause, TermList)>,
        start_clauses: Vec<Id<Clause>>,
        predicate_index: PredicateIndex,
    ) -> Self {
        Self {
            symbol_list,
            clauses,
            start_clauses,
            predicate_index,
        }
    }
}
