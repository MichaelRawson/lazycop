use crate::index::Index;
use crate::prelude::*;

pub struct Problem {
    pub symbol_list: SymbolList,
    pub clauses: Vec<(Clause, TermList)>,
    pub start_clauses: Vec<Id<Clause>>,
    pub index: Index,
}

impl Problem {
    pub fn new(
        symbol_list: SymbolList,
        clauses: Vec<(Clause, TermList)>,
        start_clauses: Vec<Id<Clause>>,
        index: Index,
    ) -> Self {
        Self {
            symbol_list,
            clauses,
            start_clauses,
            index,
        }
    }

    pub fn start_rules(&self) -> impl Iterator<Item=Rule> + '_ {
        self.start_clauses.iter().copied().map(Rule::Start)
    }
}
