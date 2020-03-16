use crate::prelude::*;

pub struct Problem {
    pub symbol_list: SymbolList,
    pub clauses: Vec<(Clause, TermList)>,
}

impl Problem {
    pub fn new(
        symbol_list: SymbolList,
        clauses: Vec<(Clause, TermList)>,
    ) -> Self {
        Self {
            symbol_list,
            clauses,
        }
    }
}
