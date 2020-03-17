use crate::output::print::PrintClause;
use crate::prelude::*;

pub trait Record {
    fn start(
        &mut self,
        _symbol_list: &SymbolList,
        _term_list: &TermList,
        _start: &Clause,
    ) {
    }
}

pub struct Silent;
impl Record for Silent {}

#[derive(Default)]
pub struct PrintProof {
    clause_number: usize,
}

impl Record for PrintProof {
    fn start(
        &mut self,
        symbol_list: &SymbolList,
        term_list: &TermList,
        start: &Clause,
    ) {
        self.clause_number += 1;
        println!(
            "cnf(c{}, axiom, {}).",
            self.clause_number,
            PrintClause(symbol_list, term_list, start)
        );
    }
}
