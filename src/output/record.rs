use crate::output::print::{PrintClause, PrintLiteral};
use crate::prelude::*;

pub trait Record {
    fn start_inference(&mut self, _inference: &'static str) {}
    fn end_inference(&mut self) {}
    fn axiom(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause: &Clause,
    ) {
    }
    fn premise(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause: &Clause,
    ) {
    }
    fn lemma(
        &mut self,
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _literal: &Literal,
    ) {
    }
    fn conclusion(
        &mut self,
        _inference: &'static str,
        _offsets: &[i32],
        _symbol_table: &SymbolTable,
        _term_graph: &TermGraph,
        _clause: &Clause,
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
    fn start_inference(&mut self, inference: &'static str) {
        println!("% {}", inference);
    }

    fn end_inference(&mut self) {
        println!();
    }

    fn axiom(
        &mut self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause: &Clause,
    ) {
        self.clause_number += 1;
        println!(
            "cnf({}, axiom, {}).",
            self.clause_number,
            PrintClause(symbol_table, term_graph, clause)
        );
    }

    fn premise(
        &mut self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause: &Clause,
    ) {
        self.clause_number += 1;
        println!(
            "cnf({}, plain, {}).",
            self.clause_number,
            PrintClause(symbol_table, term_graph, clause)
        );
    }

    fn lemma(
        &mut self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        literal: &Literal,
    ) {
        self.clause_number += 1;
        println!(
            "cnf({}, lemma, {}).",
            self.clause_number,
            PrintLiteral(symbol_table, term_graph, &literal)
        );
    }

    fn conclusion(
        &mut self,
        inference: &'static str,
        offsets: &[i32],
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause: &Clause,
    ) {
        self.clause_number += 1;
        let premise_numbers = offsets
            .iter()
            .map(|offset| self.clause_number as i32 + offset)
            .collect::<Vec<_>>();
        println!(
            "cnf({}, plain, {}, inference({}, {:?})).",
            self.clause_number,
            PrintClause(symbol_table, term_graph, clause),
            inference,
            premise_numbers
        );
    }
}
