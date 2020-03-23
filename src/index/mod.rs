//mod fingerprint;
mod symbol;
use crate::prelude::*;

type PredicateIndex = symbol::Index<(Id<Clause>, Id<Literal>)>;

#[derive(Default)]
pub struct Index {
    predicates: [PredicateIndex; 2],
}

impl Index {
    pub fn add_predicate(
        &mut self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        polarity: bool,
        term: Id<Term>,
        clause_id: Id<Clause>,
        literal_id: Id<Literal>,
    ) {
        let top_symbol = match term_graph.view(symbol_table, term) {
            TermView::Function(f, _) => f,
            _ => unreachable!(),
        };
        self.predicates[polarity as usize]
            .make_entry(top_symbol)
            .push((clause_id, literal_id));
    }

    pub fn query_predicates(
        &self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        polarity: bool,
        term: Id<Term>,
    ) -> &[(Id<Clause>, Id<Literal>)] {
        let symbol = match term_graph.view(symbol_table, term) {
            TermView::Function(f, _) => f,
            _ => unreachable!(),
        };
        self.predicates[polarity as usize].query(symbol)
    }
}
