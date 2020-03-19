//mod fingerprint;
mod top_symbol;
use crate::prelude::*;

type PredicateIndex = top_symbol::Index<Vec<(Id<Clause>, Id<Literal>)>>;

#[derive(Default)]
pub struct Index {
    predicates: [PredicateIndex; 2],
}

impl Index {
    pub fn add_lazy_predicate(
        &mut self,
        symbol_list: &SymbolList,
        term_list: &TermList,
        polarity: bool,
        term: Id<Term>,
        clause_id: Id<Clause>,
        literal_id: Id<Literal>,
    ) {
        let top_symbol = match term_list.view(symbol_list, term) {
            TermView::Function(f, _) => f,
            _ => unreachable!(),
        };
        self.predicates[polarity as usize]
            .make_entry(top_symbol)
            .push((clause_id, literal_id));
    }

    pub fn query_lazy_predicates(
        &self,
        symbol_list: &SymbolList,
        term_list: &TermList,
        polarity: bool,
        term: Id<Term>,
    ) -> impl Iterator<Item = (Id<Clause>, Id<Literal>)> + '_ {
        let top_symbol = match term_list.view(symbol_list, term) {
            TermView::Function(f, _) => f,
            _ => unreachable!(),
        };

        self.predicates[polarity as usize]
            .query(top_symbol)
            .map(|results| &results[..])
            .unwrap_or(&[])
            .iter()
            .copied()
    }
}
