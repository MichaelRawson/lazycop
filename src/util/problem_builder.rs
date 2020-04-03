use crate::prelude::*;
use fnv::FnvHashMap as HashMap;
use std::mem;

type FunctionKey = (Id<Symbol>, Vec<Id<Term>>);

#[derive(Default)]
pub struct ProblemBuilder {
    problem: Problem,
    symbols: HashMap<(String, u32), Id<Symbol>>,
    variable_map: HashMap<String, Id<Term>>,
    function_map: HashMap<FunctionKey, Id<Term>>,
    saved_terms: Vec<Id<Term>>,
    saved_literals: Vec<Literal>,
    term_graph: TermGraph,
}

impl ProblemBuilder {
    pub fn finish(self) -> Problem {
        self.problem
    }

    pub fn variable(&mut self, variable: String) {
        let term_graph = &mut self.term_graph;
        let id = *self
            .variable_map
            .entry(variable)
            .or_insert_with(|| term_graph.add_variable());
        self.saved_terms.push(id);
    }

    pub fn function(&mut self, symbol: String, arity: u32) {
        let symbol_table = &mut self.problem.symbol_table;
        let symbol = *self
            .symbols
            .entry((symbol.clone(), arity))
            .or_insert_with(|| symbol_table.push(symbol, arity));
        let args = self
            .saved_terms
            .split_off(self.saved_terms.len() - (arity as usize));

        let term_graph = &mut self.term_graph;
        let id = *self
            .function_map
            .entry((symbol, args.clone()))
            .or_insert_with(|| term_graph.add_function(symbol, &args));
        self.saved_terms.push(id);
    }

    pub fn predicate(&mut self, polarity: bool) {
        let term = self.saved_terms.pop().unwrap();
        let atom = Atom::Predicate(term);
        let symbol =
            match self.term_graph.view(&self.problem.symbol_table, term) {
                TermView::Function(symbol, _) => symbol,
                _ => unreachable!(),
            };

        let clause_id = self.problem.clauses.len().into();
        let literal_id = self.saved_literals.len().into();
        self.problem.predicate_occurrences[polarity as usize]
            .entry(symbol)
            .push((clause_id, literal_id));
        self.saved_literals.push(Literal::new(polarity, atom));
    }

    pub fn equality(&mut self, polarity: bool) {
        let right = self.saved_terms.pop().unwrap();
        let left = self.saved_terms.pop().unwrap();
        let atom = Atom::Equality(left, right);
        self.saved_literals.push(Literal::new(polarity, atom));
    }

    pub fn clause(&mut self, start_clause: bool) {
        let term_graph = mem::take(&mut self.term_graph);
        self.variable_map.clear();
        self.function_map.clear();
        let literals = mem::take(&mut self.saved_literals);
        if start_clause {
            let id = self.problem.clauses.len().into();
            self.problem.start_clauses.push(id);
        }
        self.problem
            .clauses
            .push((Clause::new(literals), term_graph));
    }
}
