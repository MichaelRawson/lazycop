use crate::prelude::*;
use crate::util::id_map::IdMap;
use std::collections::HashMap;
use std::mem;

pub(crate) struct ProblemClause {
    literals: Arena<Literal>,
    term_graph: TermGraph,
}

pub(crate) type Position = (Id<ProblemClause>, Id<Literal>);

#[derive(Default)]
pub(crate) struct Problem {
    clauses: Arena<ProblemClause>,
    pub(crate) start_clauses: Vec<Id<ProblemClause>>,
    pub(crate) predicate_occurrences: [IdMap<Symbol, Vec<Position>>; 2],
    pub(crate) symbol_table: SymbolTable,
}

impl Problem {
    pub(crate) fn clause_data(
        &self,
        id: Id<ProblemClause>,
    ) -> (&Arena<Literal>, &TermGraph) {
        let ProblemClause {
            literals,
            term_graph,
        } = &self.clauses[id];
        (literals, term_graph)
    }
}

type FunctionKey = (Id<Symbol>, Vec<Id<Term>>);

#[derive(Default)]
pub(crate) struct ProblemBuilder {
    problem: Problem,
    symbols: HashMap<(String, u32), Id<Symbol>>,
    variable_map: HashMap<String, Id<Term>>,
    function_map: HashMap<FunctionKey, Id<Term>>,
    saved_terms: Vec<Id<Term>>,
    saved_literals: Arena<Literal>,
    term_graph: TermGraph,
}

impl ProblemBuilder {
    pub(crate) fn finish(self) -> Problem {
        self.problem
    }

    pub(crate) fn variable(&mut self, variable: String) {
        let term_graph = &mut self.term_graph;
        let id = *self
            .variable_map
            .entry(variable)
            .or_insert_with(|| term_graph.add_variable());
        self.saved_terms.push(id);
    }

    pub(crate) fn function(&mut self, symbol: String, arity: u32) {
        let symbol_table = &mut self.problem.symbol_table;
        let symbol = *self
            .symbols
            .entry((symbol.clone(), arity))
            .or_insert_with(|| symbol_table.append(symbol));
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

    pub(crate) fn predicate(&mut self, polarity: bool) {
        let term = self.saved_terms.pop().expect("predicate without a term?");
        let atom = Atom::Predicate(term);
        let symbol = match self.term_graph.view(term) {
            TermView::Function(symbol, _) => symbol,
            _ => unreachable!(),
        };

        let clause_id = self.problem.clauses.len();
        let literal_id = self.saved_literals.len();
        self.problem.predicate_occurrences[polarity as usize]
            .get_mut_default(symbol)
            .push((clause_id, literal_id));
        self.saved_literals.push(Literal::new(polarity, atom));
    }

    pub(crate) fn equality(&mut self, polarity: bool) {
        let right = self.saved_terms.pop().expect("equality without term?");
        let left = self.saved_terms.pop().expect("equality without term?");
        let atom = Atom::Equality(left, right);
        self.saved_literals.push(Literal::new(polarity, atom));
    }

    pub(crate) fn clause(&mut self, start_clause: bool) {
        let term_graph = mem::take(&mut self.term_graph);
        self.variable_map.clear();
        self.function_map.clear();
        let literals = mem::take(&mut self.saved_literals);
        let problem_clause = ProblemClause {
            literals,
            term_graph,
        };
        let id = self.problem.clauses.push(problem_clause);
        if start_clause {
            self.problem.start_clauses.push(id);
        }
    }
}
