use crate::prelude::*;
use crate::util::id_map::IdMap;
use std::collections::HashMap;
use std::mem;
use std::ops::Index;

pub(crate) struct ProblemClause {
    pub(crate) literals: Arena<Literal>,
    pub(crate) term_graph: TermGraph,
}

pub(crate) struct Position {
    pub(crate) clause: Id<ProblemClause>,
    pub(crate) literal: Id<Literal>,
}

#[derive(Default)]
pub(crate) struct Problem {
    clauses: Arena<ProblemClause>,
    positions: Arena<Position>,
    pub(crate) start_clauses: Vec<Id<ProblemClause>>,
    pub(crate) predicate_occurrences: [IdMap<Symbol, Vec<Id<Position>>>; 2],
    pub(crate) symbol_table: SymbolTable,
}

impl Index<Id<ProblemClause>> for Problem {
    type Output = ProblemClause;

    fn index(&self, id: Id<ProblemClause>) -> &Self::Output {
        &self.clauses[id]
    }
}

impl Index<Id<Position>> for Problem {
    type Output = Position;

    fn index(&self, id: Id<Position>) -> &Self::Output {
        &self.positions[id]
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
        self.problem.predicate_occurrences[0]
            .ensure_capacity(symbol_table.len());
        self.problem.predicate_occurrences[1]
            .ensure_capacity(symbol_table.len());

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
        let term = self.saved_terms.pop().expect("predicate without a term");
        let atom = Atom::Predicate(term);
        let symbol = match self.term_graph.view(term) {
            TermView::Function(symbol, _) => symbol,
            _ => unreachable!(),
        };

        let clause = self.problem.clauses.len();
        let literal = self.saved_literals.len();
        let position = Position { clause, literal };
        let position = self.problem.positions.push(position);
        self.problem.predicate_occurrences[polarity as usize][symbol]
            .push(position);
        self.saved_literals.push(Literal::new(polarity, atom));
    }

    pub(crate) fn equality(&mut self, polarity: bool) {
        let right = self.saved_terms.pop().expect("equality without term");
        let left = self.saved_terms.pop().expect("equality without term");
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
