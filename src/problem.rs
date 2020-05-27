use crate::atom::Atom;
use crate::prelude::*;
use fnv::FnvHashMap;
use std::mem;

pub(crate) struct ProblemClause {
    literals: Block<Literal>,
    terms: Terms,
}

#[derive(PartialEq, Eq, Hash)]
struct PredicateQuery {
    polarity: bool,
    symbol: Id<Symbol>,
}

struct Position {
    problem_clause: Id<ProblemClause>,
    literal: Id<Literal>,
}

#[derive(Default)]
pub(crate) struct Problem {
    pub(crate) symbols: Block<Symbol>,
    clauses: Block<ProblemClause>,
    start: Vec<Id<ProblemClause>>,
    predicates: FnvHashMap<PredicateQuery, Vec<Position>>,
}

impl Problem {
    pub(crate) fn start_clauses(
        &self,
    ) -> impl Iterator<Item = Id<ProblemClause>> + '_ {
        self.start.iter().copied()
    }

    pub(crate) fn get_clause(
        &self,
        id: Id<ProblemClause>,
    ) -> (&Block<Literal>, &Terms) {
        let clause = &self.clauses[id];
        (&clause.literals, &clause.terms)
    }

    pub(crate) fn query_predicates(
        &self,
        polarity: bool,
        symbol: Id<Symbol>,
    ) -> impl Iterator<Item = (Id<ProblemClause>, Id<Literal>)> + '_ {
        let key = PredicateQuery { polarity, symbol };
        self.predicates
            .get(&key)
            .map(|x| x.as_slice())
            .unwrap_or_default()
            .iter()
            .map(|position| (position.problem_clause, position.literal))
    }
}

type FunctionKey = (Id<Symbol>, Vec<Id<Term>>);
#[derive(Default)]
pub(crate) struct ProblemBuilder {
    problem: Problem,
    conjecture_clauses: Vec<Id<ProblemClause>>,
    terms: Terms,
    symbols: FnvHashMap<(String, u32), Id<Symbol>>,
    variable_map: FnvHashMap<String, Id<Term>>,
    function_map: FnvHashMap<FunctionKey, Id<Term>>,
    saved_terms: Vec<Id<Term>>,
    saved_literals: Block<Literal>,
}

impl ProblemBuilder {
    pub(crate) fn finish(mut self) -> Problem {
        self.problem.start = self.conjecture_clauses;
        self.problem
    }

    pub(crate) fn variable(&mut self, variable: String) {
        let terms = &mut self.terms;
        let id = *self
            .variable_map
            .entry(variable)
            .or_insert_with(|| terms.add_variable());
        self.saved_terms.push(id);
    }

    pub(crate) fn function(&mut self, name: String, arity: u32) {
        let symbols = &mut self.problem.symbols;
        let symbol = *self
            .symbols
            .entry((name.clone(), arity))
            .or_insert_with(|| symbols.push(Symbol { name }));

        let args = self
            .saved_terms
            .split_off(self.saved_terms.len() - (arity as usize));
        let terms = &mut self.terms;
        let id = *self
            .function_map
            .entry((symbol, args.clone()))
            .or_insert_with(|| terms.add_function(symbol, &args));
        self.saved_terms.push(id);
    }

    pub(crate) fn predicate(&mut self, polarity: bool) {
        let term = self.pop_term();
        let atom = Atom::Predicate(term);
        let symbol = atom.get_predicate_symbol(&self.terms);

        let problem_clause = self.problem.clauses.len();
        let literal = self.saved_literals.len();
        let position = Position { problem_clause, literal };
        let key = PredicateQuery { polarity, symbol };
        self.problem.predicates.entry(key)
            .or_default()
            .push(position);
        self.saved_literals.push(Literal::new(polarity, atom));
    }

    pub(crate) fn equality(&mut self, polarity: bool) {
        let right = self.pop_term();
        let left = self.pop_term();
        let atom = Atom::Equality(left, right);
        self.saved_literals.push(Literal::new(polarity, atom));
    }

    pub(crate) fn clause(&mut self, conjecture: bool) {
        let terms = mem::take(&mut self.terms);
        self.variable_map.clear();
        self.function_map.clear();
        let literals = mem::take(&mut self.saved_literals);
        let problem_clause = ProblemClause { literals, terms };
        let problem_clause = self.problem.clauses.push(problem_clause);
        if conjecture {
            self.conjecture_clauses.push(problem_clause);
        }
    }

    fn pop_term(&mut self) -> Id<Term> {
        self.saved_terms.pop().expect("need a term")
    }
}
