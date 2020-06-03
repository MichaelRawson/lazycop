use crate::atom::Atom;
use crate::clause::Clause;
use crate::io::{exit, szs, tstp};
use crate::prelude::*;
use crate::record::Record;
use fnv::FnvHashMap;
use std::mem;

pub(crate) struct ProblemClause {
    literals: Literals,
    terms: Terms,
}

pub(crate) struct PredicateOccurrence {
    pub(crate) clause: Id<ProblemClause>,
    pub(crate) literal: Id<Literal>,
}

pub(crate) struct VariableEqualityOccurrence {
    pub(crate) clause: Id<ProblemClause>,
    pub(crate) literal: Id<Literal>,
    pub(crate) from: Id<Term>,
    pub(crate) to: Id<Term>,
}

#[derive(Default)]
pub(crate) struct Problem {
    symbols: Symbols,
    clauses: Block<ProblemClause>,
    start: Vec<Id<ProblemClause>>,
    predicate_occurrences: [Block<Vec<PredicateOccurrence>>; 2],
    variable_equality_occurrences: Vec<VariableEqualityOccurrence>,
}

impl Problem {
    pub(crate) fn signature(&self) -> &Symbols {
        &self.symbols
    }

    pub(crate) fn has_equality(&self) -> bool {
        !self.variable_equality_occurrences.is_empty()
    }

    pub(crate) fn start_clauses(
        &self,
    ) -> impl Iterator<Item = Id<ProblemClause>> + '_ {
        self.start.iter().copied()
    }

    pub(crate) fn get_clause(
        &self,
        id: Id<ProblemClause>,
    ) -> (&Literals, &Terms) {
        let clause = &self.clauses[id];
        (&clause.literals, &clause.terms)
    }

    pub(crate) fn query_predicates(
        &self,
        polarity: bool,
        symbol: Id<Symbol>,
    ) -> impl Iterator<Item = &PredicateOccurrence> + '_ {
        self.predicate_occurrences[polarity as usize][symbol.transmute()]
            .iter()
    }

    pub(crate) fn query_variable_equalities(
        &self,
    ) -> impl Iterator<Item = &VariableEqualityOccurrence> + '_ {
        self.variable_equality_occurrences.iter()
    }
}

type FunctionKey = (Id<Symbol>, Vec<Id<Term>>);
#[derive(Default)]
pub(crate) struct ProblemBuilder {
    problem: Problem,
    positive_clauses: Vec<Id<ProblemClause>>,
    axiom_clauses: Vec<Id<ProblemClause>>,
    conjecture_clauses: Vec<Id<ProblemClause>>,
    terms: Terms,
    symbols: FnvHashMap<(String, u32), Id<Symbol>>,
    variable_map: FnvHashMap<String, Id<Term>>,
    function_map: FnvHashMap<FunctionKey, Id<Term>>,
    saved_terms: Vec<Id<Term>>,
    saved_literals: Literals,
    contains_negative_literal: bool,
}

impl ProblemBuilder {
    pub(crate) fn finish(mut self) -> Problem {
        if self.axiom_clauses.is_empty() || self.conjecture_clauses.is_empty()
        {
            self.problem.start = self.positive_clauses;
        } else {
            self.problem.start = self.conjecture_clauses;
        }
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
            .or_insert_with(|| symbols.add(arity, name));

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
        self.contains_negative_literal =
            self.contains_negative_literal || !polarity;
        let term = self.pop_term();
        let atom = Atom::Predicate(term);
        let symbol = atom.get_predicate_symbol(&self.terms);

        let clause = self.problem.clauses.len();
        let literal = self.saved_literals.len();
        let occurrence = PredicateOccurrence { clause, literal };
        let predicate_positions =
            &mut self.problem.predicate_occurrences[polarity as usize];
        predicate_positions.resize((self.problem.symbols.len()).transmute());
        predicate_positions[symbol.transmute()].push(occurrence);

        self.saved_literals.push(Literal::new(polarity, atom));
    }

    pub(crate) fn equality(&mut self, polarity: bool) {
        self.contains_negative_literal =
            self.contains_negative_literal || !polarity;
        let right = self.pop_term();
        let left = self.pop_term();

        let clause = self.problem.clauses.len();
        let literal = self.saved_literals.len();

        let variable_equality_occurrences =
            &mut self.problem.variable_equality_occurrences;
        let mut add_variable_equality = |from, to| {
            let position = VariableEqualityOccurrence {
                clause,
                literal,
                from,
                to,
            };
            variable_equality_occurrences.push(position);
        };
        if self.terms.is_variable(left) {
            add_variable_equality(left, right);
        }
        if self.terms.is_variable(right) {
            add_variable_equality(right, left);
        }

        let atom = Atom::Equality(left, right);
        self.saved_literals.push(Literal::new(polarity, atom));
    }

    pub(crate) fn clause(&mut self, conjecture: bool) {
        let terms = mem::take(&mut self.terms);
        self.variable_map.clear();
        self.function_map.clear();
        let literals = mem::take(&mut self.saved_literals);

        if literals.is_empty() {
            szs::unsatisfiable();
            szs::begin_refutation();
            let mut record = tstp::TSTP::default();
            record.axiom(
                &self.problem.symbols,
                &terms,
                &literals,
                &Clause::new(Id::default(), Id::default()),
            );
            szs::end_refutation();
            exit::success()
        }

        let problem_clause = ProblemClause { literals, terms };
        let problem_clause = self.problem.clauses.push(problem_clause);
        if conjecture {
            self.conjecture_clauses.push(problem_clause);
        } else {
            self.axiom_clauses.push(problem_clause);
        }

        let is_positive_clause = !self.contains_negative_literal;
        self.contains_negative_literal = false;
        if is_positive_clause {
            self.positive_clauses.push(problem_clause);
        }
    }

    fn pop_term(&mut self) -> Id<Term> {
        self.saved_terms.pop().expect("need a term")
    }
}
