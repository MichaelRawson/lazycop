use crate::atom::Atom;
use crate::clause::Clause;
use crate::io::{exit, szs, tstp};
use crate::prelude::*;
use crate::record::Record;
use fnv::FnvHashMap;
use std::mem;

pub struct ProblemClause {
    pub literals: Literals,
    pub terms: Terms,
}

pub struct PredicateOccurrence {
    pub clause: Id<ProblemClause>,
    pub literal: Id<Literal>,
}

pub struct EqualityOccurrence {
    pub clause: Id<ProblemClause>,
    pub literal: Id<Literal>,
    pub lr: bool,
}

pub struct SubtermOccurrence {
    pub clause: Id<ProblemClause>,
    pub literal: Id<Literal>,
    pub subterm: Id<Term>,
}

#[derive(Default)]
pub struct Problem {
    symbols: Symbols,
    equalities_present: bool,
    clauses: Block<ProblemClause>,
    start: Vec<Id<ProblemClause>>,

    predicate_occurrences: Block<PredicateOccurrence>,
    equality_occurrences: Block<EqualityOccurrence>,
    subterm_occurrences: Block<SubtermOccurrence>,
    predicates: [LUT<Symbol, Vec<Id<PredicateOccurrence>>>; 2],
    variable_equalities: Vec<Id<EqualityOccurrence>>,
    function_equalities: LUT<Symbol, Vec<Id<EqualityOccurrence>>>,
    symbol_subterms: LUT<Symbol, Vec<Id<SubtermOccurrence>>>,
}

impl Problem {
    pub fn signature(&self) -> &Symbols {
        &self.symbols
    }

    pub fn num_clauses(&self) -> usize {
        self.clauses.len().as_usize()
    }

    pub fn num_start_clauses(&self) -> usize {
        self.start.len()
    }

    pub fn has_equality(&self) -> bool {
        self.equalities_present
    }

    pub fn start_clauses(
        &self,
    ) -> impl Iterator<Item = Id<ProblemClause>> + '_ {
        self.start.iter().copied()
    }

    pub fn get_clause(&self, id: Id<ProblemClause>) -> &ProblemClause {
        &self.clauses[id]
    }

    pub fn get_predicate_occurrence(
        &self,
        id: Id<PredicateOccurrence>,
    ) -> &PredicateOccurrence {
        &self.predicate_occurrences[id]
    }

    pub fn get_equality_occurrence(
        &self,
        id: Id<EqualityOccurrence>,
    ) -> &EqualityOccurrence {
        &self.equality_occurrences[id]
    }

    pub fn get_subterm_occurrence(
        &self,
        id: Id<SubtermOccurrence>,
    ) -> &SubtermOccurrence {
        &self.subterm_occurrences[id]
    }

    pub fn query_predicates(
        &self,
        polarity: bool,
        symbol: Id<Symbol>,
    ) -> impl Iterator<Item = Id<PredicateOccurrence>> + '_ {
        self.predicates[polarity as usize][symbol].iter().copied()
    }

    pub fn query_variable_equalities(
        &self,
    ) -> impl Iterator<Item = Id<EqualityOccurrence>> + '_ {
        self.variable_equalities.iter().copied()
    }

    pub fn query_function_equalities(
        &self,
        symbol: Id<Symbol>,
    ) -> impl Iterator<Item = Id<EqualityOccurrence>> + '_ {
        self.function_equalities[symbol].iter().copied()
    }

    pub fn query_all_subterms(
        &self,
    ) -> impl Iterator<Item = Id<SubtermOccurrence>> + '_ {
        self.symbol_subterms
            .range()
            .flat_map(move |id| self.symbol_subterms[id].iter().copied())
    }

    pub fn query_subterms(
        &self,
        symbol: Id<Symbol>,
    ) -> impl Iterator<Item = Id<SubtermOccurrence>> + '_ {
        self.symbol_subterms[symbol].iter().copied()
    }
}

#[derive(Default)]
pub struct ProblemBuilder {
    problem: Problem,
    negative_clauses: Vec<Id<ProblemClause>>,
    axiom_clauses: Vec<Id<ProblemClause>>,
    conjecture_clauses: Vec<Id<ProblemClause>>,
    terms: Terms,
    symbols: FnvHashMap<(String, u32), Id<Symbol>>,
    variable_map: FnvHashMap<String, Id<Term>>,
    saved_terms: Vec<Id<Term>>,
    saved_literals: Literals,
    contains_positive_literal: bool,
}

impl ProblemBuilder {
    pub fn finish(mut self) -> Problem {
        let max_symbol = self.problem.signature().len();
        self.problem.predicates[0].resize(max_symbol);
        self.problem.predicates[1].resize(max_symbol);
        self.problem.function_equalities.resize(max_symbol);

        if self.axiom_clauses.is_empty() || self.conjecture_clauses.is_empty()
        {
            self.problem.start = self.negative_clauses;
        } else {
            self.problem.start = self.conjecture_clauses;
        }
        self.problem
    }

    pub fn variable(&mut self, variable: String) {
        let terms = &mut self.terms;
        let id = *self
            .variable_map
            .entry(variable)
            .or_insert_with(|| terms.add_variable());
        self.saved_terms.push(id);
    }

    pub fn function(&mut self, name: String, arity: u32) {
        let symbols = &mut self.problem.symbols;
        let symbol = *self
            .symbols
            .entry((name.clone(), arity))
            .or_insert_with(|| {
                let symbol = Symbol { arity, name };
                symbols.push(symbol)
            });
        let args = self
            .saved_terms
            .split_off(self.saved_terms.len() - (arity as usize));

        self.saved_terms
            .push(self.terms.add_function(symbol, &args));
    }

    pub fn predicate(&mut self, polarity: bool) {
        self.contains_positive_literal =
            self.contains_positive_literal || polarity;
        let term = self.pop_term();
        let atom = Atom::Predicate(term);
        let symbol = atom.get_predicate_symbol(&self.terms);
        let clause = self.problem.clauses.len();
        let literal = self.saved_literals.len();

        for arg in self.terms.arguments(&self.problem.symbols, term) {
            let subterm = self.terms.resolve(arg);
            self.add_subterm_occurrences(clause, literal, subterm);
        }

        let occurrence = self
            .problem
            .predicate_occurrences
            .push(PredicateOccurrence { clause, literal });
        let polarity_positions =
            &mut self.problem.predicates[polarity as usize];
        polarity_positions.resize(self.problem.symbols.len());
        polarity_positions[symbol].push(occurrence);
        self.saved_literals.push(Literal::new(polarity, atom));
    }

    pub fn equality(&mut self, polarity: bool) {
        self.problem.equalities_present = true;
        self.contains_positive_literal =
            self.contains_positive_literal || polarity;
        let right = self.pop_term();
        let left = self.pop_term();

        let clause = self.problem.clauses.len();
        let literal = self.saved_literals.len();

        if polarity {
            self.add_equality_occurrence(clause, literal, left, true);
            self.add_equality_occurrence(clause, literal, right, false);
        }
        self.add_subterm_occurrences(clause, literal, left);
        self.add_subterm_occurrences(clause, literal, right);

        let atom = Atom::Equality(left, right);
        self.saved_literals.push(Literal::new(polarity, atom));
    }

    pub fn clause(&mut self, conjecture: bool) {
        let terms = mem::take(&mut self.terms);
        self.variable_map.clear();
        let literals = mem::take(&mut self.saved_literals);

        if literals.is_empty() {
            szs::unsatisfiable();
            szs::begin_incomplete_proof();
            let mut record = tstp::TSTP::default();
            record.axiom(
                &self.problem.symbols,
                &terms,
                &literals,
                Clause::new(Id::default(), Id::default()),
            );
            szs::end_incomplete_proof();
            exit::success()
        }

        let problem_clause = ProblemClause { literals, terms };
        let problem_clause = self.problem.clauses.push(problem_clause);
        if conjecture {
            self.conjecture_clauses.push(problem_clause);
        } else {
            self.axiom_clauses.push(problem_clause);
        }

        let is_negative_clause = !self.contains_positive_literal;
        self.contains_positive_literal = false;
        if is_negative_clause {
            self.negative_clauses.push(problem_clause);
        }
    }

    fn add_equality_occurrence(
        &mut self,
        clause: Id<ProblemClause>,
        literal: Id<Literal>,
        from: Id<Term>,
        lr: bool,
    ) {
        let occurrence =
            self.problem.equality_occurrences.push(EqualityOccurrence {
                clause,
                literal,
                lr,
            });
        match self.terms.view(&self.problem.symbols, from) {
            TermView::Variable(_) => {
                self.problem.variable_equalities.push(occurrence);
            }
            TermView::Function(f, _) => {
                self.problem
                    .function_equalities
                    .resize(self.problem.symbols.len());
                self.problem.function_equalities[f].push(occurrence);
            }
        }
    }

    fn add_subterm_occurrences(
        &mut self,
        clause: Id<ProblemClause>,
        literal: Id<Literal>,
        term: Id<Term>,
    ) {
        let terms = &self.terms;
        let symbols = &self.problem.symbols;
        let subterm_occurrences = &mut self.problem.subterm_occurrences;
        let symbol_subterms = &mut self.problem.symbol_subterms;
        terms.subterms(symbols, term, &mut |subterm| {
            let symbol = terms.symbol(subterm);
            let occurrence = subterm_occurrences.push(SubtermOccurrence {
                clause,
                literal,
                subterm,
            });
            symbol_subterms.resize(symbols.len());
            symbol_subterms[symbol].push(occurrence);
        });
    }

    fn pop_term(&mut self) -> Id<Term> {
        self.saved_terms.pop().expect("need a term")
    }
}
