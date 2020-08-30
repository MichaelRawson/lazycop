use crate::index::Index;
use crate::prelude::*;
use fnv::FnvHashMap;
use std::mem;

#[derive(Default)]
pub(crate) struct ProblemBuilder<'a> {
    has_equality: bool,
    symbols: Symbols,
    clauses: Block<ProblemClause>,
    index: Index,

    have_axioms: bool,
    empty_clause: Option<Id<ProblemClause>>,
    negative_clauses: Vec<Id<ProblemClause>>,
    conjecture_clauses: Vec<Id<ProblemClause>>,

    terms: Terms,
    symbol_map: FnvHashMap<(&'a str, u32), Id<Symbol>>,
    variable_map: FnvHashMap<&'a str, Id<Term>>,
    saved_terms: Vec<Id<Term>>,
    saved_literals: Literals,
    clause_negative: bool,
}

impl<'a> ProblemBuilder<'a> {
    pub(crate) fn finish(mut self) -> Problem {
        let start = if let Some(empty) = self.empty_clause {
            vec![empty]
        } else if !self.have_axioms || self.conjecture_clauses.is_empty() {
            self.negative_clauses
        } else {
            self.conjecture_clauses
        };
        self.index.set_signature(&self.symbols);
        Problem::new(
            self.symbols,
            self.clauses,
            start,
            self.index,
            self.has_equality,
        )
    }

    pub(crate) fn variable(&mut self, variable: &'a str) {
        let terms = &mut self.terms;
        let id = *self
            .variable_map
            .entry(variable)
            .or_insert_with(|| terms.add_variable());
        self.saved_terms.push(id);
    }

    pub(crate) fn function(&mut self, name: &'a str, arity: u32) {
        let symbols = &mut self.symbols;
        let symbol = *self
            .symbol_map
            .entry((name, arity))
            .or_insert_with(|| {
                let name = name.into();
                let symbol = Symbol { arity, name };
                symbols.push(symbol)
            });
        let args = self
            .saved_terms
            .split_off(self.saved_terms.len() - (arity as usize));

        self.saved_terms
            .push(self.terms.add_function(symbol, &args));
    }

    pub(crate) fn predicate(&mut self, polarity: bool) {
        self.clause_negative &= !polarity;
        let term = self.pop_term();
        let atom = Atom::Predicate(term);
        let symbol = atom.get_predicate_symbol(&self.terms);
        let clause = self.clauses.len();
        let literal = self.saved_literals.push(Literal::new(polarity, atom));

        self.index.add_predicate_occurrence(
            &self.symbols,
            clause,
            literal,
            polarity,
            symbol,
        );
        for arg in self.terms.arguments(&self.symbols, term) {
            let subterm = self.terms.resolve(arg);
            self.index.add_subterm_occurrences(
                &self.symbols,
                &self.terms,
                clause,
                literal,
                subterm,
            );
        }
    }

    pub(crate) fn equality(&mut self, polarity: bool) {
        self.has_equality = true;
        self.clause_negative &= !polarity;
        let right = self.pop_term();
        let left = self.pop_term();
        let clause = self.clauses.len();
        let atom = Atom::Equality(left, right);
        let literal = self.saved_literals.push(Literal::new(polarity, atom));

        self.index.add_subterm_occurrences(
            &self.symbols,
            &self.terms,
            clause,
            literal,
            left,
        );
        self.index.add_subterm_occurrences(
            &self.symbols,
            &self.terms,
            clause,
            literal,
            right,
        );
        if polarity {
            self.index.add_equality_occurrence(
                &self.symbols,
                &self.terms,
                clause,
                literal,
                left,
                true,
            );
            self.index.add_equality_occurrence(
                &self.symbols,
                &self.terms,
                clause,
                literal,
                right,
                false,
            );
        }
    }

    pub(crate) fn clause(&mut self, is_conjecture: bool) {
        let terms = mem::take(&mut self.terms);
        self.variable_map.clear();
        let literals = mem::take(&mut self.saved_literals);
        let is_empty = literals.is_empty();

        let problem_clause = ProblemClause { literals, terms };
        let problem_clause = self.clauses.push(problem_clause);

        if is_conjecture {
            self.conjecture_clauses.push(problem_clause);
        } else {
            self.have_axioms = true;
        }

        let is_negative_clause = !self.clause_negative;
        self.clause_negative = false;
        if is_negative_clause {
            self.negative_clauses.push(problem_clause);
        }

        if is_empty {
            self.empty_clause = Some(problem_clause);
        }
    }

    fn pop_term(&mut self) -> Id<Term> {
        self.saved_terms.pop().expect("need a term")
    }
}
