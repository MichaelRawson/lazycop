use crate::clausify;
use crate::index::Index;
use crate::prelude::*;

#[derive(Default)]
pub(crate) struct ProblemBuilder {
    has_equality: bool,
    clauses: Block<ProblemClause>,
    index: Index,

    has_axioms: bool,
    has_conjecture: bool,
    empty_clause: Option<Id<ProblemClause>>,
    conjecture_clauses: Vec<Id<ProblemClause>>,

    variables: Vec<(clausify::Variable, Id<Term>)>,
}

impl ProblemBuilder {
    pub(crate) fn finish(mut self, symbols: Symbols) -> Problem {
        let start = if let Some(empty) = self.empty_clause {
            vec![empty]
        } else if self.has_conjecture {
            self.conjecture_clauses
        } else {
            self.clauses.range().into_iter().collect()
        };
        self.index.set_signature(&symbols);
        Problem::new(
            symbols,
            self.clauses,
            start,
            self.index,
            self.has_equality,
        )
    }

    fn term(&mut self, terms: &mut Terms, term: clausify::Term) -> Id<Term> {
        match term {
            clausify::Term::Var(x) => {
                let lookup = self.variables.iter().find(|(v, _)| v == &x);
                if let Some((_, term)) = lookup {
                    *term
                } else {
                    let term = terms.add_variable();
                    self.variables.push((x, term));
                    term
                }
            }
            clausify::Term::Fun(f, args) => {
                let args = args
                    .into_iter()
                    .map(|arg| self.term(terms, arg))
                    .collect();
                terms.add_function(f, args)
            }
        }
    }

    fn literal(
        &mut self,
        symbols: &Symbols,
        terms: &mut Terms,
        literals: &mut Literals,
        literal: clausify::Literal,
    ) {
        let clausify::Literal(polarity, atom) = literal;
        match atom {
            clausify::Atom::Pred(term) => {
                let term = self.term(terms, term);
                let atom = Atom::Predicate(term);
                let symbol = atom.get_predicate_symbol(terms);
                let clause = self.clauses.end();
                let literal = literals.push(Literal::new(polarity, atom));

                self.index.add_predicate_occurrence(
                    symbols, clause, literal, polarity, symbol,
                );
                for arg in terms.arguments(symbols, term) {
                    let subterm = terms.resolve(arg);
                    self.index.add_subterm_occurrences(
                        symbols, terms, clause, literal, subterm,
                    );
                }
            }
            clausify::Atom::Eq(left, right) => {
                self.has_equality = true;
                let left = self.term(terms, left);
                let right = self.term(terms, right);
                let clause = self.clauses.end();

                let atom = Atom::Equality(left, right);
                let literal = literals.push(Literal::new(polarity, atom));

                self.index.add_subterm_occurrences(
                    symbols, terms, clause, literal, left,
                );
                self.index.add_subterm_occurrences(
                    symbols, terms, clause, literal, right,
                );
                if polarity {
                    self.index.add_equality_occurrence(
                        symbols, terms, clause, literal, left, true,
                    );
                    self.index.add_equality_occurrence(
                        symbols, terms, clause, literal, right, false,
                    );
                }
            }
        }
    }

    fn clause(
        &mut self,
        symbols: &Symbols,
        clause: Vec<clausify::Literal>,
    ) -> (Terms, Literals) {
        let mut terms = Terms::default();
        let mut literals = Literals::default();

        for literal in clause {
            self.literal(symbols, &mut terms, &mut literals, literal);
        }
        self.variables.clear();

        (terms, literals)
    }

    pub(crate) fn add_axiom(
        &mut self,
        symbols: &Symbols,
        origin: Origin,
        cnf: clausify::CNF,
    ) {
        let is_conjecture = origin.conjecture;
        let is_empty = cnf.0.is_empty();

        let (terms, literals) = self.clause(symbols, cnf.0);
        let problem_clause = self.clauses.push(ProblemClause {
            literals,
            terms,
            origin,
        });

        if is_empty {
            self.empty_clause = Some(problem_clause);
        }
        else if is_conjecture {
            self.conjecture_clauses.push(problem_clause);
            self.has_conjecture = true;
        } else {
            self.has_axioms = true;
        }
    }
}
