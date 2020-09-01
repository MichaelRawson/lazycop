use crate::cnf;
use crate::index::Index;
use crate::prelude::*;

#[derive(Default)]
pub(crate) struct ProblemBuilder {
    has_equality: bool,
    clauses: Block<ProblemClause>,
    index: Index,

    have_axioms: bool,
    empty_clause: Option<Id<ProblemClause>>,
    negative_clauses: Vec<Id<ProblemClause>>,
    conjecture_clauses: Vec<Id<ProblemClause>>,

    variables: Vec<(cnf::Variable, Id<Term>)>,
}

impl ProblemBuilder {
    pub(crate) fn finish(mut self, symbols: Symbols) -> Problem {
        let start = if let Some(empty) = self.empty_clause {
            vec![empty]
        } else if !self.have_axioms || self.conjecture_clauses.is_empty() {
            self.negative_clauses
        } else {
            self.conjecture_clauses
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

    fn term(&mut self, terms: &mut Terms, term: cnf::Term) -> Id<Term> {
        match term {
            cnf::Term::Var(x) => {
                let lookup = self.variables.iter().find(|(v, _)| v == &x);
                if let Some((_, term)) = lookup {
                    *term
                } else {
                    let term = terms.add_variable();
                    self.variables.push((x, term));
                    term
                }
            }
            cnf::Term::Fun(f, args) => {
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
        literal: cnf::Literal,
    ) {
        let cnf::Literal(polarity, atom) = literal;
        match atom {
            cnf::Atom::Pred(term) => {
                let term = self.term(terms, term);
                let atom = Atom::Predicate(term);
                let symbol = atom.get_predicate_symbol(terms);
                let clause = self.clauses.len();
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
            cnf::Atom::Eq(left, right) => {
                self.has_equality = true;
                let left = self.term(terms, left);
                let right = self.term(terms, right);
                let clause = self.clauses.len();

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
        clause: Vec<cnf::Literal>,
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
        cnf: cnf::CNF,
    ) {
        let is_conjecture = origin.conjecture;
        let is_empty = cnf.0.is_empty();
        let is_negative = cnf.0.iter().all(|literal| !literal.0);

        let (terms, literals) = self.clause(symbols, cnf.0);
        let problem_clause = self.clauses.push(ProblemClause {
            literals,
            terms,
            origin,
        });

        if is_conjecture {
            self.conjecture_clauses.push(problem_clause);
        } else {
            self.have_axioms = true;
        }
        if is_negative {
            self.negative_clauses.push(problem_clause);
        }
        if is_empty {
            self.empty_clause = Some(problem_clause);
        }
    }
}
