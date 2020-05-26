use crate::prelude::*;
use crate::record::Record;
use crate::solver::Solver;

#[derive(Clone, Copy)]
pub(crate) struct Clause {
    start: Id<Literal>,
    end: Id<Literal>,
    current: Id<Literal>,
}

impl Clause {
    pub(crate) fn new(start: Id<Literal>, end: Id<Literal>) -> Self {
        let current = start;
        Self {
            start,
            end,
            current,
        }
    }

    pub(crate) fn is_empty(self) -> bool {
        self.current == self.end
    }

    pub(crate) fn open(self) -> Range<Literal> {
        Range::new(self.current, self.end)
    }

    pub(crate) fn closed(self) -> Range<Literal> {
        Range::new(self.start, self.current)
    }

    pub(crate) fn path(self) -> Range<Literal> {
        Range::new(self.start, self.current + Offset::new(1))
    }

    pub(crate) fn remaining(self) -> Range<Literal> {
        Range::new(self.current + Offset::new(1), self.end)
    }

    pub(crate) fn current_literal(self) -> Id<Literal> {
        self.current
    }

    pub(crate) fn close_literal(&mut self) -> Id<Literal> {
        let result = self.current;
        self.current = self.current + Offset::new(1);
        result
    }

    pub(crate) fn start<R: Record>(
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Block<Literal>,
        solver: &mut Solver,
        start_clause: Id<ProblemClause>,
    ) -> Self {
        Self::copy(record, problem, terms, literals, solver, start_clause)
    }

    pub(crate) fn predicate_reduction<R: Record>(
        &mut self,
        record: &mut R,
        symbols: &Symbols,
        terms: &Terms,
        literals: &Block<Literal>,
        solver: &mut Solver,
        matching: Id<Literal>,
    ) {
        let literal = literals[self.close_literal()];
        let matching = literals[matching];
        let p = literal.atom.get_predicate();
        let q = matching.atom.get_predicate();
        solver.assert_equal(p, q);
        record.predicate_reduction(symbols, terms, literals, &self, p, q);
    }

    pub(crate) fn predicate_extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Block<Literal>,
        solver: &mut Solver,
        clause_id: Id<ProblemClause>,
        matching_id: Id<Literal>,
    ) -> Self {
        let matching_id = matching_id + (literals.len() - Id::default());
        let clause =
            Self::copy(record, problem, terms, literals, solver, clause_id);

        let p = literals[self.current].atom.get_predicate();
        let q = literals[matching_id].atom.get_predicate();
        let disequation = Literal {
            polarity: false,
            atom: Atom::Equality(p, q),
        };
        literals[matching_id] = disequation;

        self.add_strong_connection_constraints(
            solver, terms, literals, clause,
        );
        record.predicate_extension(
            &problem.symbols,
            terms,
            literals,
            &self,
            &clause,
        );
        clause
    }

    pub(crate) fn reflexivity<R: Record>(
        &mut self,
        record: &mut R,
        symbols: &Symbols,
        terms: &Terms,
        literals: &Block<Literal>,
        solver: &mut Solver,
    ) {
        let literal = literals[self.close_literal()];
        let (left, right) = literal.atom.get_equality();
        solver.assert_equal(left, right);
        record.reflexivity(symbols, terms, literals, &self, left, right);
    }

    fn copy<R: Record>(
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Block<Literal>,
        solver: &mut Solver,
        start_clause: Id<ProblemClause>,
    ) -> Self {
        let offset = terms.current_offset();
        let problem_clause = &problem[start_clause];
        terms.extend_from(&problem_clause.terms);

        let start = literals.len();
        literals.extend(problem_clause.literals.as_ref().iter().copied());
        let end = literals.len();
        for id in Range::new(start, end) {
            literals[id].offset(offset);
        }
        let clause = Self::new(start, end);
        clause.add_tautology_constraints(solver, terms, literals);

        record.copy(&problem.symbols, terms, literals, &clause);
        clause
    }

    fn add_strong_connection_constraints(
        &self,
        solver: &mut Solver,
        terms: &Terms,
        literals: &Block<Literal>,
        clause: Clause,
    ) {
        for original in self.open().map(|id| &literals[id]) {
            for new in clause.open().map(|id| &literals[id]) {
                if original.polarity != new.polarity {
                    original
                        .atom
                        .add_disequation_constraints(solver, terms, &new.atom);
                }
            }
        }
    }

    fn add_tautology_constraints(
        &self,
        solver: &mut Solver,
        terms: &Terms,
        literals: &Block<Literal>,
    ) {
        let open = self.open();
        for id in open {
            let literal = literals[id];
            if literal.polarity {
                literal.atom.add_positive_constraints(solver)
            }

            for other in open.skip(1).map(|id| &literals[id]) {
                if literal.polarity != other.polarity {
                    literal.atom.add_disequation_constraints(
                        solver,
                        terms,
                        &other.atom,
                    );
                }
            }
        }
    }
}
