use crate::atom::Atom;
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
        literals: &mut Literals,
        solver: &mut Solver,
        start_clause: Id<ProblemClause>,
    ) -> Self {
        let start =
            Self::copy(record, problem, terms, literals, solver, start_clause);
        record.inference(
            problem.signature(),
            terms,
            literals,
            "start",
            &[],
            &[&start],
        );
        start
    }

    pub(crate) fn predicate_reduction<R: Record>(
        &mut self,
        record: &mut R,
        symbols: &Symbols,
        terms: &Terms,
        literals: &mut Literals,
        solver: &mut Solver,
        q: Id<Term>,
    ) {
        let p = literals[self.close_literal()].get_predicate();
        solver.assert_equal(p, q);
        record.inference(
            symbols,
            terms,
            literals,
            "predicate_reduction",
            &[(p, q)],
            &[self],
        );
    }

    pub(crate) fn predicate_extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        solver: &mut Solver,
        problem_clause: Id<ProblemClause>,
        problem_literal: Id<Literal>,
    ) -> Self {
        let literal_offset = literals.len() - Id::default();
        let matching_literal = problem_literal + literal_offset;
        let clause = Self::copy(
            record,
            problem,
            terms,
            literals,
            solver,
            problem_clause,
        );

        let p = literals[self.close_literal()].get_predicate();
        let q = literals[matching_literal].get_predicate();
        let disequation = Literal::new(false, Atom::Equality(p, q));
        literals[matching_literal] = disequation;
        literals.swap(clause.current_literal(), matching_literal);
        self.add_strong_connection_constraints(
            solver, terms, literals, &clause,
        );

        record.inference(
            problem.signature(),
            terms,
            literals,
            "predicate_extension",
            &[],
            &[self, &clause],
        );

        clause
    }

    pub(crate) fn variable_extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        solver: &mut Solver,
        problem_clause: Id<ProblemClause>,
        problem_literal: Id<Literal>,
        target: Id<Term>,
        from: Id<Term>,
        to: Id<Term>,
    ) -> Self {
        let literal_offset = literals.len() - Id::default();
        let equation_literal = problem_literal + literal_offset;
        let clause = Self::copy(
            record,
            problem,
            terms,
            literals,
            solver,
            problem_clause,
        );

        let fresh = terms.add_variable();
        let freshened = literals[self.close_literal()].subst(
            problem.signature(),
            terms,
            target,
            fresh,
        );
        literals[equation_literal] = freshened;
        /*
        let q = literals[matching_literal].get_predicate();
        let disequation = Literal::new(false, Atom::Equality(p, q));
        literals[matching_literal] = disequation;
        literals.swap(clause.current_literal(), matching_literal);

        self.add_strong_connection_constraints(
            solver, terms, literals, &clause,
        );
        record.predicate_extension(
            &problem.signature(),
            terms,
            literals,
            &self,
            &clause,
        );
        */
        clause
    }

    pub(crate) fn reflexivity<R: Record>(
        &mut self,
        record: &mut R,
        symbols: &Symbols,
        terms: &Terms,
        literals: &Literals,
        solver: &mut Solver,
    ) {
        let literal = literals[self.close_literal()];
        let (left, right) = literal.get_equality();
        solver.assert_equal(left, right);
        record.inference(
            symbols,
            terms,
            literals,
            "reflexivity",
            &[(left, right)],
            &[self],
        );
    }

    fn copy<R: Record>(
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        solver: &mut Solver,
        clause: Id<ProblemClause>,
    ) -> Self {
        let offset = terms.current_offset();
        let (clause_literals, clause_terms) = problem.get_clause(clause);
        terms.extend_from(clause_terms);

        let start = literals.len();
        literals.extend(clause_literals.as_ref().iter().copied());
        let end = literals.len();
        for id in Range::new(start, end) {
            literals[id].offset(offset);
        }
        let clause = Self::new(start, end);
        clause.add_tautology_constraints(solver, terms, literals);

        record.axiom(problem.signature(), terms, literals, &clause);
        clause
    }

    fn add_strong_connection_constraints(
        &self,
        solver: &mut Solver,
        terms: &Terms,
        literals: &Literals,
        new: &Self,
    ) {
        for original in self.open().map(|id| &literals[id]) {
            for new in new.open().map(|id| &literals[id]) {
                if original.polarity != new.polarity {
                    original.add_disequation_constraints(solver, terms, &new);
                }
            }
        }
    }

    fn add_tautology_constraints(
        &self,
        solver: &mut Solver,
        terms: &Terms,
        literals: &Literals,
    ) {
        let open = self.open();
        for id in open {
            let literal = literals[id];
            literal.add_unit_tautology_constraints(solver);
            for other in open.skip(1).map(|id| &literals[id]) {
                if literal.polarity != other.polarity {
                    literal.add_disequation_constraints(solver, terms, &other);
                }
            }
        }
    }
}
