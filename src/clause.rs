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
        start: Start,
    ) -> Self {
        let start =
            Self::copy(record, problem, terms, literals, solver, start.clause);
        record.inference(
            problem.signature(),
            terms,
            literals,
            "start",
            std::iter::empty(),
            None,
            None,
            &[&start],
        );
        start
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn predicate_reduction<R: Record>(
        &mut self,
        record: &mut R,
        symbols: &Symbols,
        terms: &Terms,
        literals: &mut Literals,
        solver: &mut Solver,
        reduction: PredicateReduction,
        is_path: bool,
    ) {
        let p = &literals[self.close_literal()];
        let q = &literals[reduction.literal];
        let pargs = p.get_predicate_arguments(symbols, terms);
        let qargs = q.get_predicate_arguments(symbols, terms);
        let pterms = pargs.map(|p| terms.resolve(p));
        let qterms = qargs.map(|q| terms.resolve(q));
        let assertions = pterms.zip(qterms);

        let (path, lemma) = if is_path {
            (Some(reduction.literal), None)
        } else {
            (None, Some(reduction.literal))
        };
        record.inference(
            symbols,
            terms,
            literals,
            if is_path {
                "predicate_reduction"
            } else {
                "predicate_lemma"
            },
            assertions.clone(),
            path,
            lemma,
            &[self],
        );
        for (s, t) in assertions {
            solver.assert_equal(s, t);
        }
    }

    pub(crate) fn strict_predicate_extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        solver: &mut Solver,
        extension: PredicateExtension,
    ) -> Self {
        let occurrence =
            problem.get_predicate_occurrence(extension.occurrence);
        let term_offset = terms.current_offset();
        let start = literals.len();

        let clause = Self::copy(
            record,
            problem,
            terms,
            literals,
            solver,
            occurrence.clause,
        );
        let p = literals[self.close_literal()];
        self.add_strong_connection_constraints(
            solver, terms, literals, &clause,
        );
        literals.truncate(start);

        let clause_literals = &problem.get_clause(occurrence.clause).literals;
        let mut q = clause_literals[occurrence.literal];
        q.offset(term_offset);

        let pargs = p.get_predicate_arguments(problem.signature(), terms);
        let qargs = q.get_predicate_arguments(problem.signature(), terms);
        for (parg, qarg) in pargs.zip(qargs) {
            let pterm = terms.resolve(parg);
            let qterm = terms.resolve(qarg);
            solver.assert_equal(pterm, qterm);
        }

        for id in clause_literals
            .range()
            .filter(|id| *id != occurrence.literal)
        {
            let mut literal = clause_literals[id];
            literal.offset(term_offset);
            literals.push(literal);
        }
        let end = literals.len();
        let clause = Self::new(start, end);

        record.inference(
            problem.signature(),
            terms,
            literals,
            "strict_predicate_extension",
            pargs
                .map(|arg| terms.resolve(arg))
                .zip(qargs.map(|arg| terms.resolve(arg))),
            None,
            None,
            &[self, &clause],
        );

        clause
    }

    pub(crate) fn lazy_predicate_extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        solver: &mut Solver,
        extension: PredicateExtension,
    ) -> Self {
        let occurrence =
            problem.get_predicate_occurrence(extension.occurrence);
        let term_offset = terms.current_offset();
        let start = literals.len();

        let clause = Self::copy(
            record,
            problem,
            terms,
            literals,
            solver,
            occurrence.clause,
        );
        let p = literals[self.close_literal()];
        self.add_strong_connection_constraints(
            solver, terms, literals, &clause,
        );
        literals.truncate(start);

        let clause_literals = &problem.get_clause(occurrence.clause).literals;
        let mut q = clause_literals[occurrence.literal];
        q.offset(term_offset);

        let pargs = p.get_predicate_arguments(problem.signature(), terms);
        let qargs = q.get_predicate_arguments(problem.signature(), terms);
        let fresh_start = terms.len();
        for (parg, qarg) in pargs.zip(qargs) {
            let pterm = terms.resolve(parg);
            let qterm = terms.resolve(qarg);
            let fresh = terms.add_variable();
            solver.assert_equal(fresh, pterm);
            let atom = Atom::Equality(fresh, qterm);
            let disequation = Literal::new(false, atom);
            literals.push(disequation);
        }
        let fresh_end = terms.len();
        let fresh = Range::new(fresh_start, fresh_end);

        for id in clause_literals
            .range()
            .filter(|id| *id != occurrence.literal)
        {
            let mut literal = clause_literals[id];
            literal.offset(term_offset);
            literals.push(literal);
        }
        let end = literals.len();
        let clause = Self::new(start, end);

        solver.assert_not_equal(p.get_predicate(), q.get_predicate());
        record.inference(
            problem.signature(),
            terms,
            literals,
            "lazy_predicate_extension",
            fresh.zip(pargs.map(|arg| terms.resolve(arg))),
            None,
            None,
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
        extension: EqualityExtension,
    ) -> Self {
        let target = extension.target;
        let occurrence = problem.get_equality_occurrence(extension.occurrence);
        let term_offset = terms.current_offset();
        let from = occurrence.from + term_offset;
        let to = occurrence.to + term_offset;
        let start = literals.len();

        Self::copy(
            record,
            problem,
            terms,
            literals,
            solver,
            occurrence.clause,
        );
        literals.truncate(start);

        let fresh = terms.add_variable();
        let atom = Atom::Equality(fresh, to);
        let disequation = Literal::new(false, atom);
        literals.push(disequation);

        let literal = literals[self.close_literal()].subst(
            problem.signature(),
            terms,
            target,
            fresh,
        );
        literals.push(literal);

        let clause_literals = &problem.get_clause(occurrence.clause).literals;
        for id in clause_literals
            .range()
            .filter(|id| *id != occurrence.literal)
        {
            let mut literal = clause_literals[id];
            literal.offset(term_offset);
            literals.push(literal);
        }
        let end = literals.len();
        let clause = Self::new(start, end);

        solver.assert_equal(from, target);
        //solver.assert_gt(from, fresh);
        record.inference(
            problem.signature(),
            terms,
            literals,
            "equality_extension",
            Some((from, target)),
            None,
            None,
            &[self, &clause],
        );

        clause
    }

    pub(crate) fn function_extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        solver: &mut Solver,
        extension: EqualityExtension,
    ) -> Self {
        let target = extension.target;
        let occurrence = problem.get_equality_occurrence(extension.occurrence);
        let term_offset = terms.current_offset();
        let from = occurrence.from + term_offset;
        let to = occurrence.to + term_offset;
        let start = literals.len();

        Self::copy(
            record,
            problem,
            terms,
            literals,
            solver,
            occurrence.clause,
        );
        literals.truncate(start);

        let placeholder = terms.add_variable();
        let atom = Atom::Equality(placeholder, from);
        let disequation = Literal::new(false, atom);
        literals.push(disequation);

        let fresh = terms.add_variable();
        let atom = Atom::Equality(fresh, to);
        let disequation = Literal::new(false, atom);
        literals.push(disequation);

        let literal = literals[self.close_literal()].subst(
            problem.signature(),
            terms,
            target,
            fresh,
        );
        literals.push(literal);

        let clause_literals = &problem.get_clause(occurrence.clause).literals;
        for id in clause_literals
            .range()
            .filter(|id| *id != occurrence.literal)
        {
            let mut literal = clause_literals[id];
            literal.offset(term_offset);
            literals.push(literal);
        }
        let end = literals.len();
        let clause = Self::new(start, end);

        solver.assert_equal(placeholder, target);
        solver.assert_gt(placeholder, fresh);
        record.inference(
            problem.signature(),
            terms,
            literals,
            "equality_extension",
            Some((placeholder, target)),
            None,
            None,
            &[self, &clause],
        );

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
            Some((left, right)),
            None,
            None,
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
        let start = literals.len();
        let offset = terms.current_offset();

        let clause = problem.get_clause(clause);
        terms.extend(&clause.terms);
        literals.extend(&clause.literals);

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
            if literal.polarity && literal.is_equality() {
                literal.add_reflexivity_constraints(solver);
            }
            for other in open.skip(1).map(|id| &literals[id]) {
                if literal.polarity != other.polarity {
                    literal.add_disequation_constraints(solver, terms, &other);
                }
            }
        }
    }
}
