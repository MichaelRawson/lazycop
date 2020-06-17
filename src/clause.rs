use crate::constraint::Constraints;
use crate::prelude::*;
use crate::record::Record;
use std::iter::once;

fn argument_pairs<'terms>(
    symbols: &Symbols,
    terms: &'terms Terms,
    p: &Literal,
    q: &Literal,
) -> impl Iterator<Item = (Id<Term>, Id<Term>)> + Clone + 'terms {
    let pargs = p.get_predicate_arguments(symbols, terms);
    let qargs = q.get_predicate_arguments(symbols, terms);
    let pterms = pargs.map(move |p| terms.resolve(p));
    let qterms = qargs.map(move |q| terms.resolve(q));
    pterms.zip(qterms)
}

#[derive(Clone, Copy)]
pub(crate) struct Clause {
    current: Id<Literal>,
    end: Id<Literal>,
}

impl Clause {
    pub(crate) fn new(start: Id<Literal>, end: Id<Literal>) -> Self {
        let current = start;
        Self { current, end }
    }

    pub(crate) fn is_empty(self) -> bool {
        self.current == self.end
    }

    pub(crate) fn open(self) -> Range<Literal> {
        Range::new(self.current, self.end)
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
        literals: &mut Literals,
        constraints: &mut Constraints,
        start: Start,
    ) -> Self {
        let start = Self::copy(
            record,
            problem,
            terms,
            literals,
            constraints,
            start.clause,
        );
        record.inference(
            problem.signature(),
            terms,
            literals,
            "start",
            None,
            None,
            &[start],
        );
        start
    }

    pub(crate) fn reflexivity<R: Record>(
        &mut self,
        record: &mut R,
        symbols: &Symbols,
        terms: &Terms,
        literals: &Literals,
        constraints: &mut Constraints,
    ) {
        let literal = literals[self.close_literal()];
        let (left, right) = literal.get_equality();
        constraints.assert_eq(left, right);
        record.inference(
            symbols,
            terms,
            literals,
            "reflexivity",
            once((left, right)),
            None,
            &[*self],
        );
    }

    pub(crate) fn predicate_reduction<R: Record>(
        &mut self,
        record: &mut R,
        symbols: &Symbols,
        terms: &Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        reduction: PredicateReduction,
    ) {
        let p = &literals[self.close_literal()];
        let q = &literals[reduction.literal];
        let assertions = argument_pairs(symbols, terms, p, q);

        record.inference(
            symbols,
            terms,
            literals,
            "predicate_reduction",
            assertions.clone(),
            Some(q),
            &[*self],
        );
        for (s, t) in assertions {
            constraints.assert_eq(s, t);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn equality_reduction<R: Record>(
        &mut self,
        record: &mut R,
        symbols: &Symbols,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        reduction: EqualityReduction,
        lr: bool,
    ) -> Self {
        let target = reduction.target;
        let (left, right) = literals[reduction.literal].get_equality();
        let (from, to) = if lr { (left, right) } else { (right, left) };

        let (fresh, fresh_constraint) = if terms.is_variable(to) {
            (to, None)
        } else {
            let fresh = terms.add_variable();
            constraints.assert_eq(to, fresh);
            (fresh, Some((to, fresh)))
        };

        let start = literals.len();
        literals.push(literals[self.current].subst(
            symbols,
            terms,
            constraints,
            target,
            fresh,
        ));
        let end = literals.len();
        let consequence = Self::new(start, end);

        constraints.assert_eq(target, from);
        constraints.assert_gt(from, to);
        record.inference(
            symbols,
            terms,
            literals,
            "equality_reduction",
            once((target, from)).chain(fresh_constraint),
            Some(&literals[reduction.literal]),
            &[self.closed(), consequence],
        );
        consequence
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn subterm_reduction<R: Record>(
        &mut self,
        record: &mut R,
        symbols: &Symbols,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        reduction: EqualityReduction,
        lr: bool,
    ) -> Self {
        let target = reduction.target;
        let (left, right) = literals[self.current].get_equality();
        let (from, to) = if lr { (left, right) } else { (right, left) };

        let (fresh, fresh_constraint) = if terms.is_variable(to) {
            (to, None)
        } else {
            let fresh = terms.add_variable();
            constraints.assert_eq(to, fresh);
            (fresh, Some((to, fresh)))
        };

        let start = literals.len();
        literals.push(literals[reduction.literal].subst(
            symbols,
            terms,
            constraints,
            target,
            fresh,
        ));
        let end = literals.len();
        let consequence = Self::new(start, end);

        constraints.assert_eq(target, from);
        constraints.assert_gt(from, to);
        record.inference(
            symbols,
            terms,
            literals,
            "subterm_reduction",
            once((target, from)).chain(fresh_constraint),
            Some(&literals[reduction.literal]),
            &[self.closed(), consequence],
        );
        consequence
    }

    pub(crate) fn strict_predicate_extension<R: Record>(
        self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: PredicateExtension,
    ) -> Self {
        let mut extension = Self::predicate_extension(
            record,
            problem,
            terms,
            literals,
            constraints,
            extension,
        );
        let p = &literals[self.current];
        let q = &literals[extension.close_literal()];
        let assertions = argument_pairs(problem.signature(), terms, p, q);
        record.inference(
            problem.signature(),
            terms,
            literals,
            "strict_predicate_extension",
            assertions.clone(),
            None,
            &[self.closed(), extension],
        );

        for (parg, qarg) in assertions {
            constraints.assert_eq(parg, qarg);
        }
        extension
    }

    pub(crate) fn lazy_predicate_extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: PredicateExtension,
    ) -> (Self, Self) {
        let extension = Self::predicate_extension(
            record,
            problem,
            terms,
            literals,
            constraints,
            extension,
        );
        let p = &literals[self.current];
        let q = &literals[extension.current];
        let pargs = p.get_predicate_arguments(problem.signature(), terms);
        let qargs = q.get_predicate_arguments(problem.signature(), terms);

        let fresh_start = terms.len();
        for parg in pargs {
            let fresh = terms.add_variable();
            constraints.assert_eq(fresh, terms.resolve(parg));
        }
        let fresh_end = terms.len();
        let fresh = Range::new(fresh_start, fresh_end);

        let disequation_start = literals.len();
        for (fresh, qarg) in fresh.zip(qargs) {
            literals.push(Literal::disequation(terms.resolve(qarg), fresh));
        }
        let disequation_end = literals.len();
        let disequations = Self::new(disequation_start, disequation_end);

        record.inference(
            problem.signature(),
            terms,
            literals,
            "lazy_predicate_extension",
            fresh.zip(pargs.map(|p| terms.resolve(p))),
            None,
            &[self.closed(), extension.closed(), disequations],
        );
        (extension, disequations)
    }

    pub(crate) fn strict_function_extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: EqualityExtension,
    ) -> (Self, Self) {
        let target = extension.target;
        let (extension, from, to) = Self::equality_extension(
            record,
            problem,
            terms,
            literals,
            constraints,
            extension,
        );
        let start = literals.len();
        let fresh = terms.add_variable();
        constraints.assert_eq(target, from);
        constraints.assert_gt(from, fresh);

        literals.push(Literal::disequation(fresh, to));
        literals.push(literals[self.current].subst(
            problem.signature(),
            terms,
            constraints,
            target,
            fresh,
        ));
        let end = literals.len();
        let consequence = Self::new(start, end);

        record.inference(
            problem.signature(),
            terms,
            literals,
            "strict_function_extension",
            once((target, from)),
            None,
            &[self.closed(), extension.closed(), consequence],
        );
        (extension, consequence)
    }

    pub(crate) fn lazy_function_extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: EqualityExtension,
    ) -> (Self, Self) {
        let target = extension.target;
        let (extension, from, to) = Self::equality_extension(
            record,
            problem,
            terms,
            literals,
            constraints,
            extension,
        );
        let start = literals.len();
        let fresh = terms.add_variable();
        let placeholder =
            terms.fresh_function(problem.signature(), terms.symbol(from));
        constraints.assert_eq(target, placeholder);
        constraints.assert_neq(from, target);
        constraints.assert_gt(placeholder, fresh);

        let ss = terms
            .arguments(problem.signature(), placeholder)
            .map(|s| terms.resolve(s));
        let ts = terms
            .arguments(problem.signature(), from)
            .map(|t| terms.resolve(t));
        for (s, t) in ss.zip(ts) {
            literals.push(Literal::disequation(s, t));
        }
        literals.push(Literal::disequation(fresh, to));
        literals.push(literals[self.current].subst(
            problem.signature(),
            terms,
            constraints,
            target,
            fresh,
        ));
        let end = literals.len();
        let consequence = Self::new(start, end);

        record.inference(
            problem.signature(),
            terms,
            literals,
            "lazy_function_extension",
            once((placeholder, target)),
            None,
            &[self.closed(), extension.closed(), consequence],
        );
        (extension, consequence)
    }

    pub(crate) fn variable_extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: EqualityExtension,
    ) -> (Self, Self) {
        let target = extension.target;
        let (extension, from, to) = Self::equality_extension(
            record,
            problem,
            terms,
            literals,
            constraints,
            extension,
        );

        let start = literals.len();
        let fresh = terms.add_variable();
        literals.push(Literal::disequation(to, fresh));
        literals.push(literals[self.current].subst(
            problem.signature(),
            terms,
            constraints,
            target,
            fresh,
        ));
        let end = literals.len();
        let consequence = Self::new(start, end);

        constraints.assert_eq(target, from);
        constraints.assert_gt(from, fresh);
        record.inference(
            problem.signature(),
            terms,
            literals,
            "variable_extension",
            once((target, from)),
            None,
            &[self.closed(), extension.closed(), consequence],
        );

        (extension, consequence)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lazy_subterm_extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: SubtermExtension,
        lr: bool,
    ) -> (Self, Self) {
        let (left, right) = literals[self.current].get_equality();
        let (from, to) = if lr { (left, right) } else { (right, left) };
        let (extension, target) = Self::subterm_extension(
            record,
            problem,
            terms,
            literals,
            constraints,
            extension,
        );
        let (fresh, fresh_constraint) = if terms.is_variable(to) {
            (to, None)
        } else {
            let fresh = terms.add_variable();
            constraints.assert_eq(to, fresh);
            (fresh, Some((to, fresh)))
        };

        let placeholder =
            terms.fresh_function(problem.signature(), terms.symbol(target));
        constraints.assert_eq(placeholder, from);
        constraints.assert_gt(from, to);
        constraints.assert_neq(target, from);

        let start = literals.len();
        let ss = terms
            .arguments(problem.signature(), placeholder)
            .map(|s| terms.resolve(s));
        let ts = terms
            .arguments(problem.signature(), target)
            .map(|t| terms.resolve(t));
        for (s, t) in ss.zip(ts) {
            literals.push(Literal::disequation(s, t));
        }
        literals.push(literals[extension.current].subst(
            problem.signature(),
            terms,
            constraints,
            target,
            fresh,
        ));
        let end = literals.len();
        let consequence = Self::new(start, end);

        record.inference(
            problem.signature(),
            terms,
            literals,
            "lazy_subterm_extension",
            once((placeholder, from)).chain(fresh_constraint),
            None,
            &[self.closed(), extension.closed(), consequence],
        );
        (extension, consequence)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn strict_subterm_extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: SubtermExtension,
        lr: bool,
    ) -> (Self, Self) {
        let (left, right) = literals[self.current].get_equality();
        let (from, to) = if lr { (left, right) } else { (right, left) };
        let (extension, target) = Self::subterm_extension(
            record,
            problem,
            terms,
            literals,
            constraints,
            extension,
        );
        let (fresh, fresh_constraint) = if terms.is_variable(to) {
            (to, None)
        } else {
            let fresh = terms.add_variable();
            constraints.assert_eq(to, fresh);
            (fresh, Some((to, fresh)))
        };

        constraints.assert_eq(target, from);
        constraints.assert_gt(from, to);

        let start = literals.len();
        literals.push(literals[extension.current].subst(
            problem.signature(),
            terms,
            constraints,
            target,
            fresh,
        ));
        let end = literals.len();
        let consequence = Self::new(start, end);

        record.inference(
            problem.signature(),
            terms,
            literals,
            "strict_subterm_extension",
            once((target, from)).chain(fresh_constraint),
            None,
            &[self.closed(), extension.closed(), consequence],
        );
        (extension, consequence)
    }

    fn predicate_extension<R: Record>(
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: PredicateExtension,
    ) -> Self {
        let occurrence =
            problem.get_predicate_occurrence(extension.occurrence);
        Self::extension(
            record,
            problem,
            terms,
            literals,
            constraints,
            occurrence.clause,
            occurrence.literal,
        )
    }

    fn equality_extension<R: Record>(
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: EqualityExtension,
    ) -> (Self, Id<Term>, Id<Term>) {
        let occurrence = problem.get_equality_occurrence(extension.occurrence);
        let extension = Self::extension(
            record,
            problem,
            terms,
            literals,
            constraints,
            occurrence.clause,
            occurrence.literal,
        );

        let (left, right) = literals[extension.current].get_equality();
        let (from, to) = if occurrence.lr {
            (left, right)
        } else {
            (right, left)
        };
        (extension, from, to)
    }

    fn subterm_extension<R: Record>(
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: SubtermExtension,
    ) -> (Self, Id<Term>) {
        let occurrence = problem.get_subterm_occurrence(extension.occurrence);
        let target = occurrence.subterm + terms.offset();
        let extension = Self::extension(
            record,
            problem,
            terms,
            literals,
            constraints,
            occurrence.clause,
            occurrence.literal,
        );
        (extension, target)
    }

    #[allow(clippy::too_many_arguments)]
    fn extension<R: Record>(
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        clause: Id<ProblemClause>,
        literal: Id<Literal>,
    ) -> Self {
        let literal_offset = literals.offset();
        let extension =
            Self::copy(record, problem, terms, literals, constraints, clause);

        let matching = literal + literal_offset;
        let mate = literals[matching];
        for index in Range::new(extension.current, matching).rev() {
            literals[index + Offset::new(1)] = literals[index];
        }
        literals[extension.current] = mate;
        extension
    }

    fn copy<R: Record>(
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        clause: Id<ProblemClause>,
    ) -> Self {
        let start = literals.len();
        let offset = terms.offset();

        let clause = problem.get_clause(clause);
        terms.extend(&clause.terms);
        literals.extend(&clause.literals);

        let end = literals.len();
        for id in Range::new(start, end) {
            literals[id].offset(offset);
        }
        let clause = Self::new(start, end);
        clause.add_tautology_constraints(constraints, terms, literals);

        record.axiom(problem.signature(), terms, literals, clause);
        clause
    }

    fn add_tautology_constraints(
        self,
        constraints: &mut Constraints,
        terms: &Terms,
        literals: &Literals,
    ) {
        let open = self.open();
        for id in open {
            let literal = literals[id];
            if literal.polarity && literal.is_equality() {
                literal.add_reflexivity_constraints(constraints);
            }
            for other in open.skip(1).map(|id| &literals[id]) {
                if literal.polarity != other.polarity {
                    literal.add_disequation_constraints(
                        constraints,
                        terms,
                        &other,
                    );
                }
            }
        }
    }

    fn closed(self) -> Self {
        let current = self.current + Offset::new(1);
        let end = self.end;
        Self { current, end }
    }
}
