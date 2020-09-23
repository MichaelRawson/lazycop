use crate::constraint::Constraints;
use crate::prelude::*;
use crate::record::{Inference, Record};
use crate::rule::*;

fn term_argument_pairs<'terms>(
    symbols: &Symbols,
    terms: &'terms Terms,
    s: Id<Term>,
    t: Id<Term>,
) -> impl Iterator<Item = (Id<Term>, Id<Term>)> + 'terms {
    let sargs = terms.arguments(symbols, s);
    let targs = terms.arguments(symbols, t);
    let ss = sargs.into_iter().map(move |sarg| terms.resolve(sarg));
    let ts = targs.into_iter().map(move |targ| terms.resolve(targ));
    ss.zip(ts)
}

fn predicate_argument_pairs<'terms>(
    symbols: &Symbols,
    terms: &'terms Terms,
    p: &Literal,
    q: &Literal,
) -> impl Iterator<Item = (Id<Term>, Id<Term>)> + 'terms {
    let pargs = p.get_predicate_arguments(symbols, terms);
    let qargs = q.get_predicate_arguments(symbols, terms);
    let ss = pargs.into_iter().map(move |parg| terms.resolve(parg));
    let ts = qargs.into_iter().map(move |qarg| terms.resolve(qarg));
    ss.zip(ts)
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
        let axiom =
            Self::copy(problem, terms, literals, constraints, start.clause);
        record.inference(
            problem,
            terms,
            literals,
            R::Inference::new("start").axiom(start.clause, axiom.open()),
        );
        axiom
    }

    pub(crate) fn reflexivity<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &Terms,
        literals: &Literals,
        constraints: &mut Constraints,
    ) {
        let literal = literals[self.close_literal()];
        let (left, right) = literal.get_equality();
        constraints.assert_eq(left, right);
        record.inference(
            problem,
            terms,
            literals,
            R::Inference::new("reflexivity")
                .equation(left, right)
                .deduction(self.open()),
        );
    }

    pub(crate) fn reduction<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        reduction: Reduction,
    ) {
        let mut inference = R::Inference::new("reduction");
        let p = &literals[self.close_literal()];
        let q = &literals[reduction.literal];
        for (s, t) in predicate_argument_pairs(&problem.symbols, terms, p, q) {
            constraints.assert_eq(s, t);
            inference.equation(s, t);
        }

        record.inference(
            problem,
            terms,
            literals,
            inference.lemma(reduction.literal).deduction(self.open()),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_demodulation<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        demodulation: Demodulation,
        l2r: bool,
    ) -> Self {
        self.demodulation(
            record,
            problem,
            terms,
            literals,
            constraints,
            demodulation,
            true,
            l2r,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn backward_demodulation<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        demodulation: Demodulation,
        l2r: bool,
    ) -> Self {
        self.demodulation(
            record,
            problem,
            terms,
            literals,
            constraints,
            demodulation,
            false,
            l2r,
        )
    }

    fn extension(
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: Extension,
    ) -> (Id<ProblemClause>, Self) {
        let occurrence =
            &problem.index.predicate_occurrences[extension.occurrence];
        let extension = Self::extend(
            problem,
            terms,
            literals,
            constraints,
            occurrence.clause,
            occurrence.literal,
        );
        (occurrence.clause, extension)
    }

    pub(crate) fn strict_extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: Extension,
    ) -> Self {
        let (problem_clause, mut extension) =
            Self::extension(problem, terms, literals, constraints, extension);
        let mut inference = R::Inference::new("strict_extension");
        inference.axiom(problem_clause, extension.open());
        let p = &literals[self.current];
        let q = &literals[extension.close_literal()];
        for (s, t) in predicate_argument_pairs(&problem.symbols, terms, p, q) {
            constraints.assert_eq(s, t);
            inference.equation(s, t);
        }

        record.inference(
            problem,
            terms,
            literals,
            inference
                .deduction(self.remaining())
                .deduction(extension.open()),
        );
        extension
    }

    pub(crate) fn lazy_extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: Extension,
    ) -> (Self, Self) {
        let (problem_clause, extension) =
            Self::extension(problem, terms, literals, constraints, extension);
        let p = &literals[self.current];
        let q = &literals[extension.current];
        let disequation_start = literals.len();

        let fresh_start = terms.len();
        for _ in p.get_predicate_arguments(&problem.symbols, terms) {
            terms.add_variable();
        }
        let fresh_end = terms.len();
        let fresh = Range::new(fresh_start, fresh_end);

        let mut inference = R::Inference::new("lazy_extension");
        for ((s, t), fresh) in
            predicate_argument_pairs(&problem.symbols, terms, p, q).zip(fresh)
        {
            constraints.assert_eq(fresh, s);
            inference.equation(fresh, s);
            literals.push(Literal::disequation(t, fresh));
        }
        let disequation_end = literals.len();
        let disequations = Self::new(disequation_start, disequation_end);

        record.inference(
            problem,
            terms,
            literals,
            inference
                .axiom(problem_clause, extension.open())
                .deduction(self.remaining())
                .deduction(extension.remaining())
                .deduction(disequations.open()),
        );
        (extension, disequations)
    }

    pub(crate) fn strict_backward_paramodulation<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        paramodulation: BackwardParamodulation,
    ) -> (Self, Self) {
        let target = paramodulation.target;
        let (problem_clause, extension, from, to) =
            Self::backward_paramodulation(
                problem,
                terms,
                literals,
                constraints,
                paramodulation,
            );
        let start = literals.len();
        let fresh = terms.add_variable();
        constraints.assert_eq(target, from);
        constraints.assert_gt(from, fresh);

        literals.push(Literal::disequation(fresh, to));
        literals.push(literals[self.current].subst(
            &problem.symbols,
            terms,
            target,
            fresh,
        ));
        let end = literals.len();
        let consequence = Self::new(start, end);

        record.inference(
            problem,
            terms,
            literals,
            R::Inference::new("strict_backward_paramodulation")
                .axiom(problem_clause, extension.open())
                .equation(target, from)
                .deduction(self.remaining())
                .deduction(extension.remaining())
                .deduction(consequence.open()),
        );
        (extension, consequence)
    }

    pub(crate) fn lazy_backward_paramodulation<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        paramodulation: BackwardParamodulation,
    ) -> (Self, Self) {
        let target = paramodulation.target;
        let (problem_clause, extension, from, to) =
            Self::backward_paramodulation(
                problem,
                terms,
                literals,
                constraints,
                paramodulation,
            );
        let start = literals.len();
        let fresh = terms.add_variable();
        let placeholder =
            terms.fresh_function(&problem.symbols, terms.symbol(from));
        constraints.assert_eq(target, placeholder);
        constraints.assert_neq(from, target);
        constraints.assert_gt(placeholder, fresh);

        for (s, t) in
            term_argument_pairs(&problem.symbols, terms, placeholder, from)
        {
            literals.push(Literal::disequation(s, t));
        }
        literals.push(Literal::disequation(fresh, to));
        literals.push(literals[self.current].subst(
            &problem.symbols,
            terms,
            target,
            fresh,
        ));
        let end = literals.len();
        let consequence = Self::new(start, end);

        record.inference(
            problem,
            terms,
            literals,
            R::Inference::new("lazy_backward_paramodulation")
                .axiom(problem_clause, extension.open())
                .equation(placeholder, target)
                .deduction(self.remaining())
                .deduction(extension.remaining())
                .deduction(consequence.open()),
        );
        (extension, consequence)
    }

    pub(crate) fn variable_backward_paramodulation<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        paramodulation: BackwardParamodulation,
    ) -> (Self, Self) {
        let target = paramodulation.target;
        let (problem_clause, extension, from, to) =
            Self::backward_paramodulation(
                problem,
                terms,
                literals,
                constraints,
                paramodulation,
            );

        let start = literals.len();
        let fresh = terms.add_variable();
        literals.push(Literal::disequation(to, fresh));
        literals.push(literals[self.current].subst(
            &problem.symbols,
            terms,
            target,
            fresh,
        ));
        let end = literals.len();
        let consequence = Self::new(start, end);

        constraints.assert_eq(target, from);
        constraints.assert_gt(from, fresh);

        record.inference(
            problem,
            terms,
            literals,
            R::Inference::new("variable_backward_paramodulation")
                .axiom(problem_clause, extension.open())
                .equation(target, from)
                .deduction(self.remaining())
                .deduction(extension.remaining())
                .deduction(consequence.open()),
        );

        (extension, consequence)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn strict_forward_paramodulation<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        paramodulation: ForwardParamodulation,
        l2r: bool,
    ) -> (Self, Self) {
        let (problem_clause, extension, target, fresh, from, to) = self
            .forward_paramodulation(
                problem,
                terms,
                literals,
                constraints,
                paramodulation,
                l2r,
            );
        constraints.assert_eq(target, from);

        let start = literals.len();
        literals.push(literals[extension.current].subst(
            &problem.symbols,
            terms,
            target,
            fresh,
        ));
        let end = literals.len();
        let consequence = Self::new(start, end);

        record.inference(
            problem,
            terms,
            literals,
            R::Inference::new("strict_forward_paramodulation")
                .axiom(problem_clause, extension.open())
                .equation(to, fresh)
                .equation(target, from)
                .deduction(self.remaining())
                .deduction(extension.remaining())
                .deduction(consequence.open()),
        );
        (extension, consequence)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lazy_forward_paramodulation<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        paramodulation: ForwardParamodulation,
        l2r: bool,
    ) -> (Self, Self) {
        let (problem_clause, extension, target, fresh, from, to) = self
            .forward_paramodulation(
                problem,
                terms,
                literals,
                constraints,
                paramodulation,
                l2r,
            );
        let placeholder =
            terms.fresh_function(&problem.symbols, terms.symbol(target));
        constraints.assert_eq(placeholder, from);
        constraints.assert_neq(target, from);

        let start = literals.len();
        for (s, t) in
            term_argument_pairs(&problem.symbols, terms, placeholder, target)
        {
            literals.push(Literal::disequation(s, t));
        }

        literals.push(literals[extension.current].subst(
            &problem.symbols,
            terms,
            target,
            fresh,
        ));
        let end = literals.len();
        let consequence = Self::new(start, end);

        record.inference(
            problem,
            terms,
            literals,
            R::Inference::new("lazy_forward_paramodulation")
                .axiom(problem_clause, extension.open())
                .equation(to, fresh)
                .equation(placeholder, from)
                .deduction(self.remaining())
                .deduction(extension.remaining())
                .deduction(consequence.open()),
        );
        (extension, consequence)
    }

    #[allow(clippy::too_many_arguments)]
    fn demodulation<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        demodulation: Demodulation,
        forward: bool,
        l2r: bool,
    ) -> Self {
        let (inference, equality, onto) = if forward {
            ("forward_demodulation", demodulation.literal, self.current)
        } else {
            ("backward_demodulation", self.current, demodulation.literal)
        };
        let target = demodulation.target;
        let (left, right) = literals[equality].get_equality();
        let (from, to) = if l2r { (left, right) } else { (right, left) };
        let fresh = terms.add_variable();
        constraints.assert_eq(to, fresh);
        constraints.assert_eq(target, from);
        constraints.assert_gt(from, to);

        let start = literals.len();
        literals.push(literals[onto].subst(
            &problem.symbols,
            terms,
            target,
            fresh,
        ));
        let end = literals.len();
        let consequence = Self::new(start, end);

        record.inference(
            problem,
            terms,
            literals,
            R::Inference::new(inference)
                .lemma(demodulation.literal)
                .equation(to, fresh)
                .equation(target, from)
                .deduction(self.remaining())
                .deduction(consequence.open()),
        );
        consequence
    }

    fn backward_paramodulation(
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        paramodulation: BackwardParamodulation,
    ) -> (Id<ProblemClause>, Self, Id<Term>, Id<Term>) {
        let occurrence =
            &problem.index.equality_occurrences[paramodulation.occurrence];
        let extension = Self::extend(
            problem,
            terms,
            literals,
            constraints,
            occurrence.clause,
            occurrence.literal,
        );

        let (left, right) = literals[extension.current].get_equality();
        let (from, to) = if occurrence.l2r {
            (left, right)
        } else {
            (right, left)
        };
        (occurrence.clause, extension, from, to)
    }

    #[allow(clippy::type_complexity)]
    fn forward_paramodulation(
        &self,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: ForwardParamodulation,
        l2r: bool,
    ) -> (
        Id<ProblemClause>,
        Self,
        Id<Term>,
        Id<Term>,
        Id<Term>,
        Id<Term>,
    ) {
        let (left, right) = literals[self.current].get_equality();
        let (from, to) = if l2r { (left, right) } else { (right, left) };
        let occurrence =
            &problem.index.subterm_occurrences[extension.occurrence];
        let target = occurrence.subterm + terms.current_offset();
        let extension = Self::extend(
            problem,
            terms,
            literals,
            constraints,
            occurrence.clause,
            occurrence.literal,
        );
        let fresh = terms.add_variable();
        constraints.assert_eq(to, fresh);
        constraints.assert_gt(from, to);
        (occurrence.clause, extension, target, fresh, from, to)
    }

    #[allow(clippy::too_many_arguments)]
    fn extend(
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        clause: Id<ProblemClause>,
        literal: Id<Literal>,
    ) -> Self {
        let literal_offset = literals.offset();
        let extension =
            Self::copy(problem, terms, literals, constraints, clause);

        let matching = literal + literal_offset;
        let mate = literals[matching];
        for index in Range::new(extension.current, matching).into_iter().rev()
        {
            literals[index + Offset::new(1)] = literals[index];
        }
        literals[extension.current] = mate;
        extension
    }

    fn copy(
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        clause: Id<ProblemClause>,
    ) -> Self {
        let start = literals.len();
        let offset = terms.current_offset();
        let clause = &problem.clauses[clause];
        terms.extend(&clause.terms);
        literals.extend(&clause.literals);

        let end = literals.len();
        for id in Range::new(start, end) {
            literals[id].offset(offset);
        }
        let clause = Self::new(start, end);
        clause.add_tautology_constraints(constraints, terms, literals);

        clause
    }

    fn add_tautology_constraints(
        &self,
        constraints: &mut Constraints,
        terms: &Terms,
        literals: &Literals,
    ) {
        let mut open = self.open().into_iter();
        while let Some(id) = open.next() {
            let literal = literals[id];
            if literal.polarity && literal.is_equality() {
                literal.add_reflexivity_constraints(constraints);
            }
            for other in open.clone().map(|id| &literals[id]) {
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
}
