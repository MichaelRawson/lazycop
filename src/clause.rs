use crate::constraint::Constraints;
use crate::prelude::*;
use crate::record::{Inference, Record};
use crate::rule::*;

fn argument_pairs<'terms>(
    symbols: &Symbols,
    terms: &'terms Terms,
    p: &Literal,
    q: &Literal,
) -> impl Iterator<Item = (Id<Term>, Id<Term>)> + Clone + 'terms {
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
        let start = Self::copy(
            record,
            problem,
            terms,
            literals,
            constraints,
            start.clause,
        );
        record.inference(
            &problem.symbols,
            terms,
            literals,
            R::Inference::new("start").deduction(start.open()),
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
            R::Inference::new("reflexivity")
                .equation(left, right)
                .deduction(self.open()),
        );
    }

    pub(crate) fn reduction<R: Record>(
        &mut self,
        record: &mut R,
        symbols: &Symbols,
        terms: &Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        reduction: Reduction,
    ) {
        let mut inference = R::Inference::new("reduction");
        let p = &literals[self.close_literal()];
        let q = &literals[reduction.literal];
        for (s, t) in argument_pairs(symbols, terms, p, q) {
            constraints.assert_eq(s, t);
            inference.equation(s, t);
        }

        record.inference(
            symbols,
            terms,
            literals,
            inference.literal(reduction.literal).deduction(self.open()),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_demodulation<R: Record>(
        &mut self,
        record: &mut R,
        symbols: &Symbols,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        demodulation: Demodulation,
        lr: bool,
    ) -> Self {
        let target = demodulation.target;
        let (left, right) = literals[demodulation.literal].get_equality();
        let (from, to) = if lr { (left, right) } else { (right, left) };
        let fresh = terms.add_variable();
        constraints.assert_eq(to, fresh);
        constraints.assert_eq(target, from);
        constraints.assert_gt(from, to);

        let start = literals.len();
        literals
            .push(literals[self.current].subst(symbols, terms, target, fresh));
        let end = literals.len();
        let consequence = Self::new(start, end);

        record.inference(
            symbols,
            terms,
            literals,
            R::Inference::new("forward_demodulation")
                .equation(to, fresh)
                .equation(target, from)
                .literal(demodulation.literal)
                .deduction(self.remaining())
                .deduction(consequence.open()),
        );
        consequence
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn backward_demodulation<R: Record>(
        &mut self,
        record: &mut R,
        symbols: &Symbols,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        demodulation: Demodulation,
        l2r: bool,
    ) -> Self {
        let target = demodulation.target;
        let (left, right) = literals[self.current].get_equality();
        let (from, to) = if l2r { (left, right) } else { (right, left) };
        let fresh = terms.add_variable();
        constraints.assert_eq(to, fresh);
        constraints.assert_eq(target, from);
        constraints.assert_gt(from, to);

        let start = literals.len();
        literals.push(
            literals[demodulation.literal]
                .subst(symbols, terms, target, fresh),
        );
        let end = literals.len();
        let consequence = Self::new(start, end);

        record.inference(
            symbols,
            terms,
            literals,
            R::Inference::new("backward_demodulation")
                .equation(to, fresh)
                .equation(target, from)
                .literal(demodulation.literal)
                .deduction(self.remaining())
                .deduction(consequence.open()),
        );
        consequence
    }

    pub(crate) fn strict_extension<R: Record>(
        self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: Extension,
    ) -> Self {
        let mut inference = R::Inference::new("strict_extension");
        let mut extension = Self::extension(
            record,
            problem,
            terms,
            literals,
            constraints,
            extension,
        );
        let p = &literals[self.current];
        let q = &literals[extension.close_literal()];
        for (s, t) in argument_pairs(&problem.symbols, terms, p, q) {
            constraints.assert_eq(s, t);
            inference.equation(s, t);
        }

        record.inference(
            &problem.symbols,
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
        let mut inference = R::Inference::new("lazy_extension");
        let extension = Self::extension(
            record,
            problem,
            terms,
            literals,
            constraints,
            extension,
        );
        let p = &literals[self.current];
        let q = &literals[extension.current];
        let disequation_start = literals.len();

        let fresh_start = terms.len();
        for _ in p.get_predicate_arguments(&problem.symbols, terms) {
            terms.add_variable();
        }
        let fresh_end = terms.len();
        let fresh = Range::new(fresh_start, fresh_end);

        for ((s, t), fresh) in
            argument_pairs(&problem.symbols, terms, p, q).zip(fresh)
        {
            constraints.assert_eq(fresh, s);
            inference.equation(fresh, s);
            literals.push(Literal::disequation(t, fresh));
        }
        let disequation_end = literals.len();
        let disequations = Self::new(disequation_start, disequation_end);

        record.inference(
            &problem.symbols,
            terms,
            literals,
            inference
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
        let (extension, from, to) = Self::backward_paramodulation(
            record,
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
            &problem.symbols,
            terms,
            literals,
            R::Inference::new("strict_backward_paramodulation")
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
        let (extension, from, to) = Self::backward_paramodulation(
            record,
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

        let ss = terms
            .arguments(&problem.symbols, placeholder)
            .into_iter()
            .map(|s| terms.resolve(s));
        let ts = terms
            .arguments(&problem.symbols, from)
            .into_iter()
            .map(|t| terms.resolve(t));
        for (s, t) in ss.zip(ts) {
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
            &problem.symbols,
            terms,
            literals,
            R::Inference::new("lazy_backward_paramodulation")
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
        let (extension, from, to) = Self::backward_paramodulation(
            record,
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
            &problem.symbols,
            terms,
            literals,
            R::Inference::new("variable_backward_paramodulation")
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
        let (left, right) = literals[self.current].get_equality();
        let (from, to) = if l2r { (left, right) } else { (right, left) };
        let (extension, target) = Self::forward_paramodulation(
            record,
            problem,
            terms,
            literals,
            constraints,
            paramodulation,
        );
        let fresh = terms.add_variable();
        let placeholder =
            terms.fresh_function(&problem.symbols, terms.symbol(target));
        constraints.assert_eq(to, fresh);
        constraints.assert_eq(placeholder, from);
        constraints.assert_neq(target, from);
        constraints.assert_gt(from, to);

        let start = literals.len();
        let ss = terms
            .arguments(&problem.symbols, placeholder)
            .into_iter()
            .map(|s| terms.resolve(s));
        let ts = terms
            .arguments(&problem.symbols, target)
            .into_iter()
            .map(|t| terms.resolve(t));
        for (s, t) in ss.zip(ts) {
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
            &problem.symbols,
            terms,
            literals,
            R::Inference::new("lazy_forward_paramodulation")
                .equation(to, fresh)
                .equation(placeholder, from)
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
        let (left, right) = literals[self.current].get_equality();
        let (from, to) = if l2r { (left, right) } else { (right, left) };
        let fresh = terms.add_variable();

        let (extension, target) = Self::forward_paramodulation(
            record,
            problem,
            terms,
            literals,
            constraints,
            paramodulation,
        );
        constraints.assert_eq(to, fresh);
        constraints.assert_eq(target, from);
        constraints.assert_gt(from, to);

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
            &problem.symbols,
            terms,
            literals,
            R::Inference::new("strict_forward_paramodulation")
                .equation(to, fresh)
                .equation(target, from)
                .deduction(self.remaining())
                .deduction(extension.remaining())
                .deduction(consequence.open()),
        );
        (extension, consequence)
    }

    fn extension<R: Record>(
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: Extension,
    ) -> Self {
        let occurrence =
            &problem.index.predicate_occurrences[extension.occurrence];
        Self::extend(
            record,
            problem,
            terms,
            literals,
            constraints,
            occurrence.clause,
            occurrence.literal,
        )
    }

    fn backward_paramodulation<R: Record>(
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        paramodulation: BackwardParamodulation,
    ) -> (Self, Id<Term>, Id<Term>) {
        let occurrence =
            &problem.index.equality_occurrences[paramodulation.occurrence];
        let extension = Self::extend(
            record,
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
        (extension, from, to)
    }

    fn forward_paramodulation<R: Record>(
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: ForwardParamodulation,
    ) -> (Self, Id<Term>) {
        let occurrence =
            &problem.index.subterm_occurrences[extension.occurrence];
        let target = occurrence.subterm + terms.current_offset();
        let extension = Self::extend(
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
    fn extend<R: Record>(
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
        for index in Range::new(extension.current, matching).into_iter().rev()
        {
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
        record.axiom(&problem, clause);

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
        self,
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
