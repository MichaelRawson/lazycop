use crate::prelude::*;
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
    start: Id<Literal>,
    pub(crate) axiom: Id<ProblemClause>,
}

impl Clause {
    fn new(start: Id<Literal>, end: Id<Literal>) -> Self {
        let axiom = Id::default();
        let current = start;
        Self {
            current,
            end,
            start,
            axiom,
        }
    }

    pub(crate) fn new_from_axiom(
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        axiom: Id<ProblemClause>,
    ) -> Self {
        let start = literals.end();
        let offset = terms.offset();
        let clause = &problem.clauses[axiom];
        terms.extend_from(&clause.terms);
        literals.extend_from_block(&clause.literals);

        let end = literals.end();
        for id in Range::new(start, end) {
            literals[id].offset(offset);
        }
        let current = start;
        Self {
            current,
            end,
            start,
            axiom,
        }
    }

    pub(crate) fn is_empty(self) -> bool {
        self.current == self.end
    }

    pub(crate) fn original(self) -> Range<Literal> {
        Range::new(self.start, self.end)
    }

    pub(crate) fn open(self) -> Range<Literal> {
        Range::new(self.current, self.end)
    }

    pub(crate) fn current_literal(self) -> Id<Literal> {
        self.current
    }

    pub(crate) fn close_literal(&mut self) -> Id<Literal> {
        let result = self.current;
        self.current = self.current + Offset::new(1);
        result
    }

    pub(crate) fn graph(
        &self,
        graph: &mut Graph,
        symbols: &Symbols,
        terms: &Terms,
        literals: &Literals,
        bindings: &Bindings,
    ) -> Id<Node> {
        let root = graph.clause();
        for literal in self.open().into_iter() {
            let node =
                literals[literal].graph(graph, symbols, terms, bindings);
            graph.store_literal(literal, node);
            graph.connect(root, node);
        }
        root
    }

    pub(crate) fn start(
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        start: Start,
    ) -> Self {
        let axiom =
            Self::new_from_axiom(problem, terms, literals, start.clause);
        axiom.add_tautology_constraints(terms, literals, constraints);
        axiom
    }

    pub(crate) fn reflexivity(
        &mut self,
        literals: &Literals,
        constraints: &mut Constraints,
    ) {
        let literal = literals[self.close_literal()];
        let (left, right) = literal.get_equality();
        constraints.assert_eq(left, right);
    }

    pub(crate) fn distinct_objects(&mut self) {
        self.close_literal();
    }

    pub(crate) fn reduction(
        &mut self,
        problem: &Problem,
        terms: &Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        reduction: Reduction,
    ) {
        let p = &literals[self.close_literal()];
        let q = &literals[reduction.literal];
        for (s, t) in predicate_argument_pairs(&problem.symbols, terms, p, q) {
            constraints.assert_eq(s, t);
        }
    }

    pub(crate) fn demodulation(
        &mut self,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        demodulation: Demodulation,
        forward: bool,
        l2r: bool,
    ) -> Self {
        let (equality, onto) = if forward {
            (demodulation.literal, self.current)
        } else {
            (self.current, demodulation.literal)
        };
        let target = demodulation.target;
        let (left, right) = literals[equality].get_equality();
        let (from, to) = if l2r { (left, right) } else { (right, left) };
        let fresh = terms.add_variable();
        constraints.assert_eq(to, fresh);
        constraints.assert_eq(target, from);
        constraints.assert_gt(from, to);

        let start = literals.end();
        literals.push(literals[onto].subst(
            &problem.symbols,
            terms,
            target,
            fresh,
        ));
        let end = literals.end();
        Self::new(start, end)
    }

    fn extension(
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: Extension,
    ) -> Self {
        let occurrence =
            &problem.index.predicate_occurrences[extension.occurrence];
        Self::extend(
            problem,
            terms,
            literals,
            constraints,
            occurrence.clause,
            occurrence.literal,
        )
    }

    pub(crate) fn strict_extension(
        &mut self,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: Extension,
    ) -> Self {
        let mut extension =
            Self::extension(problem, terms, literals, constraints, extension);
        let p = &literals[self.current];
        let q = &literals[extension.close_literal()];
        for (s, t) in predicate_argument_pairs(&problem.symbols, terms, p, q) {
            constraints.assert_eq(s, t);
        }
        extension
    }

    pub(crate) fn lazy_extension(
        &mut self,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        extension: Extension,
    ) -> (Self, Self) {
        let extension =
            Self::extension(problem, terms, literals, constraints, extension);
        let p = &literals[self.current];
        let q = &literals[extension.current];
        let disequation_start = literals.end();

        let fresh_start = terms.end();
        for _ in p.get_predicate_arguments(&problem.symbols, terms) {
            terms.add_variable();
        }
        let fresh_end = terms.end();
        let fresh = Range::new(fresh_start, fresh_end);

        for ((s, t), fresh) in
            predicate_argument_pairs(&problem.symbols, terms, p, q).zip(fresh)
        {
            constraints.assert_eq(fresh, s);
            literals.push(Literal::disequation(fresh, t));
        }
        let disequation_end = literals.end();
        let disequations = Self::new(disequation_start, disequation_end);
        (extension, disequations)
    }

    pub(crate) fn strict_backward_paramodulation(
        &mut self,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        paramodulation: BackwardParamodulation,
    ) -> (Self, Self) {
        let target = paramodulation.target;
        let (extension, from, to) = Self::backward_paramodulation(
            problem,
            terms,
            literals,
            constraints,
            paramodulation,
        );
        let start = literals.end();
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
        let end = literals.end();
        let consequence = Self::new(start, end);
        (extension, consequence)
    }

    pub(crate) fn lazy_backward_paramodulation(
        &mut self,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        paramodulation: BackwardParamodulation,
    ) -> (Self, Self) {
        let target = paramodulation.target;
        let (extension, from, to) = Self::backward_paramodulation(
            problem,
            terms,
            literals,
            constraints,
            paramodulation,
        );
        let start = literals.end();
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
        let end = literals.end();
        let consequence = Self::new(start, end);
        (extension, consequence)
    }

    pub(crate) fn variable_backward_paramodulation(
        &mut self,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        paramodulation: BackwardParamodulation,
    ) -> (Self, Self) {
        let target = paramodulation.target;
        let (extension, from, to) = Self::backward_paramodulation(
            problem,
            terms,
            literals,
            constraints,
            paramodulation,
        );

        let start = literals.end();
        let fresh = terms.add_variable();
        literals.push(Literal::disequation(fresh, to));
        literals.push(literals[self.current].subst(
            &problem.symbols,
            terms,
            target,
            fresh,
        ));
        let end = literals.end();
        let consequence = Self::new(start, end);

        constraints.assert_eq(target, from);
        constraints.assert_gt(from, fresh);
        (extension, consequence)
    }

    pub(crate) fn strict_forward_paramodulation(
        &mut self,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        paramodulation: ForwardParamodulation,
        l2r: bool,
    ) -> (Self, Self) {
        let (extension, target, fresh, from) = self.forward_paramodulation(
            problem,
            terms,
            literals,
            constraints,
            paramodulation,
            l2r,
        );
        constraints.assert_eq(target, from);

        let start = literals.end();
        literals.push(literals[extension.current].subst(
            &problem.symbols,
            terms,
            target,
            fresh,
        ));
        let end = literals.end();
        let consequence = Self::new(start, end);
        (extension, consequence)
    }

    pub(crate) fn lazy_forward_paramodulation(
        &mut self,
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        paramodulation: ForwardParamodulation,
        l2r: bool,
    ) -> (Self, Self) {
        let (extension, target, fresh, from) = self.forward_paramodulation(
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

        let start = literals.end();
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
        let end = literals.end();
        let consequence = Self::new(start, end);
        (extension, consequence)
    }

    fn backward_paramodulation(
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        paramodulation: BackwardParamodulation,
    ) -> (Self, Id<Term>, Id<Term>) {
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
        (extension, from, to)
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
    ) -> (Self, Id<Term>, Id<Term>, Id<Term>) {
        let (left, right) = literals[self.current].get_equality();
        let (from, to) = if l2r { (left, right) } else { (right, left) };
        let occurrence =
            &problem.index.subterm_occurrences[extension.occurrence];
        let target = occurrence.subterm + terms.offset();
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
        (extension, target, fresh, from)
    }

    fn extend(
        problem: &Problem,
        terms: &mut Terms,
        literals: &mut Literals,
        constraints: &mut Constraints,
        clause: Id<ProblemClause>,
        literal: Id<Literal>,
    ) -> Self {
        let literal_offset = literals.offset();
        let extension = Self::new_from_axiom(problem, terms, literals, clause);
        extension.add_tautology_constraints(terms, literals, constraints);

        let matching = literal + literal_offset;
        let mate = literals[matching];
        for index in Range::new(extension.current, matching).into_iter().rev()
        {
            literals[index + Offset::new(1)] = literals[index];
        }
        literals[extension.current] = mate;
        extension
    }

    fn add_tautology_constraints(
        &self,
        terms: &Terms,
        literals: &Literals,
        constraints: &mut Constraints,
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
