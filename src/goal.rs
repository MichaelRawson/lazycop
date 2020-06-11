use crate::clause::Clause;
use crate::constraint::Constraints;
use crate::prelude::*;
use crate::record::Record;
use std::iter::once;

#[derive(Default)]
pub(crate) struct Goal {
    literals: Literals,
    stack: Block<Clause>,

    save_literals: Id<Literal>,
    save_stack: Block<Clause>,
}

impl Goal {
    pub(crate) fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.literals.clear();
        self.stack.clear();
    }

    pub(crate) fn save(&mut self) {
        self.save_literals = self.literals.len();
        self.save_stack.copy_from(&self.stack);
    }

    pub(crate) fn restore(&mut self) {
        self.literals.truncate(self.save_literals);
        self.stack.copy_from(&self.save_stack);
    }

    pub(crate) fn num_open_branches(&self) -> u16 {
        self.stack
            .slice()
            .iter()
            .map(|clause| Range::len(clause.remaining()) as u16)
            .sum::<u16>()
            + 1
    }

    pub(crate) fn apply_rule<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        constraints: &mut Constraints,
        rule: &Rule,
    ) {
        match *rule {
            Rule::Start(start) => {
                self.stack.push(Clause::start(
                    record,
                    problem,
                    terms,
                    &mut self.literals,
                    constraints,
                    start,
                ));
            }
            Rule::Reflexivity => {
                some(self.stack.last_mut()).reflexivity(
                    record,
                    &problem.signature(),
                    terms,
                    &self.literals,
                    constraints,
                );
                self.close_branches();
            }
            Rule::PredicateReduction(reduction) => {
                some(self.stack.last_mut()).predicate_reduction(
                    record,
                    &problem.signature(),
                    terms,
                    &mut self.literals,
                    constraints,
                    reduction,
                );
                self.close_branches();
            }
            Rule::EqualityReduction(reduction) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                    true,
                );
                let consequence = some(self.stack.last_mut())
                    .equality_reduction(
                        record,
                        &problem.signature(),
                        terms,
                        &mut self.literals,
                        constraints,
                        reduction,
                    );
                self.stack.push(consequence);
            }
            Rule::EqualityReductionReflexivity(reduction) => {
                let mut consequence = some(self.stack.last_mut())
                    .equality_reduction(
                        record,
                        &problem.signature(),
                        terms,
                        &mut self.literals,
                        constraints,
                        reduction,
                    );
                consequence.reflexivity(
                    record,
                    &problem.signature(),
                    terms,
                    &self.literals,
                    constraints,
                );
                self.close_branches();
            }
            Rule::LazyPredicateExtension(extension) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                    true,
                );
                let (extension, consequence) = some(self.stack.last_mut())
                    .lazy_predicate_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        constraints,
                        extension,
                    );
                self.stack.push(extension);
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                    false,
                );
                self.stack.push(consequence);
                self.close_branches();
            }
            Rule::StrictPredicateExtension(extension) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                    true,
                );
                let extension = some(self.stack.last_mut())
                    .strict_predicate_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        constraints,
                        extension,
                    );
                self.stack.push(extension);
                self.close_branches();
            }
            Rule::StrictFunctionExtension(extension) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                    true,
                );
                let (extension, consequence) = some(self.stack.last_mut())
                    .strict_function_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        constraints,
                        extension,
                    );
                self.stack.push(extension);
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                    false,
                );
                self.stack.push(consequence);
            }
            Rule::LazyFunctionExtension(extension) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                    true,
                );
                let (extension, consequence) = some(self.stack.last_mut())
                    .lazy_function_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        constraints,
                        extension,
                    );
                self.stack.push(extension);
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                    false,
                );
                self.stack.push(consequence);
            }
            Rule::VariableExtension(extension) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                    true,
                );
                let (extension, consequence) = some(self.stack.last_mut())
                    .variable_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        constraints,
                        extension,
                    );
                self.stack.push(extension);
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                    false,
                );
                self.stack.push(consequence);
            }
        }
    }

    fn close_branches(&mut self) {
        let last = some(self.stack.last());
        if !last.is_empty() {
            return;
        }
        self.stack.pop();
        while let Some(last) = self.stack.last_mut() {
            last.close_literal();
            if !last.is_empty() {
                return;
            }
            self.stack.pop();
        }
    }

    fn add_regularity_constraints(
        &self,
        constraints: &mut Constraints,
        terms: &Terms,
        literals: &Literals,
        strong: bool,
    ) {
        let current =
            &self.literals[some(self.stack.last()).current_literal()];
        for path in self
            .path_literals()
            .map(|id| &literals[id])
            .filter(|path| path.polarity == current.polarity)
        {
            current.add_disequation_constraints(constraints, terms, &path);
        }

        if strong {
            if !current.polarity && current.is_equality() {
                current.add_reflexivity_constraints(constraints);
            }

            for reduction in self
                .reduction_literals()
                .map(|id| &literals[id])
                .filter(|reduction| reduction.polarity != current.polarity)
            {
                current.add_disequation_constraints(
                    constraints,
                    terms,
                    &reduction,
                );
            }
        }
    }

    pub(crate) fn possible_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
        terms: &Terms,
    ) {
        let clause = some(self.stack.last());
        let literal = &self.literals[clause.current_literal()];
        if literal.is_predicate() {
            self.possible_predicate_rules(possible, problem, terms, literal);
        } else if literal.is_equality() {
            self.possible_equality_rules(possible, literal);
        }
        self.possible_equality_reduction_rules(
            possible,
            problem.signature(),
            terms,
            literal,
        );
        self.possible_equality_extension_rules(
            possible, problem, terms, literal,
        );
    }

    fn possible_predicate_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
        terms: &Terms,
        literal: &Literal,
    ) {
        self.possible_predicate_reduction_rules(possible, terms, literal);
        self.possible_predicate_extension_rules(
            possible, problem, terms, literal,
        );
    }

    fn possible_predicate_reduction_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        terms: &Terms,
        literal: &Literal,
    ) {
        let polarity = literal.polarity;
        let symbol = literal.get_predicate_symbol(terms);
        possible.extend(
            self.reduction_literals()
                .filter(|id| {
                    let reduction = &self.literals[*id];
                    reduction.polarity != polarity
                        && reduction.is_predicate()
                        && reduction.get_predicate_symbol(terms) == symbol
                })
                .map(|literal| PredicateReduction { literal })
                .map(Rule::PredicateReduction),
        );
    }

    fn possible_predicate_extension_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
        terms: &Terms,
        literal: &Literal,
    ) {
        let polarity = !literal.polarity;
        let symbol = literal.get_predicate_symbol(terms);
        let extensions = problem
            .query_predicates(polarity, symbol)
            .map(|occurrence| PredicateExtension { occurrence });

        for extension in extensions {
            if problem.has_equality() {
                possible.extend(once(Rule::LazyPredicateExtension(extension)));
            }
            possible.extend(once(Rule::StrictPredicateExtension(extension)));
        }
    }

    fn possible_equality_reduction_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        symbols: &Symbols,
        terms: &Terms,
        literal: &Literal,
    ) {
        let mut add_reduction = move |literal, target, from, reflexivity| {
            let rule_fn = if reflexivity {
                Rule::EqualityReductionReflexivity
            }
            else {
                Rule::EqualityReduction
            };

            if terms.is_variable(from)
                || terms.symbol(from) == terms.symbol(target)
            {
                let reduction = EqualityReduction {
                    literal,
                    target,
                    from,
                };
                possible.extend(once(rule_fn(reduction)));
            }
        };

        let possible_reflexivity = !literal.polarity && literal.is_equality();
        for id in self.reduction_literals() {
            let reduction = &self.literals[id];
            if !reduction.polarity || !reduction.is_equality() {
                continue;
            }
            let (left, right) = reduction.get_equality();
            literal.subterms(symbols, terms, &mut |target| {
                add_reduction(id, target, left, false);
                add_reduction(id, target, right, false);
            });
            if possible_reflexivity {
                let (target1, target2) = literal.get_equality();
                add_reduction(id, target1, left, true);
                add_reduction(id, target2, left, true);
                add_reduction(id, target1, right, true);
                add_reduction(id, target2, right, true);
            }
        }
    }

    fn possible_equality_extension_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
        terms: &Terms,
        literal: &Literal,
    ) {
        literal.subterms(problem.signature(), terms, &mut |target| {
            possible.extend(
                problem
                    .query_variable_equalities()
                    .map(|occurrence| EqualityExtension { target, occurrence })
                    .map(Rule::VariableExtension),
            );

            let symbol = terms.symbol(target);
            let function_extensions = problem
                .query_function_equalities(symbol)
                .map(|occurrence| EqualityExtension { target, occurrence });
            for extension in function_extensions {
                possible.extend(once(Rule::LazyFunctionExtension(extension)));
                possible
                    .extend(once(Rule::StrictFunctionExtension(extension)));
            }
        });
    }

    fn possible_equality_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        literal: &Literal,
    ) {
        if !literal.polarity {
            possible.extend(once(Rule::Reflexivity));
        }
    }

    fn reduction_literals(&self) -> impl Iterator<Item = Id<Literal>> + '_ {
        self.path_literals()
    }

    fn path_literals(&self) -> impl Iterator<Item = Id<Literal>> + '_ {
        self.stack
            .slice()
            .iter()
            .rev()
            .skip(1)
            .map(|clause| clause.current_literal())
    }
}
