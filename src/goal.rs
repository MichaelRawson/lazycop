use crate::clause::Clause;
use crate::prelude::*;
use crate::record::Record;
use crate::solver::Solver;

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

    pub(crate) fn num_open_branches(&self) -> u32 {
        self.stack
            .slice()
            .iter()
            .map(|clause| Range::len(clause.open()))
            .sum::<u32>()
    }

    pub(crate) fn apply_rule<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        solver: &mut Solver,
        rule: &Rule,
    ) {
        match rule {
            Rule::Start(start) => {
                self.stack.push(Clause::start(
                    record,
                    problem,
                    terms,
                    &mut self.literals,
                    solver,
                    start.start_clause,
                ));
            }
            Rule::Reduction(reduction) => {
                some(self.stack.last_mut()).predicate_reduction(
                    record,
                    &problem.signature(),
                    terms,
                    &mut self.literals,
                    solver,
                    reduction.term,
                );
                self.close_branches();
            }
            Rule::LazyPredicateExtension(extension) => {
                self.add_regularity_constraints(solver, terms, &self.literals);
                let new_clause = some(self.stack.last_mut())
                    .lazy_predicate_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        solver,
                        extension.occurrence,
                    );
                self.stack.push(new_clause);
                self.close_branches();
            }
            Rule::StrictPredicateExtension(extension) => {
                self.add_regularity_constraints(solver, terms, &self.literals);
                let new_clause = some(self.stack.last_mut())
                    .strict_predicate_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        solver,
                        extension.occurrence,
                    );
                self.stack.push(new_clause);
                self.close_branches();
            }
            Rule::LazyVariableExtension(extension) => {
                self.add_regularity_constraints(solver, terms, &self.literals);
                let new_clause = some(self.stack.last_mut())
                    .lazy_variable_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        solver,
                        extension.target,
                        extension.occurrence,
                    );
                self.stack.push(new_clause);
            }
            Rule::Reflexivity => {
                some(self.stack.last_mut()).reflexivity(
                    record,
                    &problem.signature(),
                    terms,
                    &self.literals,
                    solver,
                );
                self.close_branches();
            }
        }
    }

    fn close_branches(&mut self) {
        while !self.stack.is_empty() && some(self.stack.last_mut()).is_empty()
        {
            self.stack.pop();
        }
    }

    fn add_regularity_constraints(
        &self,
        solver: &mut Solver,
        terms: &Terms,
        literals: &Literals,
    ) {
        let current =
            &self.literals[some(self.stack.last()).current_literal()];
        if !current.polarity && current.is_equality() {
            current.add_reflexivity_constraints(solver);
        }

        for path in self.path_literals().map(|id| &literals[id]) {
            current.add_disequation_constraints(solver, terms, &path);
        }
        for ancestor in self
            .ancestor_literals()
            .map(|id| &literals[id])
            .filter(|reduction| reduction.polarity == current.polarity)
        {
            current.add_disequation_constraints(solver, terms, &ancestor);
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
        self.possible_variable_extension_rules(
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
        let literals = &self.literals;

        let path_literals = self
            .path_literals()
            .map(|id| &literals[id])
            .filter(|literal| literal.polarity != polarity);
        let ancestor_literals = self
            .ancestor_literals()
            .map(|id| &literals[id])
            .filter(|literal| literal.polarity == polarity);
        let correct_polarity = path_literals.chain(ancestor_literals);

        possible.extend(
            correct_polarity
                .filter(|literal| literal.is_predicate())
                .filter(|literal| {
                    literal.get_predicate_symbol(terms) == symbol
                })
                .map(|literal| literal.get_predicate())
                .map(|term| PredicateReduction { term })
                .map(Rule::Reduction),
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
        let matching = || {
            problem
                .query_predicates(polarity, symbol)
                .map(|occurrence| PredicateExtension { occurrence })
        };

        if problem.has_equality() {
            possible.extend(matching().map(Rule::LazyPredicateExtension));
        }
        possible.extend(matching().map(Rule::StrictPredicateExtension));
    }

    fn possible_equality_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        literal: &Literal,
    ) {
        if !literal.polarity {
            self.possible_reflexivity_rules(possible);
        }
    }

    fn possible_variable_extension_rules<E: Extend<Rule>>(
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
                    .map(|occurrence| VariableExtension { target, occurrence })
                    .map(Rule::LazyVariableExtension),
            );
        });
    }

    fn possible_reflexivity_rules<E: Extend<Rule>>(&self, possible: &mut E) {
        possible.extend(Some(Rule::Reflexivity));
    }

    fn ancestor_literals(&self) -> impl Iterator<Item = Id<Literal>> + '_ {
        let current = some(self.stack.last()).closed();
        let past = self
            .stack
            .slice()
            .iter()
            .rev()
            .skip(1)
            .flat_map(|clause| clause.closed().rev().skip(1));
        current.chain(past)
    }

    fn path_literals(&self) -> impl Iterator<Item = Id<Literal>> + '_ {
        self.stack
            .slice()
            .iter()
            .rev()
            .skip(1)
            .map(|clause| some(clause.closed().rev().next()))
    }
}
