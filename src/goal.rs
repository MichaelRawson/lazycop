use crate::clause::Clause;
use crate::prelude::*;
use crate::record::Record;
use crate::solver::Solver;

#[derive(Default)]
pub(crate) struct Goal {
    literals: Literals,
    stack: Vec<Clause>,
}

impl Goal {
    pub(crate) fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.literals.clear();
        self.stack.clear();
    }

    pub(crate) fn num_open_branches(&self) -> u32 {
        self.stack
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
                let start = Clause::start(
                    record,
                    problem,
                    terms,
                    &mut self.literals,
                    solver,
                    start.start_clause,
                );
                if !start.is_empty() {
                    self.stack.push(start);
                }
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
            Rule::PredicateExtension(extension) => {
                self.add_regularity_constraints(solver, terms, &self.literals);
                let new_clause = some(self.stack.last_mut())
                    .predicate_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        solver,
                        extension.clause,
                        extension.literal,
                    );
                self.stack.push(new_clause);
            }
            Rule::VariableExtension(extension) => {
                self.add_regularity_constraints(solver, terms, &self.literals);
                let new_clause = some(self.stack.last_mut())
                    .variable_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        solver,
                        extension.clause,
                        extension.literal,
                        extension.target,
                        extension.from,
                        extension.to,
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
        self.path_literals()
            .map(|id| &literals[id])
            .for_each(|path| {
                current.add_disequation_constraints(solver, terms, &path);
            });

        self.ancestor_literals()
            .map(|id| &literals[id])
            .filter(|reduction| reduction.polarity == current.polarity)
            .for_each(|reduction| {
                current.add_disequation_constraints(solver, terms, &reduction);
            });
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
            self.possible_equality_rules(possible, problem, terms, literal);
        }
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
        possible.extend(
            problem
                .query_predicates(polarity, symbol)
                .map(|occurrence| PredicateExtension {
                    clause: occurrence.clause,
                    literal: occurrence.literal,
                })
                .map(Rule::PredicateExtension),
        );
    }

    fn possible_equality_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
        terms: &Terms,
        literal: &Literal,
    ) {
        let polarity = literal.polarity;
        if !polarity {
            self.possible_reflexivity_rules(possible);
            /*
            self.possible_variable_extension_rules(
                possible, problem, terms, literal,
            )
            */
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
                    .map(|occurrence| VariableExtension {
                        clause: occurrence.clause,
                        literal: occurrence.literal,
                        target,
                        from: occurrence.from,
                        to: occurrence.to,
                    })
                    .map(Box::new)
                    .map(Rule::VariableExtension),
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
            .iter()
            .rev()
            .skip(1)
            .flat_map(|clause| clause.closed().rev().skip(1));
        current.chain(past)
    }

    fn path_literals(&self) -> impl Iterator<Item = Id<Literal>> + '_ {
        self.stack
            .iter()
            .rev()
            .skip(1)
            .map(|clause| some(clause.closed().rev().next()))
    }
}

impl Clone for Goal {
    fn clone(&self) -> Self {
        unimplemented!()
    }

    fn clone_from(&mut self, other: &Self) {
        self.literals.clone_from(&other.literals);
        self.stack.clone_from(&other.stack);
    }
}
