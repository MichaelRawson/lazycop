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
        match *rule {
            Rule::Start(start) => {
                self.stack.push(Clause::start(
                    record,
                    problem,
                    terms,
                    &mut self.literals,
                    solver,
                    start,
                ));
            }
            Rule::PredicateReduction(reduction) => {
                some(self.stack.last_mut()).predicate_reduction(
                    record,
                    &problem.signature(),
                    terms,
                    &mut self.literals,
                    solver,
                    reduction,
                    true,
                );
                self.close_branches();
            }
            Rule::PredicateLemma(reduction) => {
                some(self.stack.last_mut()).predicate_reduction(
                    record,
                    &problem.signature(),
                    terms,
                    &mut self.literals,
                    solver,
                    reduction,
                    false,
                );
                self.close_branches();
            }
            Rule::EqualityReduction(reduction) => todo!(),
            Rule::LazyPredicateExtension(extension) => {
                self.add_regularity_constraints(solver, terms, &self.literals);
                let new_clause = some(self.stack.last_mut())
                    .lazy_predicate_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        solver,
                        extension,
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
                        extension,
                    );
                self.stack.push(new_clause);
                self.close_branches();
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
                        extension,
                    );
                self.stack.push(new_clause);
            }
            Rule::FunctionExtension(extension) => {
                self.add_regularity_constraints(solver, terms, &self.literals);
                let new_clause = some(self.stack.last_mut())
                    .function_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        solver,
                        extension,
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
        for lemma in self
            .lemma_literals()
            .map(|id| &literals[id])
            .filter(|reduction| reduction.polarity == current.polarity)
        {
            current.add_disequation_constraints(solver, terms, &lemma);
        }
    }

    pub(crate) fn possible_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
        terms: &Terms,
        solver: &mut Solver,
    ) {
        let clause = some(self.stack.last());
        let literal = &self.literals[clause.current_literal()];
        if literal.is_predicate() {
            self.possible_predicate_rules(
                possible, problem, terms, solver, literal,
            );
        } else if literal.is_equality() {
            self.possible_equality_rules(
                possible,
                problem.signature(),
                terms,
                solver,
                literal,
            );
        }
        self.possible_equality_extension_rules(
            possible, problem, terms, literal,
        );
    }

    fn possible_predicate_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
        terms: &Terms,
        solver: &mut Solver,
        literal: &Literal,
    ) {
        self.possible_predicate_reduction_rules(
            possible,
            problem.signature(),
            terms,
            solver,
            literal,
        );
        self.possible_predicate_lemma_rules(
            possible,
            problem.signature(),
            terms,
            solver,
            literal,
        );
        self.possible_predicate_extension_rules(
            possible, problem, terms, literal,
        );
    }

    fn possible_predicate_reduction_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        symbols: &Symbols,
        terms: &Terms,
        solver: &mut Solver,
        literal: &Literal,
    ) {
        let polarity = literal.polarity;
        let symbol = literal.get_predicate_symbol(terms);
        let predicate = literal.get_predicate();
        possible.extend(
            self.path_literals()
                .filter(|id| {
                    let path = self.literals[*id];
                    path.polarity != polarity
                        && path.is_predicate()
                        && path.get_predicate_symbol(terms) == symbol
                        && solver.possibly_equal(
                            symbols,
                            terms,
                            path.get_predicate(),
                            predicate,
                        )
                })
                .map(|literal| PredicateReduction { literal })
                .map(Rule::PredicateReduction),
        );
    }

    fn possible_predicate_lemma_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        symbols: &Symbols,
        terms: &Terms,
        solver: &mut Solver,
        literal: &Literal,
    ) {
        let polarity = literal.polarity;
        let symbol = literal.get_predicate_symbol(terms);
        let predicate = literal.get_predicate();
        possible.extend(
            self.lemma_literals()
                .filter(|id| {
                    let lemma = self.literals[*id];
                    lemma.polarity == polarity
                        && lemma.is_predicate()
                        && lemma.get_predicate_symbol(terms) == symbol
                        && solver.possibly_equal(
                            symbols,
                            terms,
                            lemma.get_predicate(),
                            predicate,
                        )
                })
                .map(|literal| PredicateReduction { literal })
                .map(Rule::PredicateLemma),
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
        let extensions = || {
            problem
                .query_predicates(polarity, symbol)
                .map(|occurrence| PredicateExtension { occurrence })
        };

        if problem.has_equality() {
            possible.extend(extensions().map(Rule::LazyPredicateExtension));
        }
        possible.extend(extensions().map(Rule::StrictPredicateExtension));
    }

    fn possible_equality_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        symbols: &Symbols,
        terms: &Terms,
        solver: &mut Solver,
        literal: &Literal,
    ) {
        let (left, right) = literal.get_equality();
        if !literal.polarity {
            self.possible_reflexivity_rules(
                possible, symbols, terms, solver, left, right,
            );
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
            possible.extend(
                problem
                    .query_function_equalities(symbol)
                    .map(|occurrence| EqualityExtension { target, occurrence })
                    .map(Rule::FunctionExtension),
            );
        });
    }

    fn possible_reflexivity_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        symbols: &Symbols,
        terms: &Terms,
        solver: &mut Solver,
        left: Id<Term>,
        right: Id<Term>,
    ) {
        if solver.possibly_equal(symbols, terms, left, right) {
            possible.extend(Some(Rule::Reflexivity));
        }
    }

    fn path_literals(&self) -> impl Iterator<Item = Id<Literal>> + '_ {
        self.stack
            .slice()
            .iter()
            .rev()
            .skip(1)
            .map(|clause| some(clause.closed().rev().next()))
    }

    fn lemma_literals(&self) -> impl Iterator<Item = Id<Literal>> + '_ {
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
}
