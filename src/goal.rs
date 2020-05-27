use crate::clause::Clause;
use crate::prelude::*;
use crate::record::Record;
use crate::solver::Solver;

#[derive(Default)]
pub(crate) struct Goal {
    literals: Block<Literal>,
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
            .map(|clause| Range::len(clause.open()) - 1)
            .sum::<u32>()
            + 1
    }

    pub(crate) fn apply_rule<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        solver: &mut Solver,
        rule: Rule,
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
                self.stack.push(start);
                self.close_branches();
            }
            Rule::Reduction(reduction) => {
                self.stack.last_mut().unwrap().predicate_reduction(
                    record,
                    &problem.symbols,
                    terms,
                    &self.literals,
                    solver,
                    reduction.literal,
                );
                self.close_branches();
            }
            Rule::PredicateExtension(extension) => {
                let new_clause =
                    self.stack.last_mut().unwrap().predicate_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        solver,
                        extension.clause,
                        extension.literal,
                    );
                self.stack.push(new_clause);
                self.add_regularity_constraints(solver, terms, &self.literals);
            }
            Rule::Reflexivity => {
                self.stack.last_mut().unwrap().reflexivity(
                    record,
                    &problem.symbols,
                    terms,
                    &self.literals,
                    solver,
                );
                self.close_branches();
            }
        }
    }

    fn close_branches(&mut self) {
        let current = self.stack.last_mut().unwrap();
        if !current.is_empty() {
            return;
        }
        self.stack.pop();
        while let Some(clause) = self.stack.last_mut() {
            let id = clause.close_literal();
            self.literals[id].invert();
            if clause.is_empty() {
                self.stack.pop();
            } else {
                return;
            }
        }
    }

    fn add_regularity_constraints(
        &self,
        solver: &mut Solver,
        terms: &Terms,
        literals: &Block<Literal>,
    ) {
        let clause = self.stack.last().unwrap();
        for path in self.path_literals().map(|id| &literals[id]) {
            for open in clause.open().map(|id| &literals[id]) {
                if path.polarity == open.polarity {
                    path.add_disequation_constraints(solver, terms, &open);
                }
            }
        }
    }

    pub(crate) fn possible_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
        terms: &Terms,
    ) {
        if let Some(clause) = self.stack.last() {
            self.possible_nonstart_rules(
                possible,
                problem,
                terms,
                &self.literals,
                clause,
            );
        } else {
            self.possible_start_rules(possible, problem);
        }
    }

    fn possible_start_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
    ) {
        possible.extend(
            problem
                .start_clauses()
                .map(|start_clause| Start { start_clause })
                .map(Rule::Start),
        );
    }

    fn possible_nonstart_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
        terms: &Terms,
        literals: &Block<Literal>,
        clause: &Clause,
    ) {
        let literal = literals[clause.current_literal()];
        if literal.is_predicate() {
            self.possible_predicate_rules(
                possible,
                problem,
                terms,
                literals,
                literal.polarity,
                literal.get_predicate_symbol(terms),
            );
        } else if !literal.polarity && literal.is_equality() {
            possible.extend(Some(Rule::Reflexivity));
        }
    }

    fn possible_predicate_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
        terms: &Terms,
        literals: &Block<Literal>,
        polarity: bool,
        symbol: Id<Symbol>,
    ) {
        self.possible_reduction_rules(
            possible, terms, literals, polarity, symbol,
        );
        self.possible_predicate_extension_rules(
            possible, problem, polarity, symbol,
        );
    }

    fn possible_reduction_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        terms: &Terms,
        literals: &Block<Literal>,
        polarity: bool,
        symbol: Id<Symbol>,
    ) {
        possible.extend(
            self.reduction_literals()
                .map(|id| (id, &literals[id]))
                .filter(|(_, literal)| literal.polarity != polarity)
                .filter(|(_, literal)| literal.is_predicate())
                .filter(|(_, literal)| {
                    literal.get_predicate_symbol(terms) == symbol
                })
                .map(|(literal, _)| PredicateReduction { literal })
                .map(Rule::Reduction),
        );
    }

    fn possible_predicate_extension_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
        polarity: bool,
        symbol: Id<Symbol>,
    ) {
        possible.extend(
            problem
                .query_predicates(!polarity, symbol)
                .map(|(clause, literal)| PredicateExtension {
                    clause,
                    literal,
                })
                .map(Rule::PredicateExtension),
        );
    }

    fn reduction_literals(&self) -> impl Iterator<Item = Id<Literal>> + '_ {
        self.path_literals().chain(self.ancestor_literals())
    }

    fn ancestor_literals(&self) -> impl Iterator<Item = Id<Literal>> + '_ {
        self.stack.iter().flat_map(|clause| clause.closed())
    }

    fn path_literals(&self) -> impl Iterator<Item = Id<Literal>> + '_ {
        self.stack
            .iter()
            .rev()
            .skip(1)
            .map(|clause| clause.current_literal())
    }
}

impl Clone for Goal {
    fn clone(&self) -> Self {
        unreachable!()
    }

    fn clone_from(&mut self, other: &Self) {
        self.literals.clone_from(&other.literals);
        self.stack.clone_from(&other.stack);
    }
}
