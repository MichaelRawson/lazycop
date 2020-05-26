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
        while let Some(goal) = self.stack.last_mut() {
            let literal_id = goal.close_literal();
            self.literals[literal_id].polarity =
                !self.literals[literal_id].polarity;
            if goal.is_empty() {
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
        let goal = self.stack.last().unwrap();
        for path in self.path_literals().map(|id| &literals[id]) {
            for open in goal.open().map(|id| &literals[id]) {
                if path.polarity == open.polarity {
                    path.atom.add_disequation_constraints(
                        solver, terms, &open.atom,
                    );
                }
            }
        }
    }

    pub(crate) fn possible_rules(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        terms: &Terms,
    ) {
        if let Some(goal) = self.stack.last() {
            self.possible_nonstart_rules(
                possible,
                problem,
                terms,
                &self.literals,
                goal,
            );
        } else {
            self.possible_start_rules(possible, problem);
        }
    }

    fn possible_start_rules(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
    ) {
        possible.extend(
            problem
                .start_clauses
                .iter()
                .copied()
                .map(|start_clause| Start { start_clause })
                .map(Rule::Start),
        );
    }

    fn possible_nonstart_rules(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        terms: &Terms,
        literals: &Block<Literal>,
        clause: &Clause,
    ) {
        let literal = literals[clause.current_literal()];
        if literal.atom.is_predicate() {
            self.possible_predicate_rules(
                possible,
                problem,
                terms,
                literals,
                literal.polarity,
                literal.atom.get_predicate_symbol(terms),
            );
        } else if !literal.polarity {
            possible.push(Rule::Reflexivity);
        }
    }

    fn possible_predicate_rules(
        &self,
        possible: &mut Vec<Rule>,
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

    fn possible_reduction_rules(
        &self,
        possible: &mut Vec<Rule>,
        terms: &Terms,
        literals: &Block<Literal>,
        polarity: bool,
        f: Id<Symbol>,
    ) {
        for literal_id in self.path_literals() {
            let literal = literals[literal_id];
            if literal.polarity == polarity || !literal.atom.is_predicate() {
                continue;
            }
            let g = literal.atom.get_predicate_symbol(terms);
            if f == g {
                let literal = literal_id;
                possible.push(Rule::Reduction(PredicateReduction { literal }));
            }
        }
    }

    fn possible_predicate_extension_rules(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        polarity: bool,
        f: Id<Symbol>,
    ) {
        let opposite = !polarity as usize;
        let empty = vec![];
        let positions = problem.predicate_occurrences[opposite]
            .get(&f)
            .unwrap_or(&empty);
        possible.extend(positions.iter().copied().map(|(clause, literal)| {
            Rule::PredicateExtension(PredicateExtension { clause, literal })
        }));
    }

    fn path_literals(&self) -> impl Iterator<Item = Id<Literal>> + '_ {
        self.stack
            .iter()
            .rev()
            .skip(1)
            .flat_map(|clause| clause.path())
            .chain(self.stack.last().unwrap().closed())
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
