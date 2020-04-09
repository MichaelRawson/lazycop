use crate::core::goal::Goal;
use crate::io::record::Record;
use crate::prelude::*;

#[derive(Default)]
pub struct GoalStack {
    stack: Arena<Goal>,
    previous: Arena<Goal>,
}

impl GoalStack {
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    pub fn clear(&mut self) {
        self.stack.clear();
        self.previous.clear();
    }

    pub fn mark(&mut self) {
        self.previous.clear();
        self.previous.copy_from(&self.stack);
    }

    pub fn undo_to_mark(&mut self) {
        self.stack.clear();
        self.stack.copy_from(&self.previous);
    }

    pub fn open_branches(&self) -> u32 {
        1 + self
            .stack
            .into_iter()
            .map(|id| self.stack[id].open_branches())
            .sum::<u32>()
    }

    pub fn apply_rule<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        term_graph: &mut TermGraph,
        clause_storage: &mut ClauseStorage,
        constraint_list: &mut ConstraintList,
        rule: Rule,
    ) {
        match rule {
            Rule::Start(start) => {
                let start = Goal::start(
                    record,
                    problem,
                    term_graph,
                    clause_storage,
                    start.start_clause,
                );
                self.stack.push(start);
            }
            Rule::Reduction(reduction) => {
                let by = self.stack[reduction.goal];
                let goal =
                    self.stack.last_mut().expect("reduction on empty tableau");
                goal.reduction(
                    record,
                    &problem.symbol_table,
                    term_graph,
                    clause_storage,
                    constraint_list,
                    by,
                );
                self.check_closed_branches(clause_storage);
            }
            Rule::EqualityReduction => {
                let goal = self
                    .stack
                    .last_mut()
                    .expect("equality reduction on empty tableau");
                goal.equality_reduction(
                    record,
                    &problem.symbol_table,
                    term_graph,
                    clause_storage,
                    constraint_list,
                );
                self.check_closed_branches(clause_storage);
            }
            Rule::Extension(extension) => {
                self.add_regularity_constraints(
                    clause_storage,
                    constraint_list,
                );
                let goal =
                    self.stack.last_mut().expect("extension on empty tableau");
                let new_goal = goal.extension(
                    record,
                    problem,
                    term_graph,
                    clause_storage,
                    constraint_list,
                    extension.position,
                );
                self.stack.push(new_goal);
            }
        }
    }

    pub fn possible_rules(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
    ) {
        if let Some(goal) = self.stack.last() {
            self.possible_nonstart_rules(
                possible,
                problem,
                term_graph,
                clause_storage,
                *goal,
            );
        } else {
            self.possible_start_rules(possible, problem);
        }
    }

    fn check_closed_branches(&mut self, clause_storage: &ClauseStorage) {
        let goal = self.stack.last().expect("empty tableau");
        if !goal.is_finished() {
            return;
        }
        self.stack.pop();
        while let Some(goal) = self.stack.last_mut() {
            goal.pop_literal(clause_storage);
            if goal.is_finished() {
                self.stack.pop();
            } else {
                break;
            }
        }
    }

    fn add_regularity_constraints(
        &self,
        clause_storage: &ClauseStorage,
        constraint_list: &mut ConstraintList,
    ) {
        let mut path_literals = self.path_literals(clause_storage);
        let literal = path_literals.next().expect("empty path");
        if let Atom::Predicate(left) = literal.atom {
            for path_literal in path_literals {
                if literal.polarity == path_literal.polarity {
                    if let Atom::Predicate(right) = path_literal.atom {
                        constraint_list.add_disequality(left, right);
                    }
                }
            }
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
                .map(|start_clause| StartRule { start_clause })
                .map(Rule::Start),
        );
    }

    fn possible_nonstart_rules(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        goal: Goal,
    ) {
        let literal =
            goal.current_literal(clause_storage).expect("empty goal");
        match literal.atom {
            Atom::Predicate(term) => {
                let p = if let TermView::Function(p, _) = term_graph.view(term)
                {
                    p
                } else {
                    unreachable!("non-functional predicate");
                };
                if let Some(positions) = problem.predicate_occurrences
                    [!literal.polarity as usize]
                    .get(p)
                {
                    possible.extend(
                        positions
                            .iter()
                            .copied()
                            .map(|position| ExtensionRule { position })
                            .map(Rule::Extension),
                    );
                }
                for goal in self.stack.into_iter() {
                    let path_literal = self.stack[goal]
                        .current_literal(clause_storage)
                        .expect("empty goal in stack");
                    if path_literal.polarity == literal.polarity {
                        continue;
                    }
                    if let Atom::Predicate(_) = path_literal.atom {
                        possible.push(Rule::Reduction(ReductionRule { goal }));
                    }
                }
            }
            Atom::Equality(_, _) => {
                if !literal.polarity {
                    possible.push(Rule::EqualityReduction);
                }
            }
        }
    }

    fn path_literals<'a, 'storage, 'iterator>(
        &'a self,
        clause_storage: &'storage ClauseStorage,
    ) -> impl Iterator<Item = Literal> + 'iterator
    where
        'storage: 'iterator,
        'a: 'iterator,
    {
        self.stack.into_iter().map(move |id| {
            self.stack[id]
                .current_literal(clause_storage)
                .expect("empty clause in stack")
        })
    }
}
