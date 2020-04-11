use crate::core::goal_stack::GoalStack;
use crate::core::solver::Solver;
use crate::io::record::Record;
use crate::prelude::*;

pub(crate) struct Tableau<'problem> {
    problem: &'problem Problem,
    term_graph: TermGraph,
    clause_storage: ClauseStorage,
    constraint_list: ConstraintList,
    solver: Solver,
    goals: GoalStack,
    previous: GoalStack,
}

impl<'problem> Tableau<'problem> {
    pub(crate) fn new(problem: &'problem Problem) -> Self {
        let term_graph = TermGraph::default();
        let clause_storage = ClauseStorage::default();
        let constraint_list = ConstraintList::default();
        let solver = Solver::default();
        let goals = GoalStack::default();
        let previous = GoalStack::default();
        Self {
            problem,
            term_graph,
            clause_storage,
            constraint_list,
            solver,
            goals,
            previous,
        }
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.goals.is_empty()
    }

    pub(crate) fn open_branches(&self) -> u32 {
        self.goals.open_branches()
    }

    pub(crate) fn clear(&mut self) {
        self.term_graph.clear();
        self.clause_storage.clear();
        self.constraint_list.clear();
        self.goals.clear();
        self.previous.clear();
    }

    pub(crate) fn mark(&mut self) {
        self.term_graph.mark();
        self.clause_storage.mark();
        self.constraint_list.mark();
        self.previous.reset_to(&self.goals);
    }

    pub(crate) fn undo(&mut self) {
        self.term_graph.undo_to_mark();
        self.clause_storage.undo_to_mark();
        self.constraint_list.undo_to_mark();
        self.goals.reset_to(&self.previous);
    }

    pub(crate) fn apply_rule<R: Record>(
        &mut self,
        record: &mut R,
        rule: Rule,
    ) {
        self.goals.apply_rule(
            record,
            &self.problem,
            &mut self.term_graph,
            &mut self.clause_storage,
            &mut self.constraint_list,
            rule,
        );
    }

    pub(crate) fn possible_rules(&self, possible: &mut Vec<Rule>) {
        self.goals.possible_rules(
            possible,
            &self.problem,
            &self.term_graph,
            &self.clause_storage,
        );
    }

    pub(crate) fn solve_constraints<R: Record>(
        &mut self,
        record: &mut R,
    ) -> bool {
        self.solver.solve(
            record,
            &self.problem.symbol_table,
            &self.term_graph,
            &self.constraint_list.equalities,
            &self.constraint_list.disequalities,
        )
    }
}
