use crate::goal_stack::GoalStack;
use crate::solver::Solver;
use crate::io::record::Record;
use crate::prelude::*;

pub(crate) struct Tableau<'problem> {
    problem: &'problem Problem,
    term_graph: TermGraph,
    clause_storage: ClauseStorage,
    solver: Solver,
    goals: GoalStack,
}

impl<'problem> Tableau<'problem> {
    pub(crate) fn new(problem: &'problem Problem) -> Self {
        let term_graph = TermGraph::default();
        let clause_storage = ClauseStorage::default();
        let solver = Solver::default();
        let goals = GoalStack::default();
        Self {
            problem,
            term_graph,
            clause_storage,
            solver,
            goals,
        }
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.goals.is_empty()
    }

    pub(crate) fn num_open_branches(&self) -> u32 {
        self.goals.num_open_branches()
    }

    pub(crate) fn clear(&mut self) {
        self.term_graph.clear();
        self.clause_storage.clear();
        self.solver.clear();
        self.goals.clear();
    }

    pub(crate) fn mark(&mut self) {
        self.term_graph.mark();
        self.clause_storage.mark();
        self.goals.mark();
        self.solver.mark();
    }

    pub(crate) fn undo_to_mark(&mut self) {
        self.term_graph.undo_to_mark();
        self.clause_storage.undo_to_mark();
        self.goals.undo_to_mark();
        self.solver.undo_to_mark();
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
            &mut self.solver,
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

    pub(crate) fn solve_constraints(&mut self) {
        self.solver.solve(&self.term_graph)
    }

    pub(crate) fn check_constraints(&mut self) -> bool {
        self.solver.check(&self.term_graph)
    }

    pub(crate) fn record_unification<R: Record>(&mut self, record: &mut R) {
        record.unification(
            &self.problem.symbol_table,
            &self.term_graph,
            &self.solver.bindings,
        );
    }
}
