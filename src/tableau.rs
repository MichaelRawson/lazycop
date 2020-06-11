use crate::binding::Bindings;
use crate::constraint::Constraints;
use crate::disequation_solver::DisequationSolver;
use crate::equation_solver::EquationSolver;
use crate::goal::Goal;
use crate::occurs::{Check, SkipCheck};
use crate::ordering_solver::OrderingSolver;
use crate::prelude::*;
use crate::record::Record;

pub(crate) struct Tableau<'problem> {
    problem: &'problem Problem,
    terms: Terms,
    goal: Goal,
    bindings: Bindings,
    constraints: Constraints,
    disequation_solver: DisequationSolver,
    equation_solver: EquationSolver,
    ordering_solver: OrderingSolver,
}

impl<'problem> Tableau<'problem> {
    pub(crate) fn new(problem: &'problem Problem) -> Self {
        let terms = Terms::default();
        let goal = Goal::default();
        let bindings = Bindings::default();
        let constraints = Constraints::default();
        let disequation_solver = DisequationSolver::default();
        let equation_solver = EquationSolver::default();
        let ordering_solver = OrderingSolver::default();
        Self {
            problem,
            goal,
            terms,
            bindings,
            constraints,
            disequation_solver,
            equation_solver,
            ordering_solver,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.terms.clear();
        self.goal.clear();
        self.bindings.clear();
        self.constraints.clear();
        self.disequation_solver.clear();
        self.equation_solver.clear();
        self.ordering_solver.clear();
    }

    pub(crate) fn save(&mut self) {
        self.terms.save();
        self.goal.save();
        self.bindings.save();
        self.constraints.save();
        self.disequation_solver.save();
        self.equation_solver.save();
        self.ordering_solver.save();
    }

    pub(crate) fn restore(&mut self) {
        self.terms.restore();
        self.goal.restore();
        self.bindings.restore();
        self.constraints.restore();
        self.disequation_solver.restore();
        self.equation_solver.restore();
        self.ordering_solver.restore();
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.goal.is_empty()
    }

    pub(crate) fn num_open_branches(&self) -> u32 {
        self.goal.num_open_branches()
    }

    pub(crate) fn apply_rule<R: Record>(
        &mut self,
        record: &mut R,
        rule: &Rule,
    ) {
        self.goal.apply_rule(
            record,
            &self.problem,
            &mut self.terms,
            &mut self.constraints,
            rule,
        );
    }

    pub(crate) fn possible_rules<E: Extend<Rule>>(&self, possible: &mut E) {
        self.goal
            .possible_rules(possible, self.problem, &self.terms);
    }

    pub(crate) fn simplify_constraints(&mut self) -> bool {
        self.equation_solver.solve::<SkipCheck, _>(
            self.problem.signature(),
            &self.terms,
            &mut self.bindings,
            self.constraints.drain_equations(),
        ) && self.disequation_solver.simplify(
            self.problem.signature(),
            &self.terms,
            &self.bindings,
            self.constraints.drain_disequations(),
        ) && self.disequation_solver.simplify_symmetric(
            self.problem.signature(),
            &self.terms,
            &self.bindings,
            self.constraints.drain_symmetric_disequations(),
        ) && self.ordering_solver.simplify(
            self.problem.signature(),
            &self.terms,
            &self.bindings,
            self.constraints.drain_orderings(),
        )
    }

    pub(crate) fn solve_constraints(&mut self) -> bool {
        self.equation_solver.solve::<Check, _>(
            self.problem.signature(),
            &self.terms,
            &mut self.bindings,
            self.constraints.drain_equations(),
        ) && self.disequation_solver.simplify(
            self.problem.signature(),
            &self.terms,
            &self.bindings,
            self.constraints.drain_disequations(),
        ) && self.disequation_solver.simplify_symmetric(
            self.problem.signature(),
            &self.terms,
            &self.bindings,
            self.constraints.drain_symmetric_disequations(),
        ) && self.ordering_solver.simplify(
            self.problem.signature(),
            &self.terms,
            &self.bindings,
            self.constraints.drain_orderings(),
        ) && self.disequation_solver.check(
            self.problem.signature(),
            &self.terms,
            &self.bindings,
        ) && self.ordering_solver.check(
            self.problem.signature(),
            &self.terms,
            &self.bindings,
        )
    }

    pub(crate) fn record_unification<R: Record>(&mut self, record: &mut R) {
        record.unification(
            &self.problem.signature(),
            &self.terms,
            self.bindings.items(),
        );
    }
}
