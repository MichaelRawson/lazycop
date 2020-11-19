use crate::disequation_solver::DisequationSolver;
use crate::equation_solver::EquationSolver;
use crate::occurs::{Check, SkipCheck};
use crate::ordering_solver::OrderingSolver;
use crate::prelude::*;
use crate::tableau::Tableau;

pub(crate) struct Goal<'problem> {
    problem: &'problem Problem,
    pub(crate) terms: Terms,
    pub(crate) tableau: Tableau,
    pub(crate) bindings: Bindings,
    constraints: Constraints,
    disequation_solver: DisequationSolver,
    equation_solver: EquationSolver,
    ordering_solver: OrderingSolver,
}

impl<'problem> Goal<'problem> {
    pub(crate) fn new(problem: &'problem Problem) -> Self {
        let terms = Terms::default();
        let tableau = Tableau::default();
        let bindings = Bindings::default();
        let constraints = Constraints::default();
        let disequation_solver = DisequationSolver::default();
        let equation_solver = EquationSolver::default();
        let ordering_solver = OrderingSolver::default();
        Self {
            problem,
            terms,
            tableau,
            bindings,
            constraints,
            disequation_solver,
            equation_solver,
            ordering_solver,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.terms.clear();
        self.tableau.clear();
        self.bindings.clear();
        self.constraints.clear();
        self.disequation_solver.clear();
        self.equation_solver.clear();
        self.ordering_solver.clear();
    }

    pub(crate) fn save(&mut self) {
        self.terms.save();
        self.tableau.save();
        self.bindings.save();
        self.constraints.save();
        self.disequation_solver.save();
        self.equation_solver.save();
        self.ordering_solver.save();
    }

    pub(crate) fn restore(&mut self) {
        self.terms.restore();
        self.tableau.restore();
        self.bindings.restore();
        self.constraints.restore();
        self.disequation_solver.restore();
        self.equation_solver.restore();
        self.ordering_solver.restore();
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.tableau.is_empty()
    }

    pub(crate) fn apply_rule(&mut self, rule: Rule) -> Option<Clause> {
        self.tableau.apply_rule(
            &self.problem,
            &mut self.terms,
            &mut self.constraints,
            rule,
        )
    }

    pub(crate) fn possible_rules<E: Extend<Rule>>(&self, possible: &mut E) {
        self.tableau.possible_rules(
            possible,
            self.problem,
            &self.terms,
            &self.bindings,
        );
    }

    pub(crate) fn simplify_constraints(&mut self) -> bool {
        self.equation_solver.solve::<SkipCheck, _>(
            &self.problem.symbols,
            &self.terms,
            &mut self.bindings,
            self.constraints.drain_equations(),
        ) && self.disequation_solver.simplify(
            &self.problem.symbols,
            &self.terms,
            &self.bindings,
            self.constraints.drain_disequations(),
        ) && self.disequation_solver.simplify_symmetric(
            &self.problem.symbols,
            &self.terms,
            &self.bindings,
            self.constraints.drain_symmetric_disequations(),
        ) && self.ordering_solver.simplify(
            &self.problem.symbols,
            &self.terms,
            &self.bindings,
            self.constraints.drain_orderings(),
        )
    }

    pub(crate) fn solve_constraints(&mut self) -> bool {
        self.equation_solver.solve::<Check, _>(
            &self.problem.symbols,
            &self.terms,
            &mut self.bindings,
            self.constraints.drain_equations(),
        ) && self.disequation_solver.simplify(
            &self.problem.symbols,
            &self.terms,
            &self.bindings,
            self.constraints.drain_disequations(),
        ) && self.disequation_solver.simplify_symmetric(
            &self.problem.symbols,
            &self.terms,
            &self.bindings,
            self.constraints.drain_symmetric_disequations(),
        ) && self.disequation_solver.check(
            &self.problem.symbols,
            &self.terms,
            &self.bindings,
        ) && self.ordering_solver.simplify(
            &self.problem.symbols,
            &self.terms,
            &self.bindings,
            self.constraints.drain_orderings(),
        ) && self.ordering_solver.check(
            &self.problem.symbols,
            &self.terms,
            &self.bindings,
        )
    }

    pub(crate) fn graph(&mut self, graph: &mut Graph, rules: &[Rule]) {
        self.tableau.graph(
            graph,
            &self.problem,
            &mut self.terms,
            &mut self.bindings,
            rules,
        );
    }
}
