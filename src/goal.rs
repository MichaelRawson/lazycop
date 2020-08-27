use crate::binding::Bindings;
use crate::constraint::Constraints;
use crate::disequation_solver::DisequationSolver;
use crate::equation_solver::EquationSolver;
use crate::occurs::{Check, SkipCheck};
use crate::prelude::*;
use crate::record::Record;
use crate::tableau::Tableau;

pub(crate) struct Goal<'problem> {
    problem: &'problem Problem,
    terms: Terms,
    tableau: Tableau,
    bindings: Bindings,
    constraints: Constraints,
    disequation_solver: DisequationSolver,
    equation_solver: EquationSolver,
}

impl<'problem> Goal<'problem> {
    pub(crate) fn new(problem: &'problem Problem) -> Self {
        let terms = Terms::default();
        let tableau = Tableau::default();
        let bindings = Bindings::default();
        let constraints = Constraints::default();
        let disequation_solver = DisequationSolver::default();
        let equation_solver = EquationSolver::default();
        Self {
            problem,
            terms,
            tableau,
            bindings,
            constraints,
            disequation_solver,
            equation_solver,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.terms.clear();
        self.tableau.clear();
        self.bindings.clear();
        self.constraints.clear();
        self.disequation_solver.clear();
        self.equation_solver.clear();
    }

    pub(crate) fn save(&mut self) {
        self.terms.save();
        self.tableau.save();
        self.bindings.save();
        self.constraints.save();
        self.disequation_solver.save();
        self.equation_solver.save();
    }

    pub(crate) fn restore(&mut self) {
        self.terms.restore();
        self.tableau.restore();
        self.bindings.restore();
        self.constraints.restore();
        self.disequation_solver.restore();
        self.equation_solver.restore();
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.tableau.is_empty()
    }

    pub(crate) fn num_open_branches(&self) -> u32 {
        self.tableau.num_open_branches()
    }

    pub(crate) fn apply_rule<R: Record>(
        &mut self,
        record: &mut R,
        rule: &Rule,
    ) {
        self.tableau.apply_rule(
            record,
            &self.problem,
            &mut self.terms,
            &mut self.constraints,
            rule,
        );
    }

    pub(crate) fn possible_rules<E: Extend<Rule>>(&self, possible: &mut E) {
        self.tableau
            .possible_rules(possible, self.problem, &self.terms);
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
        )
    }

    pub(crate) fn record_unification<R: Record>(&mut self, record: &mut R) {
        record.unification(
            &self.problem.symbols,
            &self.terms,
            self.bindings.items(),
        );
    }

    pub(crate) fn graph(&mut self, graph: &mut Graph) {
        graph.initialise(&self.problem.symbols, &self.terms);
        self.tableau.graph(
            graph,
            &self.problem.symbols,
            &self.terms,
            &self.bindings,
        );
    }
}
