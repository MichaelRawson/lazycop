use crate::goal::Goal;
use crate::prelude::*;
use crate::record::Record;
use crate::solver::Solver;

pub(crate) struct Tableau<'problem> {
    problem: &'problem Problem,
    terms: Terms,
    goal: Goal,
    solver: Solver,
}

impl<'problem> Tableau<'problem> {
    pub(crate) fn new(problem: &'problem Problem) -> Self {
        let terms = Terms::default();
        let solver = Solver::default();
        let goal = Goal::default();
        Self {
            problem,
            goal,
            terms,
            solver,
        }
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.goal.is_empty()
    }

    pub(crate) fn num_open_branches(&self) -> u32 {
        self.goal.num_open_branches()
    }

    pub(crate) fn clear(&mut self) {
        self.terms.as_mut().clear();
        self.goal.clear();
        self.solver.clear();
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
            &mut self.solver,
            rule,
        );
    }

    pub(crate) fn possible_rules<E: Extend<Rule>>(&self, possible: &mut E) {
        self.goal
            .possible_rules(possible, &self.problem, &self.terms);
    }

    pub(crate) fn solve_constraints_fast(&mut self) -> bool {
        self.solver
            .solve_fast(self.problem.signature(), &self.terms)
    }

    pub(crate) fn solve_constraints_correct(&mut self) -> bool {
        self.solver
            .solve_correct(self.problem.signature(), &self.terms)
    }

    pub(crate) fn record_unification<R: Record>(&mut self, record: &mut R) {
        record.unification(
            &self.problem.signature(),
            &self.terms,
            self.solver.bindings(),
        );
    }
}

impl<'problem> Clone for Tableau<'problem> {
    fn clone(&self) -> Self {
        unimplemented!()
    }

    fn clone_from(&mut self, other: &Self) {
        self.terms.clone_from(&other.terms);
        self.goal.clone_from(&other.goal);
        self.solver.clone_from(&other.solver);
    }
}
