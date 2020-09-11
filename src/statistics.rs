use crate::problem::Problem;
use crate::record::Record;
use std::sync::atomic::{AtomicU32, Ordering};

pub(crate) struct Statistics {
    problem_equality: bool,
    problem_symbols: u32,
    problem_clauses: u32,
    start_clauses: u32,
    eliminated_goals: AtomicU32,
    retained_goals: AtomicU32,
    expanded_goals: AtomicU32,
    #[cfg(feature = "nn")]
    evaluated_goals: AtomicU32,
}

impl Statistics {
    pub(crate) fn new(problem: &Problem) -> Self {
        let problem_equality = problem.has_equality;
        let problem_symbols = problem.num_symbols();
        let problem_clauses = problem.num_clauses();
        let start_clauses = problem.num_start_clauses();
        let eliminated_goals = AtomicU32::default();
        let retained_goals = AtomicU32::default();
        let expanded_goals = AtomicU32::default();
        #[cfg(feature = "nn")]
        let evaluated_goals = AtomicU32::default();
        Self {
            problem_equality,
            problem_symbols,
            problem_clauses,
            start_clauses,
            eliminated_goals,
            retained_goals,
            expanded_goals,
            #[cfg(feature = "nn")]
            evaluated_goals,
        }
    }

    pub(crate) fn record<R: Record>(&self, record: &mut R) {
        record.statistic("equality problem", self.problem_equality);
        record.statistic("problem symbols", self.problem_symbols);
        record.statistic("problem clauses", self.problem_clauses);
        record.statistic("start clauses", self.start_clauses);
        record.statistic(
            "eliminated goals",
            self.eliminated_goals.load(Ordering::Relaxed),
        );
        record.statistic(
            "retained goals",
            self.retained_goals.load(Ordering::Relaxed),
        );
        record.statistic(
            "expanded goals",
            self.expanded_goals.load(Ordering::Relaxed),
        );
        #[cfg(feature = "nn")]
        record.statistic(
            "evaluated goals",
            self.evaluated_goals.load(Ordering::Relaxed),
        );
    }

    pub(crate) fn increment_eliminated_goals(&self) {
        self.eliminated_goals.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn increment_retained_goals(&self) {
        self.retained_goals.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn increment_expanded_goals(&self) {
        self.expanded_goals.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(feature = "nn")]
    pub(crate) fn increment_evaluated_goals(&self) {
        self.evaluated_goals.fetch_add(1, Ordering::Relaxed);
    }
}
