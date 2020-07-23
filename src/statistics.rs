use crate::problem::Problem;
use crate::record::Record;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

pub(crate) struct Statistics {
    problem_equality: bool,
    problem_symbols: u32,
    problem_clauses: u32,
    start_clauses: u32,
    discarded_goals: AtomicUsize,
    enqueued_goals: AtomicUsize,
    expanded_goals: AtomicUsize,
    exhausted_goals: AtomicUsize,
    total_goals: AtomicUsize,
}

impl Statistics {
    pub(crate) fn new(problem: &Problem) -> Self {
        let problem_equality = problem.has_equality;
        let problem_symbols = problem.num_symbols();
        let problem_clauses = problem.num_clauses();
        let start_clauses = problem.num_start_clauses();
        let discarded_goals = AtomicUsize::default();
        let enqueued_goals = AtomicUsize::default();
        let expanded_goals = AtomicUsize::default();
        let exhausted_goals = AtomicUsize::default();
        let total_goals = AtomicUsize::default();
        Self {
            problem_equality,
            problem_symbols,
            problem_clauses,
            start_clauses,
            discarded_goals,
            enqueued_goals,
            expanded_goals,
            exhausted_goals,
            total_goals,
        }
    }

    pub(crate) fn record<R: Record>(&self, record: &mut R) {
        record.statistic("equality problem", self.problem_equality);
        record.statistic("problem symbols", self.problem_symbols);
        record.statistic("problem clauses", self.problem_clauses);
        record.statistic("start clauses", self.start_clauses);
        record
            .statistic("discarded goals", self.discarded_goals.load(Relaxed));
        record.statistic("enqueued goals", self.enqueued_goals.load(Relaxed));
        record.statistic("expanded goals", self.expanded_goals.load(Relaxed));
        record
            .statistic("exhausted goals", self.exhausted_goals.load(Relaxed));
        record.statistic("total goals", self.total_goals.load(Relaxed));
    }

    pub(crate) fn increment_discarded_goals(&self) -> usize {
        self.discarded_goals.fetch_add(1, Relaxed)
    }

    pub(crate) fn increment_enqueued_goals(&self) -> usize {
        self.enqueued_goals.fetch_add(1, Relaxed)
    }

    pub(crate) fn increment_expanded_goals(&self) -> usize {
        self.expanded_goals.fetch_add(1, Relaxed)
    }

    pub(crate) fn exhausted_goals(&self, exhausted: u16) -> usize {
        self.exhausted_goals.fetch_add(exhausted as usize, Relaxed)
    }

    pub(crate) fn increment_total_goals(&self) -> usize {
        self.total_goals.fetch_add(1, Relaxed)
    }
}
