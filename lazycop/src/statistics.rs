use lazy::problem::Problem;
use lazy::record::Record;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

pub struct Statistics {
    problem_equality: bool,
    problem_symbols: usize,
    problem_clauses: usize,
    start_clauses: usize,
    discarded_tableaux: AtomicUsize,
    enqueued_tableaux: AtomicUsize,
    expanded_tableaux: AtomicUsize,
    exhausted_tableaux: AtomicUsize,
    total_tableaux: AtomicUsize,
}

impl Statistics {
    pub fn new(problem: &Problem) -> Self {
        let problem_equality = problem.has_equality();
        let problem_symbols = problem.signature().len().as_usize();
        let problem_clauses = problem.num_clauses();
        let start_clauses = problem.num_start_clauses();
        let discarded_tableaux = AtomicUsize::default();
        let enqueued_tableaux = AtomicUsize::default();
        let expanded_tableaux = AtomicUsize::default();
        let exhausted_tableaux = AtomicUsize::default();
        let total_tableaux = AtomicUsize::default();
        Self {
            problem_equality,
            problem_symbols,
            problem_clauses,
            start_clauses,
            discarded_tableaux,
            enqueued_tableaux,
            expanded_tableaux,
            exhausted_tableaux,
            total_tableaux,
        }
    }

    pub fn record<R: Record>(&self, record: &mut R) {
        record.statistic("equality problem", self.problem_equality);
        record.statistic("problem symbols", self.problem_symbols);
        record.statistic("problem clauses", self.problem_clauses);
        record.statistic("start clauses", self.start_clauses);
        record.statistic(
            "discarded tableaux",
            self.discarded_tableaux.load(Relaxed),
        );
        record.statistic(
            "enqueued tableaux",
            self.enqueued_tableaux.load(Relaxed),
        );
        record.statistic(
            "expanded tableaux",
            self.expanded_tableaux.load(Relaxed),
        );
        record.statistic(
            "exhausted tableaux",
            self.exhausted_tableaux.load(Relaxed),
        );
        record.statistic("total tableaux", self.total_tableaux.load(Relaxed));
    }

    pub fn increment_discarded_tableaux(&self) {
        self.discarded_tableaux.fetch_add(1, Relaxed);
    }

    pub fn increment_enqueued_tableaux(&self) {
        self.enqueued_tableaux.fetch_add(1, Relaxed);
    }

    pub fn increment_expanded_tableaux(&self) {
        self.expanded_tableaux.fetch_add(1, Relaxed);
    }

    pub fn exhausted_tableaux(&self, exhausted: u16) {
        self.exhausted_tableaux
            .fetch_add(exhausted as usize, Relaxed);
    }

    pub fn increment_total_tableaux(&self) {
        self.total_tableaux.fetch_add(1, Relaxed);
    }
}
