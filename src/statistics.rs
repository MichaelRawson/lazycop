use crate::problem::Problem;
use crate::record::Record;
use std::sync::atomic::{AtomicU32, Ordering};

pub(crate) struct Statistics {
    total_symbols: u32,
    total_clauses: u32,
    eliminated_leaves: AtomicU32,
    retained_leaves: AtomicU32,
    expanded_leaves: AtomicU32,
    #[cfg(feature = "cudann")]
    evaluated_leaves: AtomicU32,
}

impl Statistics {
    pub(crate) fn new(problem: &Problem) -> Self {
        let total_symbols = problem.num_symbols();
        let total_clauses = problem.num_clauses();
        let eliminated_leaves = AtomicU32::default();
        let retained_leaves = AtomicU32::default();
        let expanded_leaves = AtomicU32::default();
        #[cfg(feature = "cudann")]
        let evaluated_leaves = AtomicU32::default();
        Self {
            total_symbols,
            total_clauses,
            eliminated_leaves,
            retained_leaves,
            expanded_leaves,
            #[cfg(feature = "cudann")]
            evaluated_leaves,
        }
    }

    pub(crate) fn record<R: Record>(&self, record: &mut R) {
        record.statistic("total symbols", self.total_symbols);
        record.statistic("total clauses", self.total_clauses);
        record.statistic(
            "eliminated leaves",
            self.eliminated_leaves.load(Ordering::Relaxed),
        );
        record.statistic(
            "retained leaves",
            self.retained_leaves.load(Ordering::Relaxed),
        );
        record.statistic(
            "expanded leaves",
            self.expanded_leaves.load(Ordering::Relaxed),
        );
        #[cfg(feature = "cudann")]
        record.statistic(
            "evaluated leaves",
            self.evaluated_leaves.load(Ordering::Relaxed),
        );
    }

    pub(crate) fn increment_eliminated_leaves(&self) {
        self.eliminated_leaves.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn increment_retained_leaves(&self) {
        self.retained_leaves.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn increment_expanded_leaves(&self) {
        self.expanded_leaves.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(feature = "cudann")]
    pub(crate) fn increment_evaluated_leaves(&self) {
        self.evaluated_leaves.fetch_add(1, Ordering::Relaxed);
    }
}
