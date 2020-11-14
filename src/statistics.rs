use crate::problem::Problem;
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Default)]
pub(crate) struct Statistics {
    pub(crate) total_symbols: u32,
    pub(crate) total_clauses: u32,
    pub(crate) eliminated_leaves: AtomicU32,
    pub(crate) retained_leaves: AtomicU32,
    pub(crate) expanded_leaves: AtomicU32,
    #[cfg(feature = "cudann")]
    pub(crate) evaluated_leaves: AtomicU32,
}

impl Statistics {
    pub(crate) fn new(problem: &Problem) -> Self {
        let mut new = Self::default();
        new.total_symbols = problem.symbols.len().index();
        new.total_clauses = problem.clauses.len().index();
        new
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
