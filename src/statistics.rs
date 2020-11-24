use crate::problem::Problem;
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Default)]
pub(crate) struct Statistics {
    pub(crate) problem_symbols: u32,
    pub(crate) problem_clauses: u32,
    pub(crate) eliminated_leaves: AtomicU32,
    pub(crate) retained_leaves: AtomicU32,
    pub(crate) expanded_leaves: AtomicU32,
    #[cfg(feature = "smt")]
    pub(crate) smt_assertions: AtomicU32,
    #[cfg(feature = "smt")]
    pub(crate) smt_checks: AtomicU32,
    #[cfg(feature = "nn")]
    pub(crate) evaluated_leaves: AtomicU32,
}

impl Statistics {
    pub(crate) fn new(problem: &Problem) -> Self {
        let mut new = Self::default();
        new.problem_symbols = problem.symbols.len().index();
        new.problem_clauses = problem.clauses.len().index();
        new
    }

    pub(crate) fn increment_eliminated_leaves(&self) {
        self.eliminated_leaves.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn load_eliminated_leaves(&self) -> u32 {
        self.eliminated_leaves.load(Ordering::Relaxed)
    }

    pub(crate) fn increment_retained_leaves(&self) {
        self.retained_leaves.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn load_retained_leaves(&self) -> u32 {
        self.retained_leaves.load(Ordering::Relaxed)
    }

    pub(crate) fn increment_expanded_leaves(&self) {
        self.expanded_leaves.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn load_expanded_leaves(&self) -> u32 {
        self.expanded_leaves.load(Ordering::Relaxed)
    }

    #[cfg(feature = "smt")]
    pub(crate) fn increment_smt_assertions(&self) {
        self.smt_assertions.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(feature = "smt")]
    pub(crate) fn load_smt_assertions(&self) -> u32 {
        self.smt_assertions.load(Ordering::Relaxed)
    }

    #[cfg(feature = "smt")]
    pub(crate) fn increment_smt_checks(&self) {
        self.smt_checks.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(feature = "smt")]
    pub(crate) fn load_smt_checks(&self) -> u32 {
        self.smt_checks.load(Ordering::Relaxed)
    }

    #[cfg(feature = "nn")]
    pub(crate) fn increment_evaluated_leaves(&self) {
        self.evaluated_leaves.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(feature = "nn")]
    pub(crate) fn load_evaluated_leaves(&self) -> u32 {
        self.evaluated_leaves.load(Ordering::Relaxed)
    }
}
