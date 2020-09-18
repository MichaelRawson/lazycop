use crate::prelude::*;

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub(crate) struct Heuristic(f32);

impl Heuristic {
    pub(crate) fn new(value: f32) -> Self {
        debug_assert!(value.is_finite());
        debug_assert!(value >= 0.0);
        Self(value)
    }
}

impl Eq for Heuristic {}

impl Ord for Heuristic {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        some(self.partial_cmp(other))
    }
}
