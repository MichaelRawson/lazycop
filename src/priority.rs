use crate::prelude::*;
use std::cmp::Ordering;

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub(crate) struct Priority {
    value: f32,
}

impl Priority {
    pub(crate) fn new(value: f32) -> Self {
        debug_assert!(value.is_finite());
        Self { value }
    }
}

impl Eq for Priority {}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        some(self.partial_cmp(other))
    }
}
