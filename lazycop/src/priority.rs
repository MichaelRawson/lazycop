use lazy::prelude::*;
use std::cmp::Ordering;

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct Priority {
    value: f32
}

impl Priority {
    pub fn new(value: f32) -> Self {
        debug_assert!(value >= 0.0 && value < std::f32::MAX);
        Self { value }
    }
}

impl Eq for Priority {}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        some(self.partial_cmp(other))
    }
}
