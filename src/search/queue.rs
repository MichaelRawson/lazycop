use crate::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Item {
    priority: u32,
    rule: Id<Rule>,
}

#[derive(Default)]
pub struct Queue {
    heap: BinaryHeap<Reverse<Item>>,
}

impl Queue {
    pub fn enqueue(&mut self, rule: Id<Rule>, priority: u32) {
        self.heap.push(Reverse(Item { priority, rule }));
    }

    pub fn dequeue(&mut self) -> Option<Id<Rule>> {
        let Reverse(item) = self.heap.pop()?;
        Some(item.rule)
    }
}
