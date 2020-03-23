use crate::prelude::*;
use crate::search::script::Script;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

#[derive(Clone)]
struct Item {
    priority: u32,
    script: Rc<Script>,
}

impl PartialEq for Item {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for Item {}

impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.priority.partial_cmp(&other.priority)
    }
}

impl Ord for Item {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

#[derive(Default)]
pub struct Queue {
    heap: BinaryHeap<Reverse<Item>>,
}

impl Queue {
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    pub fn enqueue(&mut self, script: Rc<Script>, priority: u32) {
        self.heap.push(Reverse(Item { priority, script }));
    }

    pub fn dequeue(&mut self) -> Option<Rc<Script>> {
        let Reverse(item) = self.heap.pop()?;
        Some(item.script)
    }
}
