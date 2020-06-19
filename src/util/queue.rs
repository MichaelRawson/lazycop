use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Priority {
    pub estimate: u16,
    pub precedence: u16,
}

struct Item<T> {
    item: T,
    priority: Priority,
}

impl<T> PartialEq for Item<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl<T> Eq for Item<T> {}

impl<T> PartialOrd for Item<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Item<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.priority.cmp(&self.priority)
    }
}

pub struct Queue<T> {
    heap: BinaryHeap<Item<T>>,
}

impl<T> Default for Queue<T> {
    fn default() -> Self {
        let heap = BinaryHeap::default();
        Self { heap }
    }
}

impl<T> Queue<T> {
    pub fn enqueue(&mut self, item: T, priority: Priority) {
        self.heap.push(Item { item, priority });
    }

    pub fn dequeue(&mut self) -> Option<T> {
        self.heap.pop().map(|item| item.item)
    }
}
