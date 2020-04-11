use std::cmp::Ordering;
use std::collections::BinaryHeap;

struct Item<T> {
    item: T,
    priority: u32,
}

impl<T> PartialEq for Item<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl<T> Eq for Item<T> {}

impl<T> PartialOrd for Item<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.priority.partial_cmp(&self.priority)
    }
}

impl<T> Ord for Item<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.priority.cmp(&self.priority)
    }
}

pub(crate) struct Queue<T> {
    heap: BinaryHeap<Item<T>>,
}

impl<T> Default for Queue<T> {
    fn default() -> Self {
        let heap = BinaryHeap::default();
        Self { heap }
    }
}

impl<T> Queue<T> {
    pub(crate) fn clear(&mut self) {
        self.heap.clear();
    }

    pub(crate) fn enqueue(&mut self, item: T, priority: u32) {
        self.heap.push(Item { item, priority });
    }

    pub(crate) fn dequeue(&mut self) -> Option<T> {
        Some(self.heap.pop()?.item)
    }
}
