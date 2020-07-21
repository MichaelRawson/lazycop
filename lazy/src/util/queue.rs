use std::cmp::Ordering;
use std::collections::BinaryHeap;

struct Item<K, V> {
    priority: K,
    payload: V,
}

impl<K: Eq, V> PartialEq for Item<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl<K: Eq, V> Eq for Item<K, V> {}

impl<K: Ord, V> PartialOrd for Item<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord, V> Ord for Item<K, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.priority.cmp(&self.priority)
    }
}

pub struct Queue<K, V> {
    heap: BinaryHeap<Item<K, V>>,
}

impl<K: Ord, V> Default for Queue<K, V> {
    fn default() -> Self {
        let heap = BinaryHeap::default();
        Self { heap }
    }
}

impl<K: Ord, V> Queue<K, V> {
    pub fn enqueue(&mut self, priority: K, payload: V) {
        self.heap.push(Item { priority, payload });
    }

    pub fn dequeue(&mut self) -> Option<(K, V)> {
        self.heap.pop().map(|item| (item.priority, item.payload))
    }
}
