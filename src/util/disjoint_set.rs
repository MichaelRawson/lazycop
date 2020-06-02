use crate::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct Set {
    parent: Id<Set>,
    rank: u32,
}

#[derive(Default)]
pub(crate) struct Disjoint {
    sets: Block<Set>,
}

impl Disjoint {
    pub(crate) fn len(&self) -> Id<Set> {
        self.sets.len()
    }

    pub(crate) fn clear(&mut self) {
        self.sets.clear();
    }

    pub(crate) fn copy_from(&mut self, other: &Self) {
        self.sets.copy_from(&other.sets);
    }

    pub(crate) fn singleton(&mut self) -> Id<Set> {
        let parent = self.sets.len();
        let rank = 0;
        let set = Set { parent, rank };
        self.sets.push(set)
    }

    pub(crate) fn find(&mut self, mut current: Id<Set>) -> Id<Set> {
        while self.sets[current].parent != current {
            let parent = self.sets[current].parent;
            let grandparent = self.sets[parent].parent;
            self.sets[current].parent = grandparent;
            current = grandparent;
        }
        current
    }

    pub(crate) fn merge(&mut self, left: Id<Set>, right: Id<Set>) {
        let left_rank = self.sets[left].rank;
        let right_rank = self.sets[right].rank;
        if left_rank > right_rank {
            self.sets[right].parent = left;
        } else {
            self.sets[left].parent = right;
            if left_rank == right_rank {
                self.sets[right].rank += 1;
            }
        }
    }
}

impl Clone for Disjoint {
    fn clone(&self) -> Self {
        unimplemented!()
    }

}
