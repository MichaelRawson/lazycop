use crate::prelude::*;
use std::ops::Index;

#[derive(Clone, Copy)]
pub struct Clause {
    range: IdRange<Literal>,
}

impl Clause {
    pub fn is_empty(mut self) -> bool {
        self.range.next().is_none()
    }

    pub fn len(self) -> u32 {
        self.range.len()
    }

    pub fn literals<'storage>(
        self,
        storage: &'storage ClauseStorage,
    ) -> impl Iterator<Item = Literal> + 'storage {
        self.range.map(move |id| storage[id])
    }

    pub fn current_literal(
        mut self,
        storage: &ClauseStorage,
    ) -> Option<Literal> {
        self.range.next().map(|id| storage[id])
    }

    pub fn pop_literal(&mut self, storage: &ClauseStorage) -> Option<Literal> {
        self.range.next().map(|id| storage[id])
    }
}

#[derive(Default)]
pub struct ClauseStorage {
    arena: Arena<Literal>,
    mark: Id<Literal>,
}

impl ClauseStorage {
    pub fn clear(&mut self) {
        self.arena.clear();
    }

    pub fn copy(
        &mut self,
        offset: Offset<Term>,
        from: &Arena<Literal>,
    ) -> Clause {
        let start = self.arena.len();
        for id in from {
            self.arena.push(from[id].offset(offset));
        }
        let stop = self.arena.len();
        let range = IdRange::new(start, stop);
        Clause { range }
    }

    pub fn copy_replace(
        &mut self,
        offset: Offset<Term>,
        from: &Arena<Literal>,
        except: Id<Literal>,
        with: Literal,
    ) -> Clause {
        let start = self.arena.len();
        self.arena.push(with);
        for id in from {
            if id != except {
                self.arena.push(from[id].offset(offset));
            }
        }
        let stop = self.arena.len();
        let range = IdRange::new(start, stop);
        Clause { range }
    }

    pub fn mark(&mut self) {
        self.mark = self.arena.len();
    }

    pub fn undo_to_mark(&mut self) {
        self.arena.truncate(self.mark);
    }
}

impl Index<Id<Literal>> for ClauseStorage {
    type Output = Literal;

    fn index(&self, id: Id<Literal>) -> &Self::Output {
        &self.arena[id]
    }
}
