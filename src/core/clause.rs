use crate::prelude::*;
use std::ops::Index;

#[derive(Clone, Copy)]
pub(crate) struct Clause {
    range: IdRange<Literal>,
}

impl Clause {
    pub(crate) fn is_empty(mut self) -> bool {
        self.range.next().is_none()
    }

    pub(crate) fn len(self) -> u32 {
        self.range.len()
    }

    pub(crate) fn peek_rest(self) -> Clause {
        let mut range = self.range;
        range.next();
        Self { range }
    }

    pub(crate) fn literals<'storage>(
        self,
        storage: &'storage ClauseStorage,
    ) -> impl Iterator<Item = Literal> + 'storage {
        self.range.map(move |id| storage[id])
    }

    pub(crate) fn current_literal(
        mut self,
        storage: &ClauseStorage,
    ) -> Option<Literal> {
        self.range.next().map(|id| storage[id])
    }

    pub(crate) fn pop_literal(
        &mut self,
        storage: &ClauseStorage,
    ) -> Option<Literal> {
        self.range.next().map(|id| storage[id])
    }
}

#[derive(Default)]
pub(crate) struct ClauseStorage {
    arena: Arena<Literal>,
    mark: Id<Literal>,
}

impl ClauseStorage {
    pub(crate) fn clear(&mut self) {
        self.arena.clear();
    }

    pub(crate) fn copy(
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

    pub(crate) fn copy_replace(
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

    pub(crate) fn mark(&mut self) {
        self.mark = self.arena.len();
    }

    pub(crate) fn undo_to_mark(&mut self) {
        self.arena.truncate(self.mark);
    }
}

impl Index<Id<Literal>> for ClauseStorage {
    type Output = Literal;

    fn index(&self, id: Id<Literal>) -> &Self::Output {
        &self.arena[id]
    }
}
