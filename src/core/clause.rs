use crate::prelude::*;
use std::ops::Index;

#[derive(Clone, Copy)]
pub(crate) struct Clause {
    start: Id<Literal>,
    current: Id<Literal>,
    end: Id<Literal>,
}

impl Clause {
    pub(crate) fn is_empty(&self) -> bool {
        self.current == self.end
    }

    pub(crate) fn len(&self) -> u32 {
        IdRange::new(self.current, self.end).len()
    }

    pub(crate) fn closed(&self) -> impl Iterator<Item = Id<Literal>> {
        IdRange::new(self.start, self.current)
    }

    pub(crate) fn open(&self) -> impl Iterator<Item = Id<Literal>> {
        IdRange::new(self.current, self.end)
    }

    pub(crate) fn current_literal(&self) -> Id<Literal> {
        self.current
    }

    pub(crate) fn close_literal(&mut self) -> Option<Id<Literal>> {
        let result = self.open().next();
        self.current.increment();
        result
    }
}

#[derive(Default)]
pub(crate) struct ClauseStorage {
    literals: Arena<Literal>,
    mark: Id<Literal>,
}

impl ClauseStorage {
    pub(crate) fn clear(&mut self) {
        self.literals.clear();
    }

    pub(crate) fn create_clause<T: IntoIterator<Item = Literal>>(
        &mut self,
        literals: T,
    ) -> Clause {
        let start = self.literals.limit();
        self.literals.extend(literals);
        let end = self.literals.limit();
        let current = start;
        Clause {
            start,
            current,
            end,
        }
    }

    pub(crate) fn create_clause_with<T: IntoIterator<Item = Literal>>(
        &mut self,
        with: Literal,
        literals: T,
    ) -> Clause {
        let start = self.literals.limit();
        self.literals.push(with);
        self.literals.extend(literals);
        let end = self.literals.limit();
        let current = start;
        Clause {
            start,
            current,
            end,
        }
    }

    pub(crate) fn mark(&mut self) {
        self.mark = self.literals.limit();
    }

    pub(crate) fn undo_to_mark(&mut self) {
        self.literals.truncate(self.mark);
    }
}

impl Index<Id<Literal>> for ClauseStorage {
    type Output = Literal;

    fn index(&self, id: Id<Literal>) -> &Self::Output {
        &self.literals[id]
    }
}

impl Extend<Literal> for ClauseStorage {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = Literal>,
    {
        self.literals.extend(iter);
    }
}
