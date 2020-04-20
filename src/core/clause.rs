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

    pub(crate) fn closed(&self) -> IdRange<Literal> {
        IdRange::new(self.start, self.current)
    }

    pub(crate) fn open(&self) -> IdRange<Literal> {
        IdRange::new(self.current, self.end)
    }

    pub(crate) fn len(&self) -> u32 {
        self.open().len()
    }

    pub(crate) fn current_literal(&self) -> Id<Literal> {
        self.current
    }

    pub(crate) fn pop_literal(&mut self) -> Option<Id<Literal>> {
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

    pub(crate) fn clause<T: IntoIterator<Item = Literal>>(
        &mut self,
        literals: T,
    ) -> Clause {
        let start = self.literals.len();
        self.literals.extend(literals);
        let end = self.literals.len();
        let current = start;
        Clause {
            start,
            current,
            end,
        }
    }

    pub(crate) fn clause_with<T: IntoIterator<Item = Literal>>(
        &mut self,
        literals: T,
        with: Literal,
    ) -> Clause {
        let start = self.literals.len();
        self.literals.push(with);
        self.literals.extend(literals);
        let end = self.literals.len();
        let current = start;
        Clause {
            start,
            current,
            end,
        }
    }

    pub(crate) fn mark(&mut self) {
        self.mark = self.literals.len();
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
