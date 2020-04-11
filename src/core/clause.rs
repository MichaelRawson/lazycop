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
        let stop = self.literals.len();
        let range = IdRange::new(start, stop);
        Clause { range }
    }

    pub(crate) fn clause_with<T: IntoIterator<Item = Literal>>(
        &mut self,
        literals: T,
        with: Literal,
    ) -> Clause {
        let start = self.literals.len();
        self.literals.push(with);
        self.literals.extend(literals);
        let stop = self.literals.len();
        let range = IdRange::new(start, stop);
        Clause { range }
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
