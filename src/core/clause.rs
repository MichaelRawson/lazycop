use crate::prelude::*;
use std::ops::Index;

#[derive(Clone, Copy)]
pub(crate) struct Clause {
    literals: IdRange<Literal>,
}

impl Clause {
    pub(crate) fn is_empty(self) -> bool {
        self.literals.is_empty()
    }

    pub(crate) fn len(self) -> u32 {
        self.literals.len()
    }

    pub(crate) fn open(self) -> IdRange<Literal> {
        self.literals
    }

    pub(crate) fn pending(self) -> IdRange<Literal> {
        let mut literals = self.literals;
        literals.next();
        literals
    }

    pub(crate) fn current_literal(mut self) -> Id<Literal> {
        self.literals
            .next()
            .expect("current literal of empty clause")
    }

    pub(crate) fn close_literal(&mut self) -> Id<Literal> {
        self.literals
            .next()
            .expect("closing literal of empty clause")
    }
}

#[derive(Default)]
pub(crate) struct ClauseStorage {
    literals: Arena<Literal>,
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
        let literals = IdRange::new(start, end);
        Clause { literals }
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
        let literals = IdRange::new(start, end);
        Clause { literals }
    }

    pub(crate) fn mark(&mut self) {
        self.literals.mark();
    }

    pub(crate) fn undo_to_mark(&mut self) {
        self.literals.undo_to_mark();
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
