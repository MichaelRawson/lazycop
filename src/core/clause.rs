use crate::prelude::*;
use std::ops::{Index, IndexMut};

#[derive(Clone, Copy)]
pub(crate) struct Clause {
    literals: Range<Literal>,
}

impl Clause {
    pub(crate) fn is_empty(self) -> bool {
        self.literals.is_empty()
    }

    pub(crate) fn len(self) -> u32 {
        self.literals.len()
    }

    pub(crate) fn open(self) -> Range<Literal> {
        self.literals
    }

    pub(crate) fn pending(self) -> Range<Literal> {
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
    literals: Block<Literal>,
    mark: Id<Literal>,
}

impl ClauseStorage {
    pub(crate) fn len(&self) -> Id<Literal> {
        self.literals.len()
    }

    pub(crate) fn clear(&mut self) {
        self.literals.clear();
    }

    pub(crate) fn create_clause<T: IntoIterator<Item = Literal>>(
        &mut self,
        literals: T,
    ) -> Clause {
        let start = self.literals.len();
        self.literals.extend(literals);
        let end = self.literals.len();
        let literals = Range::new(start, end);
        Clause { literals }
    }

    pub(crate) fn create_clause_with<T: IntoIterator<Item = Literal>>(
        &mut self,
        with: Literal,
        literals: T,
    ) -> Clause {
        let start = self.literals.len();
        self.literals.push(with);
        self.literals.extend(literals);
        let end = self.literals.len();
        let literals = Range::new(start, end);
        Clause { literals }
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

impl IndexMut<Id<Literal>> for ClauseStorage {
    fn index_mut(&mut self, id: Id<Literal>) -> &mut Self::Output {
        &mut self.literals[id]
    }
}
