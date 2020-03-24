use crate::prelude::*;
use std::ops::Index;

#[derive(Clone)]
pub struct Clause {
    literals: Vec<Literal>,
}

impl Clause {
    pub fn new(literals: Vec<Literal>) -> Self {
        Self { literals }
    }

    pub fn offset(&mut self, offset: Offset<Term>) {
        for literal in &mut *self.literals {
            literal.offset(offset);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Literal> {
        self.literals.iter()
    }

    pub fn len(&self) -> usize {
        self.literals.len()
    }

    pub fn last_literal(&self) -> &Literal {
        self.literals.last().unwrap()
    }

    pub fn pop_literal(&mut self) -> Literal {
        self.literals.pop().unwrap()
    }

    pub fn remove_literal(&mut self, literal_id: Id<Literal>) -> Literal {
        self.literals.remove(literal_id.index())
    }
}

impl Index<Id<Literal>> for Clause {
    type Output = Literal;

    fn index(&self, id: Id<Literal>) -> &Self::Output {
        &self.literals[id.index()]
    }
}
