use crate::prelude::*;

#[derive(Clone)]
pub struct Clause {
    pub literals: Vec<Literal>,
}

impl Clause {
    pub fn new(literals: Vec<Literal>) -> Self {
        Self { literals }
    }

    pub fn offset(&mut self, offset: Offset<Term>) {
        for literal in &mut self.literals {
            literal.offset(offset);
        }
    }

    pub fn extend<T: Iterator<Item = Literal>>(&mut self, t: T) {
        self.literals.extend(t);
    }

    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
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
