use crate::prelude::*;

#[derive(Clone)]
pub struct Clause {
    pub literals: Vec<Literal>,
}

impl Clause {
    pub fn new(literals: Vec<Literal>) -> Self {
        Self { literals }
    }

    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    pub fn offset(&mut self, offset: Offset<Term>) {
        for literal in &mut self.literals {
            literal.offset(offset);
        }
    }
}
