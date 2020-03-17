use crate::prelude::*;

#[derive(Clone, Copy)]
pub struct Literal {
    pub polarity: bool,
    pub atom: Atom,
}

impl Literal {
    pub fn new(polarity: bool, atom: Atom) -> Self {
        Self { polarity, atom }
    }

    pub fn offset(&mut self, offset: Offset<Term>) {
        self.atom.offset(offset);
    }
}
