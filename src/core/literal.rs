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

    pub fn offset(&self, offset: Offset<Term>) -> Self {
        let polarity = self.polarity;
        let atom = self.atom.offset(offset);
        Self { polarity, atom }
    }
}
