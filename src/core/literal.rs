use crate::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct Literal {
    pub(crate) polarity: bool,
    pub(crate) atom: Atom,
}

impl Literal {
    pub(crate) fn new(polarity: bool, atom: Atom) -> Self {
        Self { polarity, atom }
    }

    pub(crate) fn offset(&self, offset: Offset<Term>) -> Self {
        let polarity = self.polarity;
        let atom = self.atom.offset(offset);
        Self { polarity, atom }
    }
}
