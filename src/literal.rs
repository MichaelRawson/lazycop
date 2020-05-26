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

    pub(crate) fn offset(&mut self, offset: Offset<Term>) {
        self.atom.offset(offset);
    }
}
