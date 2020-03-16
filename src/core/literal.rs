use crate::prelude::*;

#[derive(Clone, Copy)]
pub struct Literal {
    polarity: bool,
    atom: Atom,
}

impl Literal {
    pub fn new(polarity: bool, atom: Atom) -> Self {
        Self { polarity, atom }
    }
}
