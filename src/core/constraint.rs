use crate::prelude::*;

#[derive(Default)]
pub(crate) struct ConstraintList {
    pub(crate) equalities: Vec<(Id<Term>, Id<Term>)>,
    equality_mark: usize,
}

impl ConstraintList {
    pub(crate) fn add_equality(&mut self, left: Id<Term>, right: Id<Term>) {
        self.equalities.push((left, right));
    }

    pub(crate) fn clear(&mut self) {
        self.equalities.clear();
        self.equality_mark = 0;
    }

    pub(crate) fn mark(&mut self) {
        self.equality_mark = self.equalities.len();
    }

    pub(crate) fn undo_to_mark(&mut self) {
        self.equalities.truncate(self.equality_mark);
    }
}
