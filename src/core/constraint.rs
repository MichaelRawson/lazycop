use crate::prelude::*;

#[derive(Default)]
pub struct ConstraintList {
    pub equalities: Vec<(Id<Term>, Id<Term>)>,
    equality_mark: usize,
    pub disequalities: Vec<(Id<Term>, Id<Term>)>,
    disequality_mark: usize,
}

impl ConstraintList {
    pub fn add_equality(&mut self, left: Id<Term>, right: Id<Term>) {
        self.equalities.push((left, right));
    }

    pub fn add_disequality(&mut self, left: Id<Term>, right: Id<Term>) {
        self.disequalities.push((left, right));
    }

    pub fn clear(&mut self) {
        self.equalities.clear();
        self.equality_mark = 0;
        self.disequalities.clear();
        self.disequality_mark = 0;
    }

    pub fn mark(&mut self) {
        self.equality_mark = self.equalities.len();
        self.disequality_mark = self.disequalities.len();
    }

    pub fn undo_to_mark(&mut self) {
        self.equalities.truncate(self.equality_mark);
        self.disequalities.truncate(self.disequality_mark);
    }
}
