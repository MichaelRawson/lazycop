use crate::prelude::*;

#[derive(Default)]
pub(crate) struct Constraints {
    equations: Vec<(Id<Term>, Id<Term>)>,
    disequations: Vec<(Id<Term>, Id<Term>)>,
    orderings: Vec<(Id<Term>, Id<Term>)>,

    save_orderings: usize,
}

impl Constraints {
    pub(crate) fn clear(&mut self) {
        self.equations.clear();
        self.disequations.clear();
        self.orderings.clear();
    }

    pub(crate) fn save(&mut self) {
        self.save_orderings = self.orderings.len();
    }

    pub(crate) fn restore(&mut self) {
        self.equations.clear();
        self.disequations.clear();
        self.orderings.truncate(self.save_orderings);
    }

    pub(crate) fn assert_eq(&mut self, left: Id<Term>, right: Id<Term>) {
        self.equations.push((left, right));
    }

    pub(crate) fn assert_neq(&mut self, left: Id<Term>, right: Id<Term>) {
        self.disequations.push((left, right));
    }

    pub(crate) fn assert_gt(&mut self, left: Id<Term>, right: Id<Term>) {
        self.orderings.push((left, right));
    }

    pub(crate) fn drain_equations(
        &mut self,
    ) -> impl Iterator<Item = (Id<Term>, Id<Term>)> + '_ {
        self.equations.drain(..)
    }

    pub(crate) fn drain_disequations(
        &mut self,
    ) -> impl Iterator<Item = (Id<Term>, Id<Term>)> + '_ {
        self.disequations.drain(..)
    }

    pub(crate) fn keep_orderings(
        &self,
    ) -> impl Iterator<Item = (Id<Term>, Id<Term>)> + '_ {
        self.orderings.iter().copied()
    }
}
