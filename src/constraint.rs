use crate::prelude::*;

pub(crate) struct SymmetricDisequation {
    pub(crate) left1: Id<Term>,
    pub(crate) left2: Id<Term>,
    pub(crate) right1: Id<Term>,
    pub(crate) right2: Id<Term>,
}

#[derive(Default)]
pub(crate) struct Constraints {
    equations: Vec<(Id<Term>, Id<Term>)>,
    disequations: Vec<(Id<Term>, Id<Term>)>,
    symmetric_disequations: Vec<SymmetricDisequation>,
    orderings: Vec<(Id<Term>, Id<Term>)>,
}

impl Constraints {
    pub(crate) fn clear(&mut self) {
        self.equations.clear();
        self.disequations.clear();
        self.symmetric_disequations.clear();
        self.orderings.clear();
    }

    pub(crate) fn save(&mut self) {}

    pub(crate) fn restore(&mut self) {
        self.equations.clear();
        self.disequations.clear();
        self.symmetric_disequations.clear();
        self.orderings.clear();
    }

    pub(crate) fn assert_eq(&mut self, left: Id<Term>, right: Id<Term>) {
        self.equations.push((left, right));
    }

    pub(crate) fn assert_neq(&mut self, left: Id<Term>, right: Id<Term>) {
        self.disequations.push((left, right));
    }

    pub(crate) fn assert_symmetric_neq(&mut self, item: SymmetricDisequation) {
        self.symmetric_disequations.push(item);
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

    pub(crate) fn drain_symmetric_disequations(
        &mut self,
    ) -> impl Iterator<Item = SymmetricDisequation> + '_ {
        self.symmetric_disequations.drain(..)
    }

    pub(crate) fn drain_orderings(
        &mut self,
    ) -> impl Iterator<Item = (Id<Term>, Id<Term>)> + '_ {
        self.orderings.drain(..)
    }
}
