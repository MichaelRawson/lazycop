use crate::prelude::*;

pub struct SymmetricDisequation {
    pub left1: Id<Term>,
    pub left2: Id<Term>,
    pub right1: Id<Term>,
    pub right2: Id<Term>,
}

#[derive(Default)]
pub struct Constraints {
    equations: Vec<(Id<Term>, Id<Term>)>,
    disequations: Vec<(Id<Term>, Id<Term>)>,
    symmetric_disequations: Vec<SymmetricDisequation>,
    orderings: Vec<(Id<Term>, Id<Term>)>,
}

impl Constraints {
    pub fn clear(&mut self) {
        self.equations.clear();
        self.disequations.clear();
        self.symmetric_disequations.clear();
        self.orderings.clear();
    }

    pub fn save(&mut self) {}

    pub fn restore(&mut self) {
        self.equations.clear();
        self.disequations.clear();
        self.symmetric_disequations.clear();
        self.orderings.clear();
    }

    pub fn assert_eq(&mut self, left: Id<Term>, right: Id<Term>) {
        self.equations.push((left, right));
    }

    pub fn assert_neq(&mut self, left: Id<Term>, right: Id<Term>) {
        self.disequations.push((left, right));
    }

    pub fn assert_symmetric_neq(&mut self, item: SymmetricDisequation) {
        self.symmetric_disequations.push(item);
    }

    pub fn assert_gt(&mut self, left: Id<Term>, right: Id<Term>) {
        self.orderings.push((left, right));
    }

    pub fn drain_equations(
        &mut self,
    ) -> impl Iterator<Item = (Id<Term>, Id<Term>)> + '_ {
        self.equations.drain(..)
    }

    pub fn drain_disequations(
        &mut self,
    ) -> impl Iterator<Item = (Id<Term>, Id<Term>)> + '_ {
        self.disequations.drain(..)
    }

    pub fn drain_symmetric_disequations(
        &mut self,
    ) -> impl Iterator<Item = SymmetricDisequation> + '_ {
        self.symmetric_disequations.drain(..)
    }

    pub fn drain_orderings(
        &self,
    ) -> impl Iterator<Item = (Id<Term>, Id<Term>)> + '_ {
        self.orderings.iter().copied()
    }
}
