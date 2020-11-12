use crate::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct Equation {
    pub(crate) left: Id<Term>,
    pub(crate) right: Id<Term>,
}

#[derive(Clone, Copy)]
pub(crate) struct Disequation {
    pub(crate) left: Id<Term>,
    pub(crate) right: Id<Term>,
}

#[derive(Clone, Copy)]
pub(crate) struct SymmetricDisequation {
    pub(crate) left: (Id<Term>, Id<Term>),
    pub(crate) right: (Id<Term>, Id<Term>),
}

#[derive(Clone, Copy)]
pub(crate) struct Order {
    pub(crate) more: Id<Term>,
    pub(crate) less: Id<Term>,
}

#[derive(Default)]
pub(crate) struct Constraints {
    equations: Vec<Equation>,
    disequations: Vec<Disequation>,
    symmetric_disequations: Vec<SymmetricDisequation>,
    orderings: Vec<Order>,
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
        self.equations.push(Equation { left, right });
    }

    pub(crate) fn assert_neq(&mut self, left: Id<Term>, right: Id<Term>) {
        self.disequations.push(Disequation { left, right });
    }

    pub(crate) fn assert_symmetric_neq(
        &mut self,
        left: (Id<Term>, Id<Term>),
        right: (Id<Term>, Id<Term>),
    ) {
        self.symmetric_disequations
            .push(SymmetricDisequation { left, right });
    }

    pub(crate) fn assert_gt(&mut self, more: Id<Term>, less: Id<Term>) {
        self.orderings.push(Order { more, less });
    }

    pub(crate) fn drain_equations(
        &mut self,
    ) -> impl Iterator<Item = Equation> + '_ {
        self.equations.drain(..)
    }

    pub(crate) fn drain_disequations(
        &mut self,
    ) -> impl Iterator<Item = Disequation> + '_ {
        self.disequations.drain(..)
    }

    pub(crate) fn drain_symmetric_disequations(
        &mut self,
    ) -> impl Iterator<Item = SymmetricDisequation> + '_ {
        self.symmetric_disequations.drain(..)
    }

    pub(crate) fn drain_orderings(
        &mut self,
    ) -> impl Iterator<Item = Order> + '_ {
        self.orderings.drain(..)
    }
}
