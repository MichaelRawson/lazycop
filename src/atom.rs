use crate::prelude::*;
use crate::solver::Solver;

#[derive(Clone, Copy)]
pub(crate) enum Atom {
    Predicate(Id<Term>),
    Equality(Id<Term>, Id<Term>),
}

impl Atom {
    pub(crate) fn offset(&mut self, offset: Offset<Term>) {
        match self {
            Atom::Predicate(p) => {
                *p = *p + offset;
            }
            Atom::Equality(left, right) => {
                *left = *left + offset;
                *right = *right + offset;
            }
        }
    }

    pub(crate) fn is_predicate(&self) -> bool {
        match self {
            Atom::Predicate(_) => true,
            _ => false,
        }
    }

    pub(crate) fn is_equality(&self) -> bool {
        match self {
            Atom::Equality(_, _) => true,
            _ => false,
        }
    }

    pub(crate) fn get_predicate(&self) -> Id<Term> {
        match self {
            Atom::Predicate(p) => *p,
            _ => unreachable(),
        }
    }

    pub(crate) fn get_equality(&self) -> (Id<Term>, Id<Term>) {
        match self {
            Atom::Equality(left, right) => (*left, *right),
            _ => unreachable(),
        }
    }

    pub(crate) fn get_predicate_symbol(&self, terms: &Terms) -> Id<Symbol> {
        terms.symbol(self.get_predicate())
    }

    pub(crate) fn add_positive_constraints(&self, solver: &mut Solver) {
        if let Atom::Equality(left, right) = self {
            solver.assert_not_equal(*left, *right);
        }
    }

    pub(crate) fn add_disequation_constraints(
        &self,
        solver: &mut Solver,
        terms: &Terms,
        other: &Self,
    ) {
        if self.is_predicate()
            && other.is_predicate()
            && self.get_predicate_symbol(terms)
                == other.get_predicate_symbol(terms)
        {
            solver
                .assert_not_equal(self.get_predicate(), other.get_predicate());
        }
    }
}
