use crate::prelude::*;

#[derive(Clone, Copy)]
pub(crate) enum Atom {
    Predicate(Id<Term>),
    Equality(Id<Term>, Id<Term>),
}

impl Atom {
    pub(crate) fn offset(&self, offset: Offset<Term>) -> Self {
        match self {
            Atom::Predicate(p) => Atom::Predicate(*p + offset),
            Atom::Equality(left, right) => {
                Atom::Equality(*left + offset, *right + offset)
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
            _ => unreachable!("predicate term of non-predicate"),
        }
    }

    pub(crate) fn get_predicate_symbol(
        &self,
        term_graph: &TermGraph,
    ) -> Id<Symbol> {
        match term_graph.view(self.get_predicate()) {
            (_, TermView::Function(p, _)) => p,
            _ => unreachable!("non-function predicate symbol"),
        }
    }

    pub(crate) fn get_equality(&self) -> (Id<Term>, Id<Term>) {
        match self {
            Atom::Equality(left, right) => (*left, *right),
            _ => unreachable!("equality term of non-equality"),
        }
    }

    pub(crate) fn add_positive_constraints(&self, solver: &mut Solver) {
        if let Atom::Equality(left, right) = self {
            solver.assert_not_equal(*left, *right);
        }
    }

    pub(crate) fn add_disequation_constraints(
        &self,
        solver: &mut Solver,
        term_graph: &TermGraph,
        other: &Self,
    ) {
        if self.is_predicate()
            && other.is_predicate()
            && self.get_predicate_symbol(term_graph)
                == other.get_predicate_symbol(term_graph)
        {
            solver
                .assert_not_equal(self.get_predicate(), other.get_predicate());
        } else if self.is_equality() && other.is_equality() {
            let (l1, r1) = self.get_equality();
            let (l2, r2) = other.get_equality();
            solver.assert_not_equal_symmetric((l1, r1), (l2, r2));
        }
    }
}
