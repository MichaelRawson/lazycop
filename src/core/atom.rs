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
            TermView::Function(p, _) => p,
            _ => unreachable!("non-function predicate symbol"),
        }
    }

    pub(crate) fn get_equality(&self) -> (Id<Term>, Id<Term>) {
        match self {
            Atom::Equality(left, right) => (*left, *right),
            _ => unreachable!("equality term of non-equality"),
        }
    }

    pub(crate) fn possibly_equal(
        left: &Self,
        right: &Self,
        term_graph: &TermGraph,
    ) -> bool {
        if left.is_predicate() {
            right.is_predicate()
                && left.get_predicate_symbol(term_graph)
                    == right.get_predicate_symbol(term_graph)
        } else {
            !right.is_predicate()
        }
    }
}
