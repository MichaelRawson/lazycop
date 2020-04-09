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

    pub(crate) fn compute_disequation(
        left: &Self,
        right: &Self,
        constraint_list: &mut ConstraintList,
        term_graph: &TermGraph,
    ) {
        if let (Atom::Predicate(p), Atom::Predicate(q)) = (left, right) {
            if let (TermView::Function(f, _), TermView::Function(g, _)) =
                (term_graph.view(*p), term_graph.view(*q))
            {
                if f == g {
                    constraint_list.add_disequality(*p, *q);
                }
            }
        }
    }
}
