use crate::prelude::*;

#[derive(Default)]
pub(crate) struct Solver {
    equations: Vec<(Id<Term>, Id<Term>)>,
}

impl Solver {
    pub(crate) fn assert_equal(&mut self, left: Id<Term>, right: Id<Term>) {
        self.equations.push((left, right));
    }

    pub(crate) fn assert_not_equal(
        &mut self,
        _left: Id<Term>,
        _right: Id<Term>,
    ) {
    }

    pub(crate) fn clear(&mut self) {
        self.equations.clear();
    }

    pub(crate) fn solve_fast(&mut self, terms: &Terms) -> bool {
        true
    }

    pub(crate) fn solve_correct(&mut self, terms: &Terms) -> bool {
        true
    }

    /*
    fn occurs(&self, terms: &Terms, x: Id<Variable>, term: Id<Term>) -> bool {
        let (_, view) = self.view(terms, term);
        match view {
            TermView::Variable(y) => x == y,
            TermView::Function(_, mut ts) => {
                ts.any(|t| self.occurs(terms, x, t))
            }
        }
    }
    */
}

impl Clone for Solver {
    fn clone(&self) -> Self {
        unimplemented!()
    }

    fn clone_from(&mut self, other: &Self) {
        self.equations.clone_from(&other.equations);
    }
}
