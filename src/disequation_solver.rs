use crate::constraint::{Disequation, SymmetricDisequation};
use crate::prelude::*;

#[derive(Clone, Copy)]
struct AtomicDisequation {
    variable: Id<Variable>,
    term: Id<Term>,
}

#[derive(Default)]
pub(crate) struct DisequationSolver {
    atomic: Block<AtomicDisequation>,
    solved: Block<Range<AtomicDisequation>>,

    save_atomic: Length<AtomicDisequation>,
    save_solved: Length<Range<AtomicDisequation>>,
}

impl DisequationSolver {
    pub(crate) fn clear(&mut self) {
        self.atomic.clear();
        self.solved.clear();
    }

    pub(crate) fn save(&mut self) {
        self.save_atomic = self.atomic.len();
        self.save_solved = self.solved.len();
    }

    pub(crate) fn restore(&mut self) {
        self.atomic.truncate(self.save_atomic);
        self.solved.truncate(self.save_solved);
    }

    pub(crate) fn simplify<I: Iterator<Item = Disequation>>(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
        disequations: I,
    ) -> bool {
        for Disequation { left, right } in disequations {
            let reset = self.atomic.len();
            let start = self.atomic.end();
            if self.simplify_disequation(symbols, terms, bindings, left, right)
            {
                self.atomic.truncate(reset);
                continue;
            }
            let end = self.atomic.end();
            if start == end {
                return false;
            }
            let solved = Range::new(start, end);
            self.solved.push(solved);
        }
        true
    }

    pub(crate) fn simplify_symmetric<
        I: Iterator<Item = SymmetricDisequation>,
    >(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
        disequations: I,
    ) -> bool {
        for SymmetricDisequation { left, right } in disequations {
            let (left1, left2) = left;
            let (right1, right2) = right;
            let reset = self.atomic.len();
            let start = self.atomic.end();
            if self
                .simplify_disequation(symbols, terms, bindings, left1, left2)
                || self.simplify_disequation(
                    symbols, terms, bindings, right1, right2,
                )
            {
                self.atomic.truncate(reset);
                continue;
            }
            let end = self.atomic.end();
            if start == end {
                return false;
            }
            let solved = Range::new(start, end);
            self.solved.push(solved);

            let reset = self.atomic.len();
            let start = self.atomic.end();
            if self
                .simplify_disequation(symbols, terms, bindings, left1, right2)
                || self.simplify_disequation(
                    symbols, terms, bindings, right1, left2,
                )
            {
                self.atomic.truncate(reset);
                continue;
            }
            let end = self.atomic.end();
            if start == end {
                return false;
            }
            let solved = Range::new(start, end);
            self.solved.push(solved);
        }
        true
    }

    pub(crate) fn check(
        &self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
    ) -> bool {
        self.solved
            .range()
            .into_iter()
            .rev()
            .map(|id| self.solved[id])
            .all(|solved| {
                self.check_solved_disequation(symbols, terms, bindings, solved)
            })
    }

    fn check_solved_disequation(
        &self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
        solved: Range<AtomicDisequation>,
    ) -> bool {
        solved.into_iter().map(|id| self.atomic[id]).any(|atomic| {
            !bindings.is_bound(atomic.variable)
                || self.check_disequation(
                    symbols,
                    terms,
                    bindings,
                    atomic.variable.transmute(),
                    atomic.term,
                )
        })
    }

    fn check_disequation(
        &self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
        left: Id<Term>,
        right: Id<Term>,
    ) -> bool {
        let left = bindings.resolve(left);
        let right = bindings.resolve(right);
        if left == right {
            return false;
        }
        if let (TermView::Function(f, ss), TermView::Function(g, ts)) =
            (terms.view(symbols, left), terms.view(symbols, right))
        {
            f != g
                || ss
                    .into_iter()
                    .zip(ts.into_iter())
                    .map(|(s, t)| (terms.resolve(s), terms.resolve(t)))
                    .any(|(s, t)| {
                        self.check_disequation(symbols, terms, bindings, s, t)
                    })
        } else {
            true
        }
    }

    fn simplify_disequation(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
        left: Id<Term>,
        right: Id<Term>,
    ) -> bool {
        let left = bindings.resolve(left);
        let right = bindings.resolve(right);
        match (terms.view(symbols, left), terms.view(symbols, right)) {
            (TermView::Variable(x), TermView::Variable(y)) => {
                if x == y {
                    return false;
                }
                let variable = x;
                let term = right;
                let atomic = AtomicDisequation { variable, term };
                self.atomic.push(atomic);
                false
            }
            (TermView::Variable(variable), _)
                if !bindings.occurs(symbols, terms, variable, right) =>
            {
                let term = right;
                let atomic = AtomicDisequation { variable, term };
                self.atomic.push(atomic);
                false
            }
            (_, TermView::Variable(variable))
                if !bindings.occurs(symbols, terms, variable, left) =>
            {
                let term = left;
                let atomic = AtomicDisequation { variable, term };
                self.atomic.push(atomic);
                false
            }
            (TermView::Function(f, ss), TermView::Function(g, ts))
                if f == g =>
            {
                ss.into_iter()
                    .zip(ts.into_iter())
                    .map(|(s, t)| (terms.resolve(s), terms.resolve(t)))
                    .any(|(s, t)| {
                        self.simplify_disequation(
                            symbols, terms, bindings, s, t,
                        )
                    })
            }
            _ => true,
        }
    }
}
