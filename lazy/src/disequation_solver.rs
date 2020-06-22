use crate::binding::Bindings;
use crate::constraint::SymmetricDisequation;
use crate::prelude::*;

#[derive(Clone, Copy)]
struct AtomicDisequation {
    variable: Id<Variable>,
    term: Id<Term>,
}

#[derive(Default)]
pub struct DisequationSolver {
    atomic: Block<AtomicDisequation>,
    solved: Block<Range<AtomicDisequation>>,

    save_atomic: Id<AtomicDisequation>,
    save_solved: Id<Range<AtomicDisequation>>,
}

impl DisequationSolver {
    pub fn clear(&mut self) {
        self.atomic.clear();
        self.solved.clear();
    }

    pub fn save(&mut self) {
        self.save_atomic = self.atomic.len();
        self.save_solved = self.solved.len();
    }

    pub fn restore(&mut self) {
        self.atomic.truncate(self.save_atomic);
        self.solved.truncate(self.save_solved);
    }

    pub fn simplify<I: Iterator<Item = (Id<Term>, Id<Term>)>>(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
        disequations: I,
    ) -> bool {
        for (left, right) in disequations {
            let start = self.atomic.len();
            if self.simplify_disequation(symbols, terms, bindings, left, right)
            {
                self.atomic.truncate(start);
                continue;
            }
            let end = self.atomic.len();
            if start == end {
                return false;
            }
            let solved = Range::new(start, end);
            self.solved.push(solved);
        }
        true
    }

    pub fn simplify_symmetric<I: Iterator<Item = SymmetricDisequation>>(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
        disequations: I,
    ) -> bool {
        for symmetric in disequations {
            let SymmetricDisequation {
                left1,
                left2,
                right1,
                right2,
            } = symmetric;

            let start = self.atomic.len();
            if self
                .simplify_disequation(symbols, terms, bindings, left1, left2)
                || self.simplify_disequation(
                    symbols, terms, bindings, right1, right2,
                )
            {
                self.atomic.truncate(start);
                continue;
            }
            let end = self.atomic.len();
            if start == end {
                return false;
            }
            let solved = Range::new(start, end);
            self.solved.push(solved);

            let start = self.atomic.len();
            if self
                .simplify_disequation(symbols, terms, bindings, left1, right2)
                || self.simplify_disequation(
                    symbols, terms, bindings, right1, left2,
                )
            {
                self.atomic.truncate(start);
                continue;
            }
            let end = self.atomic.len();
            if start == end {
                return false;
            }
            let solved = Range::new(start, end);
            self.solved.push(solved);
        }
        true
    }

    pub fn check(
        &self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
    ) -> bool {
        self.solved
            .range()
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
        solved.map(|id| self.atomic[id]).any(|atomic| {
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
        let (left, lview) = bindings.view(symbols, terms, left);
        let (right, rview) = bindings.view(symbols, terms, right);
        if left == right {
            return false;
        }
        if let (TermView::Function(f, ss), TermView::Function(g, ts)) =
            (lview, rview)
        {
            f != g
                || ss
                    .zip(ts)
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
        let (left, lview) = bindings.view(symbols, terms, left);
        let (right, rview) = bindings.view(symbols, terms, right);
        match (lview, rview) {
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
                ss.zip(ts)
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
