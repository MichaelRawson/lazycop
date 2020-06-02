use crate::occurs;
use crate::occurs::Occurs;
use crate::prelude::*;
use crate::util::disjoint_set::{Disjoint, Set};

#[derive(Clone, Copy)]
struct AtomicDisequation {
    variable: Id<Variable>,
    term: Id<Term>,
}

#[derive(Default)]
pub(crate) struct Solver {
    equations: Vec<(Id<Term>, Id<Term>)>,
    disequations: Vec<(Id<Term>, Id<Term>)>,
    atomic_disequations: Block<AtomicDisequation>,
    solved_disequations: Block<Range<AtomicDisequation>>,
    aliases: Disjoint,
    to_alias: Block<Option<Id<Set>>>,
    from_alias: Block<Id<Term>>,

    save_atomic_disequations: Id<AtomicDisequation>,
    save_solved_disequations: Id<Range<AtomicDisequation>>,
    save_aliases: Disjoint,
    save_to_alias: Block<Option<Id<Set>>>,
    save_from_alias: Block<Id<Term>>,
}

impl Solver {
    pub(crate) fn clear(&mut self) {
        self.equations.clear();
        self.disequations.clear();
        self.atomic_disequations.clear();
        self.solved_disequations.clear();
        self.aliases.clear();
        self.to_alias.clear();
        self.from_alias.clear();
    }

    pub(crate) fn save(&mut self) {
        self.save_atomic_disequations = self.atomic_disequations.len();
        self.save_solved_disequations = self.solved_disequations.len();
        self.save_aliases.copy_from(&self.aliases);
        self.save_to_alias.copy_from(&self.to_alias);
        self.save_from_alias.copy_from(&self.from_alias);
    }

    pub(crate) fn restore(&mut self) {
        self.equations.clear();
        self.disequations.clear();
        self.atomic_disequations.truncate(self.save_atomic_disequations);
        self.solved_disequations.truncate(self.save_solved_disequations);
        self.aliases.copy_from(&self.save_aliases);
        self.to_alias.copy_from(&self.save_to_alias);
        self.from_alias.copy_from(&self.save_from_alias);
    }

    pub(crate) fn assert_equal(&mut self, left: Id<Term>, right: Id<Term>) {
        self.equations.push((left, right));
    }

    pub(crate) fn assert_not_equal(
        &mut self,
        left: Id<Term>,
        right: Id<Term>,
    ) {
        self.disequations.push((left, right));
    }

    pub(crate) fn bindings(
        &mut self,
    ) -> impl Iterator<Item = (Id<Variable>, Id<Term>)> + '_ {
        let to_alias = &self.to_alias;
        let from_alias = &self.from_alias;
        let aliases = &mut self.aliases;
        to_alias
            .range()
            .filter_map(move |id| {
                to_alias[id].map(|alias| (id.transmute(), alias))
            })
            .map(move |(x, alias)| {
                let alias = aliases.find(alias);
                let term = from_alias[alias.transmute()];
                (x, term)
            })
    }

    pub(crate) fn simplify(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
    ) -> bool {
        self.to_alias.resize(terms.len().transmute());
        while let Some((left, right)) = self.equations.pop() {
            if !self.solve_equation::<occurs::SkipCheck>(
                symbols, terms, left, right,
            ) {
                return false;
            }
        }
        !self.simplify_disequations(symbols, terms)
    }

    pub(crate) fn solve(&mut self, symbols: &Symbols, terms: &Terms) -> bool {
        self.to_alias.resize(terms.len().transmute());
        while let Some((left, right)) = self.equations.pop() {
            if !self
                .solve_equation::<occurs::Check>(symbols, terms, left, right)
            {
                return false;
            }
        }
        if self.simplify_disequations(symbols, terms) {
            return false;
        }
        for solved in self.solved_disequations.range() {
            let solved = self.solved_disequations[solved];
            if self.check_solved_disequation(symbols, terms, solved) {
                return false;
            }
        }
        true
    }

    fn solve_equation<O: Occurs>(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        left: Id<Term>,
        right: Id<Term>,
    ) -> bool {
        let (left, lview) = self.lookup(symbols, terms, left);
        let (right, rview) = self.lookup(symbols, terms, right);
        if left == right {
            return true;
        }
        match (lview, rview) {
            (TermView::Variable(x), TermView::Variable(y)) => {
                self.alias(x, y);
                true
            }
            (TermView::Variable(x), _) => {
                if O::CHECK && self.occurs(symbols, terms, x, right) {
                    false
                } else {
                    self.bind(x, right);
                    true
                }
            }
            (_, TermView::Variable(x)) => {
                if O::CHECK && self.occurs(symbols, terms, x, left) {
                    false
                } else {
                    self.bind(x, left);
                    true
                }
            }
            (TermView::Function(f, ss), TermView::Function(g, ts))
                if f == g =>
            {
                ss.zip(ts)
                    .map(|(s, t)| (terms.resolve(s), terms.resolve(t)))
                    .all(|(s, t)| {
                        self.solve_equation::<O>(symbols, terms, s, t)
                    })
            }
            (TermView::Function(_, _), TermView::Function(_, _)) => false,
        }
    }

    fn check_solved_disequation(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        solved: Range<AtomicDisequation>,
    ) -> bool {
        for id in solved {
            let atomic = self.atomic_disequations[id];
            if !self.check_disequation(
                symbols,
                terms,
                atomic.variable.transmute(),
                atomic.term,
            ) {
                return false;
            }
        }
        true
    }

    fn check_disequation(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        left: Id<Term>,
        right: Id<Term>,
    ) -> bool {
        let (left, lview) = self.lookup(symbols, terms, left);
        let (right, rview) = self.lookup(symbols, terms, right);
        if left == right {
            return true;
        }
        match (lview, rview) {
            (TermView::Function(f, ss), TermView::Function(g, ts))
                if f == g =>
            {
                ss.zip(ts)
                    .map(|(s, t)| (terms.resolve(s), terms.resolve(t)))
                    .all(|(s, t)| self.check_disequation(symbols, terms, s, t))
            }
            _ => false,
        }
    }

    fn simplify_disequations(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
    ) -> bool {
        while let Some((left, right)) = self.disequations.pop() {
            let start = self.atomic_disequations.len();
            if !self.simplify_disequation(symbols, terms, left, right) {
                self.atomic_disequations.truncate(start);
                continue;
            }
            let end = self.atomic_disequations.len();
            if start == end {
                return true;
            }
            let solved = Range::new(start, end);
            self.solved_disequations.push(solved);
        }
        false
    }

    fn simplify_disequation(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        left: Id<Term>,
        right: Id<Term>,
    ) -> bool {
        let (left, lview) = self.lookup(symbols, terms, left);
        let (right, rview) = self.lookup(symbols, terms, right);
        if left == right {
            return true;
        }
        match (lview, rview) {
            (TermView::Variable(variable), _)
                if !self.occurs(symbols, terms, variable, right) =>
            {
                let term = right;
                let atomic = AtomicDisequation { variable, term };
                self.atomic_disequations.push(atomic);
                true
            }
            (_, TermView::Variable(variable))
                if !self.occurs(symbols, terms, variable, left) =>
            {
                let term = left;
                let atomic = AtomicDisequation { variable, term };
                self.atomic_disequations.push(atomic);
                true
            }
            (TermView::Function(f, ss), TermView::Function(g, ts))
                if f == g =>
            {
                ss.zip(ts)
                    .map(|(s, t)| (terms.resolve(s), terms.resolve(t)))
                    .all(|(s, t)| {
                        self.simplify_disequation(symbols, terms, s, t)
                    })
            }
            _ => false,
        }
    }

    fn occurs(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        x: Id<Variable>,
        term: Id<Term>,
    ) -> bool {
        let (_, view) = self.lookup(symbols, terms, term);
        match view {
            TermView::Variable(y) => x == y,
            TermView::Function(_, args) => args
                .map(|t| terms.resolve(t))
                .any(|t| self.occurs(symbols, terms, x, t)),
        }
    }

    fn lookup(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        term: Id<Term>,
    ) -> (Id<Term>, TermView) {
        match terms.view(symbols, term) {
            TermView::Variable(x) => {
                if let Some(alias) = self.to_alias[x.transmute()] {
                    let alias = self.aliases.find(alias);
                    let term = self.from_alias[alias.transmute()];
                    (term, terms.view(symbols, term))
                } else {
                    let alias = self.aliases.singleton();
                    self.to_alias[x.transmute()] = Some(alias);
                    let term = x.transmute();
                    self.from_alias.resize(self.aliases.len().transmute());
                    self.from_alias[alias.transmute()] = term;
                    (term, TermView::Variable(x))
                }
            }
            view => (term, view),
        }
    }

    fn alias(&mut self, x: Id<Variable>, y: Id<Variable>) {
        let x_alias = some(self.to_alias[x.transmute()]);
        let y_alias = some(self.to_alias[y.transmute()]);
        self.aliases.merge(x_alias, y_alias);
    }

    fn bind(&mut self, x: Id<Variable>, term: Id<Term>) {
        let alias = some(self.to_alias[x.transmute()]);
        self.from_alias[alias.transmute()] = term;
    }
}
