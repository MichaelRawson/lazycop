use crate::occurs::Occurs;
use crate::prelude::*;
use crate::util::disjoint_set::{Disjoint, Set};

#[derive(Default)]
pub(crate) struct EquationSolver {
    aliases: Disjoint,
    to_alias: LUT<Term, Option<Id<Set>>>,
    from_alias: LUT<Set, Id<Term>>,

    save_aliases: Disjoint,
    save_to_alias: LUT<Term, Option<Id<Set>>>,
    save_from_alias: LUT<Set, Id<Term>>,
}

impl EquationSolver {
    pub(crate) fn clear(&mut self) {
        self.aliases.clear();
        self.to_alias.resize(Length::default());
        self.from_alias.resize(Length::default());
    }

    pub(crate) fn save(&mut self) {
        self.save_aliases.copy_from(&self.aliases);
        self.save_to_alias.copy_from(&self.to_alias);
        self.save_from_alias.copy_from(&self.from_alias);
    }

    pub(crate) fn restore(&mut self) {
        self.aliases.copy_from(&self.save_aliases);
        self.to_alias.copy_from(&self.save_to_alias);
        self.from_alias.copy_from(&self.save_from_alias);
    }

    pub(crate) fn solve<
        O: Occurs,
        I: Iterator<Item = (Id<Term>, Id<Term>)>,
    >(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &mut Bindings,
        mut equations: I,
    ) -> bool {
        self.to_alias.resize(terms.len());
        let success = equations.all(|(left, right)| {
            self.solve_equation::<O>(symbols, terms, left, right)
        });
        if !success {
            return false;
        }

        bindings.resize(terms.len().transmute());
        for id in self.to_alias.range() {
            if let Some(alias) = self.to_alias[id] {
                let set = self.aliases.find(alias);
                let term = self.from_alias[set];
                bindings.bind(id.transmute(), term);
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
        match (lview, rview) {
            (TermView::Variable(x), TermView::Variable(y)) => {
                if x != y {
                    self.alias(x, y);
                }
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
                ss.into_iter()
                    .zip(ts.into_iter())
                    .map(|(s, t)| (terms.resolve(s), terms.resolve(t)))
                    .all(|(s, t)| {
                        self.solve_equation::<O>(symbols, terms, s, t)
                    })
            }
            (TermView::Function(_, _), TermView::Function(_, _)) => false,
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
                .into_iter()
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
                if let Some(alias) = self.to_alias[term] {
                    let alias = self.aliases.find(alias);
                    let term = self.from_alias[alias];
                    (term, terms.view(symbols, term))
                } else {
                    let alias = self.aliases.singleton();
                    self.to_alias[term] = Some(alias);
                    self.from_alias.resize(self.aliases.len());
                    self.from_alias[alias] = term;
                    (term, TermView::Variable(x))
                }
            }
            view => (term, view),
        }
    }

    fn alias(&mut self, x: Id<Variable>, y: Id<Variable>) {
        let x_alias = unwrap(self.to_alias[x.transmute()]);
        let y_alias = unwrap(self.to_alias[y.transmute()]);
        self.aliases.merge(x_alias, y_alias);
    }

    fn bind(&mut self, x: Id<Variable>, term: Id<Term>) {
        let alias = unwrap(self.to_alias[x.transmute()]);
        self.from_alias[alias] = term;
    }
}
