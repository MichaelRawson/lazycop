use crate::binding::Bindings;
use crate::occurs::Occurs;
use crate::prelude::*;
use crate::util::disjoint_set::{Disjoint, Set};

#[derive(Default)]
pub(crate) struct EquationSolver {
    aliases: Disjoint,
    to_alias: Block<Option<Id<Set>>>,
    from_alias: Block<Id<Term>>,

    save_aliases: Disjoint,
    save_to_alias: Block<Option<Id<Set>>>,
    save_from_alias: Block<Id<Term>>,
}

impl EquationSolver {
    pub(crate) fn clear(&mut self) {
        self.aliases.clear();
        self.to_alias.clear();
        self.from_alias.clear();
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
        self.to_alias.resize(terms.len().transmute());
        let success = equations.all(|(left, right)| {
            self.solve_equation::<O>(symbols, terms, left, right)
        });
        if !success {
            return false;
        }

        bindings.resize(terms.len().transmute());
        for id in self.to_alias.range() {
            let variable = id.transmute();
            if let Some(alias) = self.to_alias[id] {
                let set = self.aliases.find(alias);
                let term = self.from_alias[set.transmute()];
                bindings.bind(variable, term);
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
