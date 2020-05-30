use crate::occurs;
use crate::occurs::Occurs;
use crate::prelude::*;
use crate::util::disjoint_set::{Disjoint, Set};

#[derive(Default)]
pub(crate) struct Solver {
    equations: Vec<(Id<Term>, Id<Term>)>,
    disequations: Vec<(Id<Term>, Id<Term>)>,
    aliases: Disjoint,
    to_alias: Block<Option<Id<Set>>>,
    from_alias: Block<Id<Term>>,
}

impl Solver {
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

    pub(crate) fn clear(&mut self) {
        self.equations.clear();
        self.disequations.clear();
        self.aliases.clear();
        self.to_alias.clear();
        self.from_alias.clear();
    }

    pub(crate) fn bindings(
        &mut self,
    ) -> impl Iterator<Item = (Id<Variable>, Id<Term>)> + '_ {
        let to_alias = &self.to_alias;
        let from_alias = &self.from_alias;
        let aliases = &mut self.aliases;
        to_alias
            .into_iter()
            .filter_map(move |id| {
                to_alias[id].map(|alias| (id.transmute(), alias))
            })
            .map(move |(x, alias)| {
                let alias = aliases.find(alias);
                let term = from_alias[alias.transmute()];
                (x, term)
            })
            .filter(|(x, term)| x.transmute() != *term)
    }

    pub(crate) fn solve_fast(&mut self, terms: &Terms) -> bool {
        self.to_alias.resize(terms.as_ref().len().transmute());
        while let Some((left, right)) = self.equations.pop() {
            if !self.solve_equation::<occurs::SkipCheck>(terms, left, right) {
                return false;
            }
        }
        true
    }

    pub(crate) fn solve_correct(&mut self, terms: &Terms) -> bool {
        self.to_alias.resize(terms.as_ref().len().transmute());
        while let Some((left, right)) = self.equations.pop() {
            if !self.solve_equation::<occurs::Check>(terms, left, right) {
                return false;
            }
        }
        while let Some((left, right)) = self.disequations.pop() {
            if self.check_disequation(terms, left, right) {
                return false;
            }
        }
        true
    }

    fn solve_equation<O: Occurs>(
        &mut self,
        terms: &Terms,
        left: Id<Term>,
        right: Id<Term>,
    ) -> bool {
        let (left, lview) = self.lookup(terms, left);
        let (right, rview) = self.lookup(terms, right);
        if left == right {
            return true;
        }
        match (lview, rview) {
            (TermView::Variable(x), TermView::Variable(y)) => {
                self.alias(x, y);
                true
            }
            (TermView::Variable(x), _) => {
                if O::CHECK && self.occurs(terms, x, right) {
                    false
                } else {
                    self.bind(x, right);
                    true
                }
            }
            (_, TermView::Variable(x)) => {
                if O::CHECK && self.occurs(terms, x, left) {
                    false
                } else {
                    self.bind(x, left);
                    true
                }
            }
            (TermView::Function(f, ts), TermView::Function(g, ss))
                if f == g =>
            {
                ts.zip(ss)
                    .map(|(t, s)| (terms.resolve(t), terms.resolve(s)))
                    .all(|(t, s)| self.solve_equation::<O>(terms, t, s))
            }
            (TermView::Function(_, _), TermView::Function(_, _)) => false,
        }
    }

    pub(crate) fn check_equation(
        &mut self,
        terms: &Terms,
        left: Id<Term>,
        right: Id<Term>,
    ) -> bool {
        let (left, lview) = self.lookup(terms, left);
        let (right, rview) = self.lookup(terms, right);
        if left == right {
            return true;
        }
        match (lview, rview) {
            (TermView::Variable(_), TermView::Variable(_)) => true,
            (TermView::Variable(x), _) => !self.occurs(terms, x, right),
            (_, TermView::Variable(x)) => !self.occurs(terms, x, left),
            (TermView::Function(f, ts), TermView::Function(g, ss))
                if f == g =>
            {
                ts.zip(ss)
                    .map(|(t, s)| (terms.resolve(t), terms.resolve(s)))
                    .all(|(t, s)| self.check_equation(terms, t, s))
            }
            (TermView::Function(_, _), TermView::Function(_, _)) => false,
        }
    }

    fn check_disequation(
        &mut self,
        terms: &Terms,
        left: Id<Term>,
        right: Id<Term>,
    ) -> bool {
        let (left, lview) = self.lookup(terms, left);
        let (right, rview) = self.lookup(terms, right);
        if left == right {
            return true;
        }
        match (lview, rview) {
            (TermView::Function(f, ts), TermView::Function(g, ss))
                if f == g =>
            {
                ts.zip(ss)
                    .map(|(t, s)| (terms.resolve(t), terms.resolve(s)))
                    .all(|(t, s)| self.check_disequation(terms, t, s))
            }
            _ => false,
        }
    }

    fn occurs(
        &mut self,
        terms: &Terms,
        x: Id<Variable>,
        term: Id<Term>,
    ) -> bool {
        let (_, view) = self.lookup(terms, term);
        match view {
            TermView::Variable(y) => x == y,
            TermView::Function(_, args) => args
                .map(|t| terms.resolve(t))
                .any(|t| self.occurs(terms, x, t)),
        }
    }

    fn lookup(
        &mut self,
        terms: &Terms,
        term: Id<Term>,
    ) -> (Id<Term>, TermView) {
        match terms.view(term) {
            TermView::Variable(x) => {
                if let Some(alias) = self.to_alias[x.transmute()] {
                    let alias = self.aliases.find(alias);
                    let term = self.from_alias[alias.transmute()];
                    (term, terms.view(term))
                } else {
                    let alias = self.aliases.singleton();
                    self.to_alias[x.transmute()] = Some(alias);
                    let term = x.transmute();
                    self.from_alias
                        .resize(self.aliases.as_ref().len().transmute());
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

impl Clone for Solver {
    fn clone(&self) -> Self {
        unimplemented!()
    }

    fn clone_from(&mut self, other: &Self) {
        self.equations.clone_from(&other.equations);
        self.disequations.clone_from(&other.disequations);
        self.aliases.clone_from(&other.aliases);
        self.to_alias.clone_from(&other.to_alias);
        self.from_alias.clone_from(&other.from_alias);
    }
}
