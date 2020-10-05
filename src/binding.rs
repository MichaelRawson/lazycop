use crate::prelude::*;

#[derive(Default)]
pub(crate) struct Bindings {
    bound: LUT<Variable, Option<Id<Term>>>,
    save: LUT<Variable, Option<Id<Term>>>,
}

impl Bindings {
    pub(crate) fn clear(&mut self) {
        self.bound.resize(Length::default());
    }

    pub(crate) fn save(&mut self) {
        self.save.copy_from(&self.bound);
    }

    pub(crate) fn restore(&mut self) {
        self.bound.copy_from(&self.save);
    }

    pub(crate) fn resize(&mut self, len: Length<Term>) {
        self.bound.resize(len.transmute());
    }

    pub(crate) fn bind(&mut self, x: Id<Variable>, term: Id<Term>) {
        self.bound[x] = Some(term);
    }

    pub(crate) fn get(&self, x: Id<Variable>) -> Option<Id<Term>> {
        self.bound[x]
    }

    pub(crate) fn is_bound(&self, x: Id<Variable>) -> bool {
        self.bound[x].is_some()
    }

    pub(crate) fn new_bindings(
        &self,
    ) -> impl Iterator<Item = (Id<Variable>, Id<Term>)> + '_ {
        self.bound
            .range()
            .into_iter()
            .filter(move |variable| {
                self.save[*variable] != self.bound[*variable]
            })
            .filter_map(move |variable| {
                self.bound[variable].map(|term| (variable, term))
            })
            .filter(|(variable, term)| variable.transmute() != *term)
    }

    pub(crate) fn view(
        &self,
        symbols: &Symbols,
        terms: &Terms,
        mut term: Id<Term>,
    ) -> (Id<Term>, TermView) {
        if terms.is_variable(term) {
            term = self.lookup(term.transmute());
        }
        (term, terms.view(symbols, term))
    }

    pub(crate) fn occurs(
        &self,
        symbols: &Symbols,
        terms: &Terms,
        x: Id<Variable>,
        term: Id<Term>,
    ) -> bool {
        let (_, view) = self.view(symbols, terms, term);
        match view {
            TermView::Variable(y) => x == y,
            TermView::Function(_, args) => args
                .into_iter()
                .map(|t| terms.resolve(t))
                .any(|t| self.occurs(symbols, terms, x, t)),
        }
    }

    fn lookup(&self, x: Id<Variable>) -> Id<Term> {
        self.bound[x].unwrap_or_else(|| x.transmute())
    }
}
