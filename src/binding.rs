use crate::prelude::*;

#[derive(Default)]
pub(crate) struct Bindings {
    bound: LUT<Variable, Option<Id<Term>>>,
    save: LUT<Variable, Option<Id<Term>>>,
}

impl Bindings {
    pub(crate) fn clear(&mut self) {
        self.bound.clear();
    }

    pub(crate) fn save(&mut self) {
        self.save.duplicate(&self.bound);
    }

    pub(crate) fn restore(&mut self) {
        self.bound.duplicate(&self.save);
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

    pub(crate) fn occurs(
        &self,
        symbols: &Symbols,
        terms: &Terms,
        x: Id<Variable>,
        term: Id<Term>,
    ) -> bool {
        let term = self.resolve(term);
        match terms.view(symbols, term) {
            TermView::Variable(y) => x == y,
            TermView::Function(_, args) => args
                .into_iter()
                .any(|t| self.occurs(symbols, terms, x, terms.resolve(t))),
        }
    }

    pub(crate) fn resolve(&self, x: Id<Term>) -> Id<Term> {
        self.bound[x.transmute()].unwrap_or(x)
    }
}
