use crate::prelude::*;

#[derive(Default)]
pub(crate) struct Bindings {
    bound: Block<Option<Id<Term>>>,
    save: Block<Option<Id<Term>>>,
}

impl Bindings {
    pub(crate) fn clear(&mut self) {
        self.bound.clear();
    }

    pub(crate) fn save(&mut self) {
        self.save.copy_from(&self.bound);
    }

    pub(crate) fn restore(&mut self) {
        self.bound.copy_from(&self.save);
    }

    pub(crate) fn resize(&mut self, len: Id<Term>) {
        self.bound.resize(len.transmute());
    }

    pub(crate) fn bind(&mut self, x: Id<Variable>, term: Id<Term>) {
        self.bound[x.transmute()] = Some(term);
    }

    pub(crate) fn is_bound(&self, x: Id<Variable>) -> bool {
        self.bound[x.transmute()].is_some()
    }

    pub(crate) fn items(
        &self,
    ) -> impl Iterator<Item = (Id<Variable>, Id<Term>)> + '_ {
        self.bound.range().filter_map(move |id| {
            let variable = id.transmute();
            let term = self.bound[id.transmute()]?;
            Some((variable, term))
        })
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
                .map(|t| terms.resolve(t))
                .any(|t| self.occurs(symbols, terms, x, t)),
        }
    }

    fn lookup(&self, x: Id<Variable>) -> Id<Term> {
        self.bound[x.transmute()].unwrap_or_else(|| x.transmute())
    }
}
