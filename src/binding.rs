use crate::prelude::*;

#[derive(Default)]
pub struct Bindings {
    bound: Block<Option<Id<Term>>>,
    save: Block<Option<Id<Term>>>,
}

impl Bindings {
    pub fn clear(&mut self) {
        self.bound.clear();
    }

    pub fn save(&mut self) {
        self.save.copy_from(&self.bound);
    }

    pub fn restore(&mut self) {
        self.bound.copy_from(&self.save);
    }

    pub fn resize(&mut self, len: Id<Term>) {
        self.bound.resize(len.transmute());
    }

    pub fn bind(&mut self, x: Id<Variable>, term: Id<Term>) {
        self.bound[x.transmute()] = Some(term);
    }

    pub fn is_bound(&self, x: Id<Variable>) -> bool {
        self.bound[x.transmute()].is_some()
    }

    pub fn items(
        &self,
    ) -> impl Iterator<Item = (Id<Variable>, Id<Term>)> + '_ {
        self.bound.range().filter_map(move |id| {
            let variable = id.transmute();
            let term = self.bound[id.transmute()]?;
            Some((variable, term))
        })
    }

    pub fn view(
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

    pub fn occurs(
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
