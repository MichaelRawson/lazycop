use crate::prelude::*;

#[derive(Default)]
pub(crate) struct Bindings {
    bound: LUT<Variable, Option<Id<Term>>>,
    save: LUT<Variable, Option<Id<Term>>>,
}

impl Bindings {
    pub(crate) fn clear(&mut self) {
        self.bound.resize(Id::default());
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
        self.bound[x] = Some(term);
    }

    pub(crate) fn is_bound(&self, x: Id<Variable>) -> bool {
        self.bound[x].is_some()
    }

    pub(crate) fn items(
        &self,
    ) -> impl Iterator<Item = (Id<Variable>, Id<Term>)> + '_ {
        self.bound.range().into_iter().filter_map(move |variable| {
            self.bound[variable].map(|term| (variable, term))
        })
    }

    #[cfg(feature = "nn")]
    pub(crate) fn graph(
        &self,
        graph: &mut Graph,
        symbols: &Symbols,
        terms: &Terms,
        term: Id<Term>,
    ) -> Id<Node> {
        let (term, view) = self.view(symbols, terms, term);
        if let Some(node) = graph.get_cached_term(term) {
            return node;
        }
        let node = match view {
            TermView::Variable(_) => graph.variable(),
            TermView::Function(f, args) => {
                let symbol = graph.symbol(f);
                if Range::is_empty(args) {
                    return symbol;
                }
                let application = graph.application(symbol);
                let mut previous = symbol;
                for arg in args {
                    let term =
                        self.graph(graph, symbols, terms, terms.resolve(arg));
                    let argument = graph.argument(application, previous, term);
                    previous = argument;
                }
                application
            }
        };
        graph.cache_term(term, node);
        node
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
