use crate::prelude::*;

#[derive(Default)]
pub struct Bindings {
    bound: LUT<Variable, Option<Id<Term>>>,
    save: LUT<Variable, Option<Id<Term>>>,
}

impl Bindings {
    pub fn clear(&mut self) {
        self.bound.resize(Id::default());
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
        self.bound[x] = Some(term);
    }

    pub fn is_bound(&self, x: Id<Variable>) -> bool {
        self.bound[x].is_some()
    }

    pub fn items(
        &self,
    ) -> impl Iterator<Item = (Id<Variable>, Id<Term>)> + '_ {
        self.bound.range().filter_map(move |variable| {
            self.bound[variable].map(|term| (variable, term))
        })
    }

    pub fn graph(
        &self,
        graph: &mut Graph,
        symbols: &Symbols,
        terms: &Terms,
        term: Id<Term>,
    ) -> Id<Node> {
        let (term, view) = self.view(symbols, terms, term);
        if let Some(node) = graph.get_term(term) {
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
        graph.set_term(term, node);
        node
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
        self.bound[x].unwrap_or_else(|| x.transmute())
    }
}
