use crate::prelude::*;

pub(crate) struct Variable;

#[derive(Clone, Copy)]
pub(crate) enum TermView {
    Variable(Id<Variable>),
    Function(Id<Symbol>, Range<Term>),
}

#[derive(Clone, Copy)]
pub(crate) enum Term {
    Symbol(Id<Symbol>, u32),
    Reference(Offset<Term>),
}

#[derive(Default)]
pub(crate) struct Terms {
    terms: Block<Term>,
}

impl Terms {
    pub(crate) fn clear(&mut self) {
        self.terms.clear();
    }

    pub(crate) fn len(&self) -> Id<Term> {
        self.terms.len()
    }

    pub(crate) fn current_offset(&self) -> Offset<Term> {
        self.len() - Id::default()
    }

    pub(crate) fn extend_from(&mut self, other: &Self) {
        self.terms.extend(other.terms.as_ref().iter().copied());
    }

    pub(crate) fn add_variable(&mut self) -> Id<Term> {
        let id = self.terms.len();
        self.add_reference(id)
    }

    pub(crate) fn add_function(
        &mut self,
        symbol: Id<Symbol>,
        args: &[Id<Term>],
    ) -> Id<Term> {
        let id = self.terms.len();
        self.terms.push(Term::Symbol(symbol, args.len() as u32));
        for arg in args {
            self.add_reference(*arg);
        }
        id
    }

    pub(crate) fn view(&self, id: Id<Term>) -> (Id<Term>, TermView) {
        let id = self.resolve_reference(id);
        (id, self.view_no_ref(id))
    }

    fn view_no_ref(&self, id: Id<Term>) -> TermView {
        match self.terms[id] {
            Term::Symbol(symbol, arity) => {
                let args = Range::new_with_len(id, arity);
                TermView::Function(symbol, args)
            }
            Term::Reference(_) => TermView::Variable(id.transmute()),
        }
    }

    fn add_reference(&mut self, referred: Id<Term>) -> Id<Term> {
        let id = self.terms.len();
        let offset = referred - id;
        self.terms.push(Term::Reference(offset));
        id
    }

    fn resolve_reference(&self, id: Id<Term>) -> Id<Term> {
        match self.terms[id] {
            Term::Reference(offset) => id + offset,
            _ => id,
        }
    }
}

impl Clone for Terms {
    fn clone(&self) -> Self {
        unreachable!()
    }

    fn clone_from(&mut self, other: &Self) {
        self.terms.clone_from(&other.terms);
    }
}
