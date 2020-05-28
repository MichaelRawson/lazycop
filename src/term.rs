use crate::prelude::*;

pub(crate) struct Argument;
pub(crate) struct Variable;

#[derive(Clone, Copy)]
pub(crate) enum TermView {
    Variable(Id<Variable>),
    Function(Id<Symbol>, Range<Argument>),
}

#[derive(Clone, Copy)]
pub(crate) enum Term {
    Symbol(Id<Symbol>),
    Arity(u32),
    Reference(Offset<Term>),
}

impl Term {
    fn as_symbol(self) -> Id<Symbol> {
        if let Term::Symbol(symbol) = self {
            symbol
        } else {
            unreachable()
        }
    }

    fn as_arity(self) -> u32 {
        if let Term::Arity(arity) = self {
            arity
        } else {
            unreachable()
        }
    }

    fn as_reference(self) -> Offset<Term> {
        if let Term::Reference(offset) = self {
            offset
        } else {
            unreachable()
        }
    }
}

#[derive(Default)]
pub(crate) struct Terms {
    terms: Block<Term>,
}

impl Terms {
    pub(crate) fn clear(&mut self) {
        self.terms.clear();
    }

    pub(crate) fn current_offset(&self) -> Offset<Term> {
        self.terms.len() - Id::default()
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
        self.terms.push(Term::Symbol(symbol));
        self.terms.push(Term::Arity(args.len() as u32));
        for arg in args {
            self.add_reference(*arg);
        }
        id
    }

    pub(crate) fn resolve(&self, argument: Id<Argument>) -> Id<Term> {
        let id = argument.transmute();
        id + self.terms[id].as_reference()
    }

    pub(crate) fn view(&self, mut id: Id<Term>) -> TermView {
        match self.terms[id] {
            Term::Symbol(symbol) => {
                id = id + Offset::new(1);
                let arity = self.terms[id].as_arity();
                id = id + Offset::new(1);
                let args = Range::new_with_len(id.transmute(), arity);
                TermView::Function(symbol, args)
            }
            Term::Reference(_) => TermView::Variable(id.transmute()),
            _ => unreachable(),
        }
    }

    pub(crate) fn symbol(&self, id: Id<Term>) -> Id<Symbol> {
        self.terms[id].as_symbol()
    }

    fn add_reference(&mut self, referred: Id<Term>) -> Id<Term> {
        let id = self.terms.len();
        let offset = referred - id;
        self.terms.push(Term::Reference(offset));
        id
    }
}

impl Clone for Terms {
    fn clone(&self) -> Self {
        unreachable()
    }

    fn clone_from(&mut self, other: &Self) {
        self.terms.clone_from(&other.terms);
    }
}
