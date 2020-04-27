use crate::prelude::*;

pub(crate) struct Variable;

#[derive(Clone, Copy)]
pub(crate) enum TermView {
    Variable(Id<Variable>),
    Function(Id<Symbol>, IdRange<Term>),
}

#[derive(Clone, Copy)]
pub(crate) enum Term {
    Symbol(Id<Symbol>, u32),
    Reference(Offset<Term>),
}

#[derive(Default)]
pub(crate) struct TermGraph {
    arena: Arena<Term>,
    mark: Id<Term>,
}

impl TermGraph {
    pub(crate) fn clear(&mut self) {
        self.arena.clear();
    }

    pub(crate) fn len(&self) -> usize {
        self.arena.len()
    }

    pub(crate) fn current_offset(&self) -> Offset<Term> {
        self.arena.limit().as_offset()
    }

    pub(crate) fn extend_from(&mut self, other: &Self) {
        self.arena.extend_from(&other.arena);
    }

    pub(crate) fn mark(&mut self) {
        self.mark = self.arena.limit();
    }

    pub(crate) fn undo_to_mark(&mut self) {
        self.arena.truncate(self.mark);
    }

    pub(crate) fn add_variable(&mut self) -> Id<Term> {
        let id = self.arena.limit();
        self.add_reference(id)
    }

    pub(crate) fn add_function(
        &mut self,
        symbol: Id<Symbol>,
        args: &[Id<Term>],
    ) -> Id<Term> {
        let id = self.arena.limit();
        self.arena.push(Term::Symbol(symbol, args.len() as u32));
        for arg in args {
            self.add_reference(*arg);
        }
        id
    }

    pub(crate) fn view(&self, id: Id<Term>) -> (Id<Term>, TermView) {
        let id = self.resolve_reference(id);
        (id, self.view_no_ref(id))
    }

    pub(crate) fn view_no_ref(&self, id: Id<Term>) -> TermView {
        match self.arena[id] {
            Term::Symbol(symbol, arity) => {
                let args = IdRange::new_after(id, arity);
                TermView::Function(symbol, args)
            }
            Term::Reference(_) => TermView::Variable(id.transmute()),
        }
    }

    fn add_reference(&mut self, referred: Id<Term>) -> Id<Term> {
        let id = self.arena.limit();
        let offset = referred - id;
        self.arena.push(Term::Reference(offset));
        id
    }

    fn resolve_reference(&self, id: Id<Term>) -> Id<Term> {
        match self.arena[id] {
            Term::Reference(offset) => id + offset,
            _ => id,
        }
    }
}
