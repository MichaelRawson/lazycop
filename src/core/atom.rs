use crate::core::unification::{might_unify, unify};
use crate::prelude::*;

#[derive(Clone, Copy)]
pub enum Atom {
    Predicate(Id<Term>),
    Equality(Id<Term>, Id<Term>),
}

impl Atom {
    pub fn is_predicate(&self) -> bool {
        match self {
            Atom::Predicate(_) => true,
            _ => false,
        }
    }

    pub fn predicate_term(&self) -> Id<Term> {
        match self {
            Atom::Predicate(p) => *p,
            _ => unreachable!(),
        }
    }

    pub fn offset(&mut self, offset: Offset<Term>) {
        match self {
            Atom::Predicate(p) => {
                *p = *p + offset;
            }
            Atom::Equality(left, right) => {
                *left = *left + offset;
                *right = *right + offset;
            }
        }
    }

    pub fn might_unify(
        &self,
        symbol_list: &SymbolList,
        term_list: &TermList,
        other: &Self,
    ) -> bool {
        match (self, other) {
            (Atom::Predicate(p), Atom::Predicate(q)) => {
                might_unify(symbol_list, term_list, *p, *q)
            }
            _ => false,
        }
    }

    pub fn unify(
        &self,
        symbol_list: &SymbolList,
        term_list: &mut TermList,
        other: &Self,
    ) -> bool {
        match (self, other) {
            (Atom::Predicate(p), Atom::Predicate(q)) => {
                unify(symbol_list, term_list, *p, *q)
            }
            _ => unreachable!(),
        }
    }

    pub fn might_self_unify(
        &self,
        symbol_list: &SymbolList,
        term_list: &TermList,
    ) -> bool {
        match self {
            Atom::Equality(left, right) => {
                might_unify(symbol_list, term_list, *left, *right)
            }
            _ => false,
        }
    }

    pub fn self_unify(
        &self,
        symbol_list: &SymbolList,
        term_list: &mut TermList,
    ) -> bool {
        match self {
            Atom::Equality(left, right) => {
                unify(symbol_list, term_list, *left, *right)
            }
            _ => unreachable!(),
        }
    }

    pub fn lazy_constraints<'symbol, 'term, 'iterator>(
        &self,
        symbol_list: &'symbol SymbolList,
        term_list: &'term TermList,
        other: &Self,
    ) -> impl Iterator<Item = Self> + 'iterator
    where
        'symbol: 'iterator,
        'term: 'iterator,
    {
        match (self, other) {
            (Atom::Predicate(p), Atom::Predicate(q)) => {
                let p_view = term_list.view(symbol_list, *p);
                let q_view = term_list.view(symbol_list, *q);
                match (p_view, q_view) {
                    (
                        TermView::Function(p, pargs),
                        TermView::Function(q, qargs),
                    ) => {
                        assert!(p == q);
                        assert!(pargs.len() == qargs.len());
                        pargs
                            .zip(qargs)
                            .filter(move |(t, s)| {
                                !term_list.equal(symbol_list, *t, *s)
                            })
                            .map(|(t, s)| Atom::Equality(t, s))
                    }
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        }
    }
}
