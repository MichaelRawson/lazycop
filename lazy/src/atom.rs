use crate::constraint::{Constraints, SymmetricDisequation};
use crate::prelude::*;

#[derive(Clone, Copy)]
pub enum Atom {
    Predicate(Id<Term>),
    Equality(Id<Term>, Id<Term>),
}

impl Atom {
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

    pub fn is_predicate(&self) -> bool {
        match self {
            Atom::Predicate(_) => true,
            _ => false,
        }
    }

    pub fn is_equality(&self) -> bool {
        match self {
            Atom::Equality(_, _) => true,
            _ => false,
        }
    }

    pub fn get_predicate(&self) -> Id<Term> {
        match self {
            Atom::Predicate(p) => *p,
            _ => unreachable(),
        }
    }

    pub fn get_equality(&self) -> (Id<Term>, Id<Term>) {
        match self {
            Atom::Equality(left, right) => (*left, *right),
            _ => unreachable(),
        }
    }

    pub fn get_predicate_symbol(&self, terms: &Terms) -> Id<Symbol> {
        terms.symbol(self.get_predicate())
    }

    pub fn get_predicate_arguments(
        &self,
        symbols: &Symbols,
        terms: &Terms,
    ) -> Range<Argument> {
        terms.arguments(symbols, self.get_predicate())
    }

    pub fn graph(
        &self,
        graph: &mut Graph,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
    ) -> Id<Node> {
        match self {
            Atom::Predicate(p) => bindings.graph(graph, symbols, terms, *p),
            Atom::Equality(left, right) => {
                let left = bindings.graph(graph, symbols, terms, *left);
                let right = bindings.graph(graph, symbols, terms, *right);
                graph.equality(left, right)
            }
        }
    }

    pub fn subterms<F: FnMut(Id<Term>)>(
        &self,
        symbols: &Symbols,
        terms: &Terms,
        f: &mut F,
    ) {
        match self {
            Atom::Predicate(p) => {
                terms.proper_subterms(symbols, *p, f);
            }
            Atom::Equality(left, right) => {
                terms.subterms(symbols, *left, f);
                terms.subterms(symbols, *right, f);
            }
        }
    }

    pub fn subst(
        &self,
        symbols: &Symbols,
        terms: &mut Terms,
        constraints: &mut Constraints,
        from: Id<Term>,
        to: Id<Term>,
    ) -> Self {
        match self {
            Atom::Predicate(p) => {
                Atom::Predicate(terms.subst(symbols, *p, from, to))
            }
            Atom::Equality(left, right) => {
                let subst = terms.subst(symbols, *left, from, to);
                if subst != *left {
                    constraints.assert_gt(*left, *right);
                    return Atom::Equality(subst, *right);
                }
                constraints.assert_gt(*right, *left);
                let subst = terms.subst(symbols, *right, from, to);
                Atom::Equality(*left, subst)
            }
        }
    }

    pub fn add_reflexivity_constraints(&self, constraints: &mut Constraints) {
        let (left, right) = self.get_equality();
        constraints.assert_neq(left, right);
    }

    pub fn add_disequation_constraints(
        &self,
        constraints: &mut Constraints,
        terms: &Terms,
        other: &Self,
    ) {
        if self.is_predicate()
            && other.is_predicate()
            && self.get_predicate_symbol(terms)
                == other.get_predicate_symbol(terms)
        {
            constraints
                .assert_neq(self.get_predicate(), other.get_predicate());
        } else if self.is_equality() && other.is_equality() {
            let (left1, right1) = self.get_equality();
            let (left2, right2) = other.get_equality();
            let symmetric = SymmetricDisequation {
                left1,
                left2,
                right1,
                right2,
            };
            constraints.assert_symmetric_neq(symmetric);
        }
    }
}
