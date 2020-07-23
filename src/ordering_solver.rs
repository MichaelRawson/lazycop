use crate::binding::Bindings;
use crate::prelude::*;
use std::cmp::Ordering;

// algorithm and function names are lpo_6 from Loechner's PhD thesis
// with thanks to Bernd Loechner and Stephan Schulz

fn symbol_precedence(
    symbols: &Symbols,
    left: Id<Symbol>,
    right: Id<Symbol>,
) -> Ordering {
    let left_arity = symbols[left].arity;
    let right_arity = symbols[right].arity;
    left_arity.cmp(&right_arity).then_with(|| right.cmp(&left))
}

fn alpha(
    symbols: &Symbols,
    terms: &Terms,
    bindings: &Bindings,
    ss: Range<Argument>,
    t: Id<Term>,
) -> Option<Ordering> {
    let any_gte = ss
        .map(|s_i| terms.resolve(s_i))
        .filter_map(|s_i| lpo(symbols, terms, bindings, s_i, t))
        .any(|ordering| ordering != Ordering::Less);
    if any_gte {
        Some(Ordering::Greater)
    } else {
        None
    }
}

fn flip(ordering: Option<Ordering>) -> Option<Ordering> {
    ordering.map(|ordering| ordering.reverse())
}

fn ma(
    symbols: &Symbols,
    terms: &Terms,
    bindings: &Bindings,
    s: Id<Term>,
    ts: Range<Argument>,
) -> Option<Ordering> {
    for t in ts {
        let t = terms.resolve(t);
        match lpo(symbols, terms, bindings, s, t) {
            Some(Ordering::Greater) => {}
            Some(_) => return Some(Ordering::Less),
            None => return flip(alpha(symbols, terms, bindings, ts, s)),
        }
    }
    Some(Ordering::Greater)
}

fn lma(
    symbols: &Symbols,
    terms: &Terms,
    bindings: &Bindings,
    s: Id<Term>,
    t: Id<Term>,
    mut ss: Range<Argument>,
    mut ts: Range<Argument>,
) -> Option<Ordering> {
    while let (Some(s_i), Some(t_i)) = (ss.next(), ts.next()) {
        let s_i = terms.resolve(s_i);
        let t_i = terms.resolve(t_i);
        match lpo(symbols, terms, bindings, s_i, t_i) {
            Some(Ordering::Less) => {
                return flip(ma(symbols, terms, bindings, t, ss))
            }
            Some(Ordering::Equal) => {
                continue;
            }
            Some(Ordering::Greater) => {
                return ma(symbols, terms, bindings, s, ts);
            }
            None => {
                return aa(symbols, terms, bindings, s, t, ss, ts);
            }
        }
    }
    Some(Ordering::Equal)
}

fn aa(
    symbols: &Symbols,
    terms: &Terms,
    bindings: &Bindings,
    s: Id<Term>,
    t: Id<Term>,
    ss: Range<Argument>,
    ts: Range<Argument>,
) -> Option<Ordering> {
    alpha(symbols, terms, bindings, ss, t)
        .or_else(|| flip(alpha(symbols, terms, bindings, ts, s)))
}

fn lpo(
    symbols: &Symbols,
    terms: &Terms,
    bindings: &Bindings,
    s: Id<Term>,
    t: Id<Term>,
) -> Option<Ordering> {
    let (s, sview) = bindings.view(symbols, terms, s);
    let (t, tview) = bindings.view(symbols, terms, t);
    match (sview, tview) {
        (TermView::Variable(x), TermView::Variable(y)) => {
            if x == y {
                Some(Ordering::Equal)
            } else {
                None
            }
        }
        (TermView::Variable(x), TermView::Function(_, _)) => {
            if bindings.occurs(symbols, terms, x, t) {
                Some(Ordering::Less)
            } else {
                None
            }
        }
        (TermView::Function(_, _), TermView::Variable(x)) => {
            if bindings.occurs(symbols, terms, x, s) {
                Some(Ordering::Greater)
            } else {
                None
            }
        }
        (TermView::Function(f, ss), TermView::Function(g, ts)) => {
            match symbol_precedence(symbols, f, g) {
                Ordering::Less => flip(ma(symbols, terms, bindings, t, ss)),
                Ordering::Equal => lma(symbols, terms, bindings, s, t, ss, ts),
                Ordering::Greater => ma(symbols, terms, bindings, s, ts),
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct OrderingSolver {
    remaining: Vec<(Id<Term>, Id<Term>)>,
    save: usize,
}

impl OrderingSolver {
    pub(crate) fn clear(&mut self) {
        self.remaining.clear();
    }

    pub(crate) fn save(&mut self) {
        self.save = self.remaining.len();
    }

    pub(crate) fn restore(&mut self) {
        self.remaining.truncate(self.save);
    }

    pub(crate) fn check(
        &self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
    ) -> bool {
        self.remaining
            .iter()
            .rev()
            .copied()
            .filter_map(|(left, right)| {
                lpo(symbols, terms, bindings, left, right)
            })
            .all(|ordering| ordering == Ordering::Greater)
    }

    pub(crate) fn simplify<I: Iterator<Item = (Id<Term>, Id<Term>)>>(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
        orderings: I,
    ) -> bool {
        for (left, right) in orderings {
            match lpo(symbols, terms, bindings, left, right) {
                None => {
                    self.remaining.push((left, right));
                }
                Some(Ordering::Greater) => {}
                Some(_) => {
                    return false;
                }
            }
        }
        true
    }
}
