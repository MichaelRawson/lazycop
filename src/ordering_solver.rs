use crate::binding::Bindings;
use crate::prelude::*;
use std::cmp::Ordering;

fn symbol_precedence(
    symbols: &Symbols,
    left: Id<Symbol>,
    right: Id<Symbol>,
) -> Ordering {
    let left_arity = symbols.arity(left);
    let right_arity = symbols.arity(right);
    left_arity.cmp(&right_arity).then_with(|| left.cmp(&right))
}

fn lpo_args_gt(
    symbols: &Symbols,
    terms: &Terms,
    bindings: &Bindings,
    left: Id<Term>,
    args: Range<Argument>,
) -> Option<bool> {
    let mut sure = true;
    for arg in args {
        let arg = terms.resolve(arg);
        match lpo(symbols, terms, bindings, left, arg) {
            Some(Ordering::Greater) => {}
            None => {
                sure = false;
            }
            _ => {
                return Some(false);
            }
        }
    }
    if sure {
        Some(true)
    } else {
        None
    }
}

fn lpo(
    symbols: &Symbols,
    terms: &Terms,
    bindings: &Bindings,
    left: Id<Term>,
    right: Id<Term>,
) -> Option<Ordering> {
    let (left, lview) = bindings.view(symbols, terms, left);
    let (right, rview) = bindings.view(symbols, terms, right);
    if left == right {
        return Some(Ordering::Equal);
    }

    let (f, ss) = match lview {
        TermView::Variable(_) => {
            if terms.is_variable(right) {
                return None;
            } else {
                return lpo(symbols, terms, bindings, right, left)
                    .map(|ord| ord.reverse());
            }
        }
        TermView::Function(f, ss) => (f, ss),
    };

    let mut lpo_1 = Some(Ordering::Less);
    for s in ss {
        let s = terms.resolve(s);
        match lpo(symbols, terms, bindings, s, right) {
            Some(Ordering::Equal) | Some(Ordering::Greater) => {
                return Some(Ordering::Greater);
            }
            Some(Ordering::Less) => {}
            None => {
                lpo_1 = None;
            }
        }
    }

    let (g, ts) = match rview {
        TermView::Variable(_) => return None,
        TermView::Function(g, ts) => (g, ts),
    };

    match symbol_precedence(symbols, f, g) {
        Ordering::Less => lpo_1,
        Ordering::Equal => {
            if !lpo_args_gt(symbols, terms, bindings, left, ts)? {
                return lpo_1;
            }
            for (s, t) in ss.zip(ts) {
                let s = terms.resolve(s);
                let t = terms.resolve(t);
                match lpo(symbols, terms, bindings, s, t)? {
                    Ordering::Less => {
                        return lpo_1;
                    }
                    Ordering::Equal => {}
                    Ordering::Greater => {
                        return Some(Ordering::Greater);
                    }
                }
            }
            Some(Ordering::Equal)
        }
        Ordering::Greater => {
            if lpo_args_gt(symbols, terms, bindings, left, ts)? {
                Some(Ordering::Greater)
            } else {
                lpo_1
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct OrderingSolver;

impl OrderingSolver {
    pub(crate) fn clear(&mut self) {}

    pub(crate) fn save(&mut self) {}

    pub(crate) fn restore(&mut self) {}

    pub(crate) fn check<I: Iterator<Item = (Id<Term>, Id<Term>)>>(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
        gt: I,
    ) -> bool {
        gt.filter_map(|(left, right)| {
            lpo(symbols, terms, bindings, left, right)
        })
        .all(|ordering| ordering == Ordering::Greater)
    }
}
