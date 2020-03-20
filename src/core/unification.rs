use crate::prelude::*;
use std::mem;

fn occurs(
    symbol_list: &SymbolList,
    term_list: &TermList,
    variable: Id<Term>,
    term: Id<Term>,
) -> bool {
    let mut check = vec![term];
    let mut next = vec![];
    while !check.is_empty() {
        for term in &check {
            match term_list.view(symbol_list, *term) {
                TermView::Variable(other) => {
                    if variable == other {
                        return true;
                    }
                }
                TermView::Function(_, args) => {
                    next.extend(args);
                }
            }
        }
        next.sort_unstable();
        next.dedup();
        mem::swap(&mut check, &mut next);
        next.clear();
    }
    false
}

pub fn might_unify(
    symbol_list: &SymbolList,
    term_list: &TermList,
    left: Id<Term>,
    right: Id<Term>,
) -> bool {
    let mut constraints = vec![(left, right)];
    while let Some((left, right)) = constraints.pop() {
        if left == right {
            continue;
        }

        let left_view = term_list.view(symbol_list, left);
        let right_view = term_list.view(symbol_list, right);
        if let (TermView::Function(f, ts), TermView::Function(g, ss)) =
            (left_view, right_view)
        {
            if f == g {
                assert_eq!(ts.len(), ss.len());
                constraints.extend(ts.zip(ss));
            } else {
                return false;
            }
        }
    }
    true
}

pub fn unify(
    symbol_list: &SymbolList,
    term_list: &mut TermList,
    left: Id<Term>,
    right: Id<Term>,
) -> bool {
    let mut constraints = vec![(left, right)];
    while let Some((left, right)) = constraints.pop() {
        if left == right {
            continue;
        }

        let left_view = term_list.view(symbol_list, left);
        let right_view = term_list.view(symbol_list, right);
        match (left_view, right_view) {
            (TermView::Variable(x), TermView::Variable(y)) => {
                term_list.bind(x, y);
            }
            (TermView::Variable(variable), TermView::Function(_, _)) => {
                if occurs(symbol_list, term_list, variable, right) {
                    return false;
                }
                term_list.bind(variable, right);
            }
            (TermView::Function(_, _), TermView::Variable(_)) => {
                constraints.push((right, left));
            }
            (TermView::Function(f, ts), TermView::Function(g, ss)) => {
                if f == g {
                    assert_eq!(ts.len(), ss.len());
                    constraints.extend(ts.zip(ss));
                } else {
                    return false;
                }
            }
        }
    }
    true
}
