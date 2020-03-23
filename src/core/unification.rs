use crate::prelude::*;
use std::cell::RefCell;

thread_local! {
    static OCCURS_BUF: RefCell<Vec<Id<Term>>> = RefCell::new(vec![]);
    static MIGHT_UNIFY_BUF: RefCell<Vec<(Id<Term>, Id<Term>)>> =
        RefCell::new(vec![]);
    static UNIFY_CHECKED_BUF: RefCell<Vec<(Id<Term>, Id<Term>)>> =
        RefCell::new(vec![]);
    static UNIFY_UNCHECKED_BUF: RefCell<Vec<(Id<Term>, Id<Term>)>> =
        RefCell::new(vec![]);
    static UNIFY_OR_DISEQUATION_BUF: RefCell<Vec<(Id<Term>, Id<Term>)>> =
        RefCell::new(vec![]);
}

fn occurs(
    symbol_table: &SymbolTable,
    term_graph: &TermGraph,
    variable: Id<Term>,
    term: Id<Term>,
) -> bool {
    OCCURS_BUF.with(|check| {
        let mut check = check.borrow_mut();
        check.clear();
        check.push(term);
        while let Some(term) = check.pop() {
            match term_graph.view(symbol_table, term) {
                TermView::Variable(other) => {
                    if variable == other {
                        return true;
                    }
                }
                TermView::Function(_, args) => {
                    check.extend(args);
                }
            }
        }
        false
    })
}

pub fn might_unify(
    symbol_table: &SymbolTable,
    term_graph: &TermGraph,
    left: Id<Term>,
    right: Id<Term>,
) -> bool {
    MIGHT_UNIFY_BUF.with(|constraints| {
        let mut constraints = constraints.borrow_mut();
        constraints.clear();
        constraints.push((left, right));
        while let Some((left, right)) = constraints.pop() {
            if left == right {
                continue;
            }

            let left_view = term_graph.view(symbol_table, left);
            let right_view = term_graph.view(symbol_table, right);
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
    })
}

pub fn unify_checked(
    symbol_table: &SymbolTable,
    term_graph: &mut TermGraph,
    left: Id<Term>,
    right: Id<Term>,
) -> bool {
    UNIFY_CHECKED_BUF.with(|constraints| {
        let mut constraints = constraints.borrow_mut();
        constraints.clear();
        constraints.push((left, right));
        while let Some((left, right)) = constraints.pop() {
            if left == right {
                continue;
            }

            let left_view = term_graph.view(symbol_table, left);
            let right_view = term_graph.view(symbol_table, right);
            match (left_view, right_view) {
                (TermView::Variable(x), TermView::Variable(y)) => {
                    term_graph.bind(x, y);
                }
                (TermView::Variable(variable), TermView::Function(_, _)) => {
                    if occurs(symbol_table, term_graph, variable, right) {
                        return false;
                    }
                    term_graph.bind(variable, right);
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
    })
}

pub fn unify_unchecked(
    symbol_table: &SymbolTable,
    term_graph: &mut TermGraph,
    left: Id<Term>,
    right: Id<Term>,
) -> bool {
    UNIFY_UNCHECKED_BUF.with(|constraints| {
        let mut constraints = constraints.borrow_mut();
        constraints.clear();
        constraints.push((left, right));
        while let Some((left, right)) = constraints.pop() {
            if left == right {
                continue;
            }

            let left_view = term_graph.view(symbol_table, left);
            let right_view = term_graph.view(symbol_table, right);
            match (left_view, right_view) {
                (TermView::Variable(x), TermView::Variable(y)) => {
                    term_graph.bind(x, y);
                }
                (TermView::Variable(variable), TermView::Function(_, _)) => {
                    term_graph.bind(variable, right);
                }
                (TermView::Function(_, _), TermView::Variable(variable)) => {
                    term_graph.bind(variable, left);
                }
                (TermView::Function(_, ts), TermView::Function(_, ss)) => {
                    constraints.extend(ts.zip(ss));
                }
            }
        }
        true
    })
}

pub fn unify_or_disequations<'symbol, 'term, 'iterator>(
    symbol_table: &'symbol SymbolTable,
    term_graph: &'term mut TermGraph,
    left: Id<Term>,
    right: Id<Term>,
) -> impl Iterator<Item = (Id<Term>, Id<Term>)> + 'iterator
where
    'symbol: 'iterator,
    'term: 'iterator,
{
    UNIFY_OR_DISEQUATION_BUF.with(|constraints| {
        let mut constraints = constraints.borrow_mut();
        constraints.clear();
        constraints.push((left, right));
    });
    std::iter::from_fn(move || {
        UNIFY_OR_DISEQUATION_BUF.with(|constraints| {
            let mut constraints = constraints.borrow_mut();
            while let Some((left, right)) = constraints.pop() {
                if left == right {
                    continue;
                }

                let left_view = term_graph.view(symbol_table, left);
                let right_view = term_graph.view(symbol_table, right);
                match (left_view, right_view) {
                    (TermView::Variable(x), TermView::Variable(y)) => {
                        term_graph.bind(x, y);
                    }
                    (
                        TermView::Variable(variable),
                        TermView::Function(_, _),
                    ) => {
                        if occurs(symbol_table, term_graph, variable, right) {
                            return Some((left, right));
                        } else {
                            term_graph.bind(variable, right);
                        }
                    }
                    (TermView::Function(_, _), TermView::Variable(_)) => {
                        constraints.push((right, left));
                    }
                    (TermView::Function(f, ts), TermView::Function(g, ss)) => {
                        if f == g {
                            assert_eq!(ts.len(), ss.len());
                            constraints.extend(ts.zip(ss));
                        } else {
                            return Some((left, right));
                        }
                    }
                }
            }
            None
        })
    })
}
