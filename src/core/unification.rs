use crate::prelude::*;

fn occurs(
    symbol_table: &SymbolTable,
    term_graph: &TermGraph,
    variable: Id<Term>,
    term: Id<Term>,
) -> bool {
    let mut check = vec![term];
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
}

pub fn might_unify(
    symbol_table: &SymbolTable,
    term_graph: &TermGraph,
    left: Id<Term>,
    right: Id<Term>,
) -> bool {
    let mut constraints = vec![(left, right)];
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
}

pub fn generate_disequations<'symbol, 'term, 'iterator>(
    symbol_table: &'symbol SymbolTable,
    term_graph: &'term mut TermGraph,
    left: Id<Term>,
    right: Id<Term>,
) -> impl Iterator<Item = (Id<Term>, Id<Term>)> + 'iterator
where
    'symbol: 'iterator,
    'term: 'iterator,
{
    let mut constraints = vec![(left, right)];
    std::iter::from_fn(move || {
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
}

pub trait UnificationPolicy {
    fn unify(
        symbol_table: &SymbolTable,
        term_graph: &mut TermGraph,
        left: Id<Term>,
        right: Id<Term>,
    ) -> bool;
}

pub struct Safe;

impl UnificationPolicy for Safe {
    fn unify(
        symbol_table: &SymbolTable,
        term_graph: &mut TermGraph,
        left: Id<Term>,
        right: Id<Term>,
    ) -> bool {
        let mut constraints = vec![(left, right)];
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
    }
}

pub struct Fast;

impl UnificationPolicy for Fast {
    fn unify(
        symbol_table: &SymbolTable,
        term_graph: &mut TermGraph,
        left: Id<Term>,
        right: Id<Term>,
    ) -> bool {
        let mut constraints = vec![(left, right)];
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
    }
}
