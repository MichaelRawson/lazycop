use crate::prelude::*;
use smallvec::SmallVec;

type Stack<T> = SmallVec<[T; 128]>;

fn occurs(
    symbol_table: &SymbolTable,
    term_graph: &TermGraph,
    variable: Id<Term>,
    term: Id<Term>,
) -> bool {
    let mut check: Stack<_> = smallvec![term];
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

pub fn equal_terms(
    symbol_table: &SymbolTable,
    term_graph: &TermGraph,
    left: Id<Term>,
    right: Id<Term>,
) -> bool {
    let mut constraints: Stack<_> = smallvec![(left, right)];
    while let Some((left, right)) = constraints.pop() {
        if left == right {
            continue;
        }

        let left_view = term_graph.view(symbol_table, left);
        let right_view = term_graph.view(symbol_table, right);
        match (left_view, right_view) {
            (TermView::Variable(x), TermView::Variable(y)) if x == y => {}
            (TermView::Function(f, ts), TermView::Function(g, ss))
                if f == g =>
            {
                constraints.extend(ts.zip(ss));
            }
            _ => {
                return false;
            }
        }
    }
    true
}

pub fn might_unify(
    symbol_table: &SymbolTable,
    term_graph: &TermGraph,
    left: Id<Term>,
    right: Id<Term>,
) -> bool {
    let mut constraints: Stack<_> = smallvec![(left, right)];
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
                constraints.extend(ts.zip(ss));
            } else {
                return false;
            }
        }
    }
    true
}

pub fn unify_checked(
    symbol_table: &SymbolTable,
    term_graph: &mut TermGraph,
    left: Id<Term>,
    right: Id<Term>,
) -> bool {
    let mut constraints: Stack<_> = smallvec![(left, right)];
    while let Some((left, right)) = constraints.pop() {
        if left == right {
            continue;
        }

        let left_view = term_graph.view(symbol_table, left);
        let right_view = term_graph.view(symbol_table, right);
        match (left_view, right_view) {
            (TermView::Variable(x), TermView::Variable(y)) => {
                term_graph.bind_vars(x, y);
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
                    constraints.extend(ts.zip(ss));
                } else {
                    return false;
                }
            }
        }
    }
    true
}

pub fn unify_unchecked(
    symbol_table: &SymbolTable,
    term_graph: &mut TermGraph,
    left: Id<Term>,
    right: Id<Term>,
) -> bool {
    let mut constraints: Stack<_> = smallvec![(left, right)];
    while let Some((left, right)) = constraints.pop() {
        if left == right {
            continue;
        }

        let left_view = term_graph.view(symbol_table, left);
        let right_view = term_graph.view(symbol_table, right);
        match (left_view, right_view) {
            (TermView::Variable(x), TermView::Variable(y)) => {
                term_graph.bind_vars(x, y);
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

pub fn unify_or_disequations(
    symbol_table: &SymbolTable,
    term_graph: &mut TermGraph,
    left: Id<Term>,
    right: Id<Term>,
) -> Vec<Literal> {
    let mut literals = vec![];
    let mut constraints: Stack<_> = smallvec![(left, right)];
    while let Some((left, right)) = constraints.pop() {
        if left == right {
            continue;
        }

        let left_view = term_graph.view(symbol_table, left);
        let right_view = term_graph.view(symbol_table, right);
        let is_disequation = match (left_view, right_view) {
            (TermView::Variable(x), TermView::Variable(y)) => {
                term_graph.bind_vars(x, y);
                false
            }
            (TermView::Variable(variable), TermView::Function(_, _)) => {
                if occurs(symbol_table, term_graph, variable, right) {
                    true
                } else {
                    term_graph.bind(variable, right);
                    false
                }
            }
            (TermView::Function(_, _), TermView::Variable(_)) => {
                constraints.push((right, left));
                false
            }
            (TermView::Function(f, ts), TermView::Function(g, ss)) => {
                if f == g {
                    constraints.extend(ts.zip(ss).filter(|(t, s)| t != s));
                    false
                } else {
                    true
                }
            }
        };
        if is_disequation {
            literals.push(Literal::new(false, Atom::Equality(left, right)));
        }
    }
    literals
}
