use crate::prelude::*;
use std::fmt;

pub struct PrintSymbol<'symbols>(pub &'symbols SymbolTable, pub Id<Symbol>);

impl fmt::Display for PrintSymbol<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let PrintSymbol(symbol_table, symbol_id) = self;
        write!(f, "{}", symbol_table.name(*symbol_id))
    }
}

pub struct PrintTerm<'symbols, 'terms>(
    pub &'symbols SymbolTable,
    pub &'terms TermGraph,
    pub Id<Term>,
);

impl fmt::Display for PrintTerm<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let PrintTerm(symbol_table, term_graph, term_id) = self;
        let mut todo = vec![vec![*term_id]];

        while let Some(args) = todo.last_mut() {
            if let Some(next_arg) = args.pop() {
                let mut needs_comma = !args.is_empty();
                match term_graph.view(symbol_table, next_arg) {
                    TermView::Variable(id) => {
                        write!(f, "X{}", id.index())?;
                    }
                    TermView::Function(symbol, new_args) => {
                        write!(f, "{}", PrintSymbol(symbol_table, symbol))?;
                        let mut new_args: Vec<_> = new_args.collect();
                        if !new_args.is_empty() {
                            write!(f, "(")?;
                            needs_comma = false;
                            new_args.reverse();
                            todo.push(new_args);
                        }
                    }
                }
                if needs_comma {
                    write!(f, ",")?;
                }
            } else {
                todo.pop();
                if let Some(args) = todo.last_mut() {
                    write!(f, ")")?;
                    if !args.is_empty() {
                        write!(f, ",")?;
                    }
                }
            }
        }
        Ok(())
    }
}

pub struct PrintLiteral<'symbols, 'terms, 'literal>(
    pub &'symbols SymbolTable,
    pub &'terms TermGraph,
    pub &'literal Literal,
);

impl fmt::Display for PrintLiteral<'_, '_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let PrintLiteral(symbol_table, term_graph, literal) = self;
        match (literal.polarity, literal.atom) {
            (true, Atom::Predicate(p)) => {
                write!(f, "{}", PrintTerm(symbol_table, term_graph, p))
            }
            (false, Atom::Predicate(p)) => {
                write!(f, "~{}", PrintTerm(symbol_table, term_graph, p))
            }
            (true, Atom::Equality(left, right)) => write!(
                f,
                "{} = {}",
                PrintTerm(symbol_table, term_graph, left),
                PrintTerm(symbol_table, term_graph, right)
            ),
            (false, Atom::Equality(left, right)) => write!(
                f,
                "{} != {}",
                PrintTerm(symbol_table, term_graph, left),
                PrintTerm(symbol_table, term_graph, right)
            ),
        }
    }
}

pub struct PrintClause<'symbols, 'terms, 'clause>(
    pub &'symbols SymbolTable,
    pub &'terms TermGraph,
    pub &'clause Clause,
);

impl fmt::Display for PrintClause<'_, '_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let PrintClause(symbol_table, term_graph, clause) = self;

        let mut literals = clause.iter();
        if let Some(literal) = literals.next() {
            write!(f, "{}", PrintLiteral(symbol_table, term_graph, literal))?;
        } else {
            write!(f, "$false")?;
        }
        for literal in literals {
            write!(
                f,
                " | {}",
                PrintLiteral(symbol_table, term_graph, literal)
            )?;
        }
        Ok(())
    }
}
