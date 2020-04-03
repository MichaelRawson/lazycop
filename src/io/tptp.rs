use crate::io::exit;
use crate::io::szs;
use crate::prelude::*;
use crate::util::problem_builder::ProblemBuilder;
use std::fmt;
use std::io::Read;
use tptp::parsers::TPTPIterator;
use tptp::syntax as ast;
use tptp::visitor::Visitor;

const BUFSIZE: usize = 1024;

fn report_inappropriate<T: fmt::Display>(t: T) -> ! {
    println!("% unsupported input feature: {}", t);
    szs::inappropriate();
    exit::failure()
}

#[derive(Default)]
struct TPTPProblemBuilder {
    builder: ProblemBuilder,
}

impl TPTPProblemBuilder {
    fn finish(self) -> Problem {
        self.builder.finish()
    }
}

impl<'v> Visitor<'v> for TPTPProblemBuilder {
    fn visit_variable(&mut self, variable: ast::Variable) {
        self.builder.variable(format!("{}", variable));
    }

    fn visit_fof_plain_term(&mut self, fof_plain_term: ast::FofPlainTerm) {
        match fof_plain_term {
            ast::FofPlainTerm::Constant(c) => {
                self.builder.function(format!("{}", c), 0);
            }
            ast::FofPlainTerm::Function(f, args) => {
                let arity = args.0.len() as u32;
                for arg in args.0 {
                    self.visit_fof_term(arg);
                }
                self.builder.function(format!("{}", f), arity);
            }
        }
    }

    fn visit_literal(&mut self, literal: ast::Literal) {
        match literal {
            ast::Literal::Atomic(ast::FofAtomicFormula::Plain(p)) => {
                self.visit_fof_plain_atomic_formula(p);
                self.builder.predicate(true);
            }
            ast::Literal::Atomic(ast::FofAtomicFormula::Defined(
                ast::FofDefinedAtomicFormula::Infix(infix),
            )) => {
                self.visit_fof_term(infix.left);
                self.visit_fof_term(infix.right);
                self.builder.equality(true);
            }
            ast::Literal::NegatedAtomic(ast::FofAtomicFormula::Plain(p)) => {
                self.visit_fof_plain_atomic_formula(p);
                self.builder.predicate(false);
            }
            ast::Literal::NegatedAtomic(ast::FofAtomicFormula::Defined(
                ast::FofDefinedAtomicFormula::Infix(infix),
            )) => {
                self.visit_fof_term(infix.left);
                self.visit_fof_term(infix.right);
                self.builder.equality(false);
            }
            ast::Literal::Infix(infix) => {
                self.visit_fof_term(infix.left);
                self.visit_fof_term(infix.right);
                self.builder.equality(false);
            }
            ast::Literal::Atomic(ast::FofAtomicFormula::Defined(d)) => {
                report_inappropriate(d)
            }
            ast::Literal::Atomic(ast::FofAtomicFormula::System(s)) => {
                report_inappropriate(s)
            }
            _ => todo!(),
        }
    }

    fn visit_cnf_annotated(&mut self, annotated: ast::CnfAnnotated) {
        self.visit_cnf_formula(annotated.formula);
        let start_clause =
            annotated.role == ast::FormulaRole::NegatedConjecture;
        self.builder.clause(start_clause);
    }

    fn visit_fof_annotated(&mut self, annotated: ast::FofAnnotated) {
        report_inappropriate(annotated)
    }
}

fn read_stdin_chunk(buf: &mut Vec<u8>) -> usize {
    let mut tmp = [0; BUFSIZE];
    let read = std::io::stdin().lock().read(&mut tmp).unwrap_or_else(|e| {
        println!("% error reading from stdin: {}", e);
        szs::os_error();
        exit::failure()
    });
    buf.extend_from_slice(&tmp[0..read]);
    read
}

pub fn load_from_stdin() -> Problem {
    let mut builder = TPTPProblemBuilder::default();
    let mut buf = vec![];

    while read_stdin_chunk(&mut buf) > 0 {
        let parser = TPTPIterator::<()>::new(&buf);
        for result in parser {
            let input = result.unwrap_or_else(|_| {
                println!("% unsupported syntax");
                szs::input_error();
                exit::failure()
            });
            builder.visit_tptp_input(input);
        }
    }
    builder.finish()
}

/*
struct PrintSymbol<'symbols>(pub &'symbols SymbolTable, pub Id<Symbol>);

impl fmt::Display for PrintSymbol<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let PrintSymbol(symbol_table, symbol_id) = self;
        write!(f, "{}", symbol_table.name(*symbol_id))
    }
}

struct PrintTerm<'symbols, 'terms>(
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

struct PrintLiteral<'symbols, 'terms, 'literal>(
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

struct PrintClause<'symbols, 'terms, 'clause>(
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
*/
