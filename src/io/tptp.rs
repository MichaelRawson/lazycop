use crate::io::exit;
use crate::io::szs;
use crate::prelude::*;
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
struct TPTP {
    builder: ProblemBuilder,
}

impl TPTP {
    fn finish(self) -> Problem {
        self.builder.finish()
    }
}

impl<'v> Visitor<'v> for TPTP {
    fn visit_variable(&mut self, variable: &ast::Variable) {
        self.builder.variable(format!("{}", variable));
    }

    fn visit_fof_plain_term(&mut self, fof_plain_term: &ast::FofPlainTerm) {
        match fof_plain_term {
            ast::FofPlainTerm::Constant(c) => {
                self.builder.function(format!("{}", c), 0);
            }
            ast::FofPlainTerm::Function(f, args) => {
                let arity = args.0.len() as u32;
                for arg in &args.0 {
                    self.visit_fof_term(arg);
                }
                self.builder.function(format!("{}", f), arity);
            }
        }
    }

    fn visit_fof_defined_term(
        &mut self,
        fof_defined_term: &ast::FofDefinedTerm,
    ) {
        self.builder.function(format!("{}", fof_defined_term), 0);
    }

    fn visit_literal(&mut self, literal: &ast::Literal) {
        match literal {
            ast::Literal::Atomic(ast::FofAtomicFormula::Plain(p)) => {
                self.visit_fof_plain_atomic_formula(p);
                self.builder.predicate(true);
            }
            ast::Literal::Atomic(ast::FofAtomicFormula::Defined(defined)) => {
                match &**defined {
                    ast::FofDefinedAtomicFormula::Infix(infix) => {
                        self.visit_fof_term(&infix.left);
                        self.visit_fof_term(&infix.right);
                        self.builder.equality(true);
                    }
                    defined if format!("{}", defined) == "$false" => {}
                    _ => report_inappropriate(defined),
                }
            }
            ast::Literal::NegatedAtomic(ast::FofAtomicFormula::Plain(p)) => {
                self.visit_fof_plain_atomic_formula(p);
                self.builder.predicate(false);
            }
            ast::Literal::NegatedAtomic(ast::FofAtomicFormula::Defined(
                defined,
            )) => match &**defined {
                ast::FofDefinedAtomicFormula::Infix(infix) => {
                    self.visit_fof_term(&infix.left);
                    self.visit_fof_term(&infix.right);
                    self.builder.equality(false);
                }
                _ => report_inappropriate(defined),
            },
            ast::Literal::Infix(infix) => {
                self.visit_fof_term(&infix.left);
                self.visit_fof_term(&infix.right);
                self.builder.equality(false);
            }
            ast::Literal::Atomic(ast::FofAtomicFormula::System(system)) => {
                report_inappropriate(system)
            }
            ast::Literal::NegatedAtomic(ast::FofAtomicFormula::System(
                system,
            )) => report_inappropriate(system),
        }
    }

    fn visit_cnf_annotated(&mut self, annotated: &ast::CnfAnnotated) {
        self.visit_cnf_formula(&annotated.formula);
        let start_clause =
            annotated.role == ast::FormulaRole::NegatedConjecture;
        self.builder.clause(start_clause);
    }

    fn visit_fof_annotated(&mut self, annotated: &ast::FofAnnotated) {
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
    buf.extend(&tmp[0..read]);
    read
}

pub fn load_from_stdin() -> Problem {
    let mut builder = TPTP::default();
    let mut buf = vec![];

    while read_stdin_chunk(&mut buf) > 0 {
        let mut parser = TPTPIterator::<()>::new(&buf);
        for result in &mut parser {
            let input = result.unwrap_or_else(|_| {
                println!("% unsupported syntax");
                szs::input_error();
                exit::failure()
            });
            builder.visit_tptp_input(&input);
        }
        buf = parser.remaining.to_vec();
    }
    assert!(buf.is_empty());
    builder.finish()
}
