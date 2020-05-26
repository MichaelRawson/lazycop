use crate::io::exit;
use crate::io::szs;
use crate::prelude::*;
use crate::record::Record;
use crate::util::fresh::Fresh;
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

pub(crate) fn load_from_stdin() -> Problem {
    let mut builder = TPTPProblemBuilder::default();
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

fn print_symbol(symbols: &Symbols, symbol: Id<Symbol>) {
    print!("{}", symbols.name(symbol));
}

fn print_term(
    variable_map: &mut Fresh,
    symbols: &Symbols,
    terms: &Terms,
    term: Id<Term>,
) {
    let mut arg_stack = vec![vec![term]];
    while let Some(args) = arg_stack.last_mut() {
        if let Some(next_arg) = args.pop() {
            let mut needs_comma = !args.is_empty();
            match terms.view(next_arg) {
                (_, TermView::Variable(x)) => {
                    print!("X{}", variable_map.get(x));
                }
                (_, TermView::Function(symbol, new_args)) => {
                    print_symbol(symbols, symbol);
                    let mut new_args: Vec<_> = new_args.collect();
                    if !new_args.is_empty() {
                        print!("(");
                        needs_comma = false;
                        new_args.reverse();
                        arg_stack.push(new_args);
                    }
                }
            }
            if needs_comma {
                print!(",");
            }
        } else {
            arg_stack.pop();
            if let Some(args) = arg_stack.last_mut() {
                print!(")");
                if !args.is_empty() {
                    print!(",");
                }
            }
        }
    }
}

fn print_literal(
    variable_map: &mut Fresh,
    symbols: &Symbols,
    terms: &Terms,
    literal: Literal,
) {
    match (literal.polarity, literal.atom) {
        (true, Atom::Predicate(p)) => {
            print_term(variable_map, symbols, terms, p);
        }
        (false, Atom::Predicate(p)) => {
            print!("~");
            print_term(variable_map, symbols, terms, p);
        }
        (true, Atom::Equality(left, right)) => {
            print_term(variable_map, symbols, terms, left);
            print!(" = ");
            print_term(variable_map, symbols, terms, right)
        }
        (false, Atom::Equality(left, right)) => {
            print_term(variable_map, symbols, terms, left);
            print!(" != ");
            print_term(variable_map, symbols, terms, right)
        }
    }
}

fn print_literals(
    variable_map: &mut Fresh,
    symbols: &Symbols,
    terms: &Terms,
    literals: &Block<Literal>,
    mut range: Range<Literal>,
) {
    if let Some(id) = range.next() {
        let literal = literals[id];
        print_literal(variable_map, symbols, terms, literal);
    } else {
        print!("$false");
        return;
    }
    for id in range {
        let literal = literals[id];
        print!(" | ");
        print_literal(variable_map, symbols, terms, literal);
    }
}

#[derive(Default)]
pub(crate) struct TPTPProof {
    variable_map: Fresh,
    clause_stack: Vec<usize>,
    assumptions_list: Vec<usize>,
    clause_number: usize,
}

impl Record for TPTPProof {
    fn copy(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        literals: &Block<Literal>,
        clause: &Clause,
    ) {
        print!("cnf({}, plain, ", self.clause_number);
        print_literals(
            &mut self.variable_map,
            symbols,
            terms,
            literals,
            clause.open(),
        );
        println!(").");
        self.clause_stack.push(self.clause_number);
        self.clause_number += 1;
    }

    fn predicate_reduction(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        literals: &Block<Literal>,
        clause: &Clause,
        left: Id<Term>,
        right: Id<Term>,
    ) {
        let parent = self
            .clause_stack
            .pop()
            .expect("reduction on empty clause stack");
        print!("cnf({}, assumption, ", self.clause_number);
        print_term(&mut self.variable_map, symbols, terms, left);
        print!(" = ");
        print_term(&mut self.variable_map, symbols, terms, right);
        println!(").");
        self.assumptions_list.push(self.clause_number);
        self.clause_number += 1;
        print!("cnf({}, plain, ", self.clause_number);
        print_literals(
            &mut self.variable_map,
            symbols,
            terms,
            literals,
            clause.open(),
        );
        if !clause.is_empty() {
            self.clause_stack.push(self.clause_number);
        }
        println!(
            ", inference(reduction, [assumptions([{}])], [{}])).",
            self.clause_number - 1,
            parent
        );
        self.clause_number += 1;
    }

    fn predicate_extension(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        literals: &Block<Literal>,
        clause: &Clause,
        new_clause: &Clause,
    ) {
        let copy = self.clause_stack.pop().unwrap();
        let parent = self.clause_stack.pop().unwrap();
        print!("cnf({}, plain, ", self.clause_number);
        print_literals(
            &mut self.variable_map,
            symbols,
            terms,
            literals,
            clause.remaining(),
        );
        if !clause.is_empty() {
            self.clause_stack.push(self.clause_number);
        }
        println!(
            ", inference(predicate_extension, [], [{}, {}])).",
            parent, copy
        );
        self.clause_number += 1;

        print!("cnf({}, plain, ", self.clause_number);
        print_literals(
            &mut self.variable_map,
            symbols,
            terms,
            literals,
            new_clause.open(),
        );
        println!(
            ", inference(predicate_extension, [], [{}, {}])).",
            parent, copy
        );
        self.clause_stack.push(self.clause_number);
        self.clause_number += 1;
    }

    fn reflexivity(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        literals: &Block<Literal>,
        clause: &Clause,
        left: Id<Term>,
        right: Id<Term>,
    ) {
        let parent = self.clause_stack.pop().unwrap();
        print!("cnf({}, assumption, ", self.clause_number);
        print_term(&mut self.variable_map, symbols, terms, left);
        print!(" = ");
        print_term(&mut self.variable_map, symbols, terms, right);
        println!(").");
        self.assumptions_list.push(self.clause_number);
        self.clause_number += 1;

        print!("cnf({}, plain, ", self.clause_number);
        print_literals(
            &mut self.variable_map,
            symbols,
            terms,
            literals,
            clause.open(),
        );
        if !literals.is_empty() {
            self.clause_stack.push(self.clause_number);
        }
        println!(
            ", inference(reflexivity, [assumptions([{}])], [{}])).",
            self.clause_number - 1,
            parent
        );
        self.clause_number += 1;
    }

    /*
    fn unification(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &IdMap<Variable, Option<Id<Term>>>,
    ) {
        print!(
            "cnf({}, plain, $false, inference(unification, [",
            self.clause_number,
        );
        let mut after_first_bind = false;
        for id in bindings {
            if let Some(bound) = bindings[id] {
                if after_first_bind {
                    print!(", ");
                }
                print!("bind(X{}, ", self.variable_map.get(id));
                print_term(
                    &mut self.variable_map,
                    literals,
                    terms,
                    bound,
                );
                print!(")");
                after_first_bind = true;
            }
        }
        println!("], {:?})).", self.assumptions_list);
        self.clause_stack.clear();
        self.assumptions_list.clear();
    }
    */
}
