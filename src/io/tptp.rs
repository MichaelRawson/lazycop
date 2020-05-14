use crate::io::exit;
use crate::io::record::Record;
use crate::io::szs;
use crate::prelude::*;
use crate::util::fresh::Fresh;
use crate::util::id_map::IdMap;
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

    fn visit_fof_defined_term(
        &mut self,
        fof_defined_term: ast::FofDefinedTerm,
    ) {
        self.builder.function(format!("{}", fof_defined_term), 0);
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
            ast::Literal::Atomic(atomic) => {
                if format!("{}", &atomic) != "$false" {
                    report_inappropriate(atomic)
                }
            }
            ast::Literal::NegatedAtomic(negated) => {
                report_inappropriate(negated)
            }
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
            builder.visit_tptp_input(input);
        }
        buf = parser.remaining.to_vec();
    }
    assert!(buf.is_empty());
    builder.finish()
}

fn print_symbol(symbol_table: &SymbolTable, symbol: Id<Symbol>) {
    print!("{}", symbol_table.name(symbol));
}

fn print_term(
    variable_map: &mut Fresh,
    symbol_table: &SymbolTable,
    term_graph: &TermGraph,
    term: Id<Term>,
) {
    let mut arg_stack = vec![vec![term]];
    while let Some(args) = arg_stack.last_mut() {
        if let Some(next_arg) = args.pop() {
            let mut needs_comma = !args.is_empty();
            match term_graph.view(next_arg) {
                (_, TermView::Variable(x)) => {
                    print!("X{}", variable_map.get(x));
                }
                (_, TermView::Function(symbol, new_args)) => {
                    print_symbol(symbol_table, symbol);
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
    symbol_table: &SymbolTable,
    term_graph: &TermGraph,
    literal: Literal,
) {
    match (literal.polarity, literal.atom) {
        (true, Atom::Predicate(p)) => {
            print_term(variable_map, symbol_table, term_graph, p);
        }
        (false, Atom::Predicate(p)) => {
            print!("~");
            print_term(variable_map, symbol_table, term_graph, p);
        }
        (true, Atom::Equality(left, right)) => {
            print_term(variable_map, symbol_table, term_graph, left);
            print!(" = ");
            print_term(variable_map, symbol_table, term_graph, right)
        }
        (false, Atom::Equality(left, right)) => {
            print_term(variable_map, symbol_table, term_graph, left);
            print!(" != ");
            print_term(variable_map, symbol_table, term_graph, right)
        }
    }
}

fn print_literals(
    variable_map: &mut Fresh,
    symbol_table: &SymbolTable,
    term_graph: &TermGraph,
    clause_storage: &ClauseStorage,
    mut literals: Range<Literal>,
) {
    if let Some(literal) = literals.next() {
        let literal = clause_storage[literal];
        print_literal(variable_map, symbol_table, term_graph, literal);
    } else {
        print!("$false");
        return;
    }
    for literal in literals {
        let literal = clause_storage[literal];
        print!(" | ");
        print_literal(variable_map, symbol_table, term_graph, literal);
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
    fn start(
        &mut self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        literals: Range<Literal>,
    ) {
        print!("cnf({}, axiom, ", self.clause_number);
        print_literals(
            &mut self.variable_map,
            symbol_table,
            term_graph,
            clause_storage,
            literals,
        );
        println!(").");
        self.clause_stack.push(self.clause_number);
        self.clause_number += 1;
    }

    fn reduction(
        &mut self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        literals: Range<Literal>,
        left: Id<Term>,
        right: Id<Term>,
    ) {
        let parent = self
            .clause_stack
            .pop()
            .expect("reduction on empty clause stack");
        print!("cnf({}, assumption, ", self.clause_number);
        print_term(&mut self.variable_map, symbol_table, term_graph, left);
        print!(" = ");
        print_term(&mut self.variable_map, symbol_table, term_graph, right);
        println!(").");
        self.assumptions_list.push(self.clause_number);
        self.clause_number += 1;
        print!("cnf({}, plain, ", self.clause_number);
        print_literals(
            &mut self.variable_map,
            symbol_table,
            term_graph,
            clause_storage,
            literals,
        );
        if !literals.is_empty() {
            self.clause_stack.push(self.clause_number);
        }
        println!(
            ", inference(reduction, [assumptions([{}])], [{}])).",
            self.clause_number - 1,
            parent
        );
        self.clause_number += 1;
    }

    fn extension(
        &mut self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        literals: Range<Literal>,
        extension_literals: Range<Literal>,
        left: Id<Term>,
        right: Id<Term>,
    ) {
        let parent = self
            .clause_stack
            .pop()
            .expect("printing extension on empty stack");
        print!("cnf({}, assumption, ", self.clause_number);
        print_term(&mut self.variable_map, symbol_table, term_graph, left);
        print!(" = ");
        print_term(&mut self.variable_map, symbol_table, term_graph, right);
        println!(").");
        self.assumptions_list.push(self.clause_number);
        self.clause_number += 1;

        print!("cnf({}, plain, ", self.clause_number);
        print_literals(
            &mut self.variable_map,
            symbol_table,
            term_graph,
            clause_storage,
            literals,
        );
        if !literals.is_empty() {
            self.clause_stack.push(self.clause_number);
        }
        println!(", inference(extension, [], [{}])).", parent);
        self.clause_number += 1;

        print!("cnf({}, plain, ", self.clause_number);
        print_literals(
            &mut self.variable_map,
            symbol_table,
            term_graph,
            clause_storage,
            extension_literals,
        );
        println!(", inference(extension, [], [{}])).", parent);
        self.clause_stack.push(self.clause_number);
        self.clause_number += 1;
    }

    fn lemma(
        &mut self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        literals: Range<Literal>,
        left: Id<Term>,
        right: Id<Term>,
    ) {
        let parent = self
            .clause_stack
            .pop()
            .expect("printing lemma on empty clause stack");
        print!("cnf({}, assumption, ", self.clause_number);
        print_term(&mut self.variable_map, symbol_table, term_graph, left);
        print!(" = ");
        print_term(&mut self.variable_map, symbol_table, term_graph, right);
        println!(").");
        self.assumptions_list.push(self.clause_number);
        self.clause_number += 1;
        print!("cnf({}, plain, ", self.clause_number);
        print_literals(
            &mut self.variable_map,
            symbol_table,
            term_graph,
            clause_storage,
            literals,
        );
        if !literals.is_empty() {
            self.clause_stack.push(self.clause_number);
        }
        println!(
            ", inference(lemma, [assumptions([{}])], [{}])).",
            self.clause_number - 1,
            parent
        );
        self.clause_number += 1;
    }

    fn lazy_extension(
        &mut self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        literals: Range<Literal>,
        extension_literals: Range<Literal>,
    ) {
        let parent = self
            .clause_stack
            .pop()
            .expect("printing lazy extension on empty stack");
        print!("cnf({}, plain, ", self.clause_number);
        print_literals(
            &mut self.variable_map,
            symbol_table,
            term_graph,
            clause_storage,
            literals,
        );
        if !literals.is_empty() {
            self.clause_stack.push(self.clause_number);
        }
        println!(", inference(lazy_extension, [], [{}])).", parent);
        self.clause_number += 1;

        print!("cnf({}, plain, ", self.clause_number);
        print_literals(
            &mut self.variable_map,
            symbol_table,
            term_graph,
            clause_storage,
            extension_literals,
        );
        println!(", inference(lazy_extension, [], [{}])).", parent);
        self.clause_stack.push(self.clause_number);
        self.clause_number += 1;
    }

    fn reflexivity(
        &mut self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        literals: Range<Literal>,
        left: Id<Term>,
        right: Id<Term>,
    ) {
        let parent = self
            .clause_stack
            .pop()
            .expect("printing equality reduction on empty stack");
        print!("cnf({}, assumption, ", self.clause_number);
        print_term(&mut self.variable_map, symbol_table, term_graph, left);
        print!(" = ");
        print_term(&mut self.variable_map, symbol_table, term_graph, right);
        println!(").");
        self.assumptions_list.push(self.clause_number);
        self.clause_number += 1;

        print!("cnf({}, plain, ", self.clause_number);
        print_literals(
            &mut self.variable_map,
            symbol_table,
            term_graph,
            clause_storage,
            literals,
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

    fn unification(
        &mut self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
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
                    symbol_table,
                    term_graph,
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
}
