use crate::io::exit;
use crate::io::record::Record;
use crate::io::szs;
use crate::prelude::*;
use crate::util::id_map::IdMap;
use crate::util::variable_map::VariableMap;
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
            ast::Literal::Atomic(atomic) => report_inappropriate(atomic),
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

struct FmtSymbol<'a>(pub(crate) &'a SymbolTable, pub(crate) Id<Symbol>);

impl fmt::Display for FmtSymbol<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let FmtSymbol(symbol_table, symbol_id) = self;
        write!(f, "{}", symbol_table.name(*symbol_id))
    }
}

struct FmtTerm<'a>(
    pub(crate) &'a VariableMap,
    pub(crate) &'a SymbolTable,
    pub(crate) &'a TermGraph,
    pub(crate) Id<Term>,
);

impl fmt::Display for FmtTerm<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let FmtTerm(variable_map, symbol_table, term_graph, term_id) = self;
        let mut arg_stack = vec![vec![*term_id]];

        while let Some(args) = arg_stack.last_mut() {
            if let Some(next_arg) = args.pop() {
                let mut needs_comma = !args.is_empty();
                match term_graph.view(next_arg) {
                    TermView::Variable(x) => {
                        write!(f, "X{}", variable_map.get(x))?;
                    }
                    TermView::Function(symbol, new_args) => {
                        write!(f, "{}", FmtSymbol(symbol_table, symbol))?;
                        let mut new_args: Vec<_> = new_args.collect();
                        if !new_args.is_empty() {
                            write!(f, "(")?;
                            needs_comma = false;
                            new_args.reverse();
                            arg_stack.push(new_args);
                        }
                    }
                }
                if needs_comma {
                    write!(f, ",")?;
                }
            } else {
                arg_stack.pop();
                if let Some(args) = arg_stack.last_mut() {
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

struct FmtLiteral<'a>(
    pub(crate) &'a VariableMap,
    pub(crate) &'a SymbolTable,
    pub(crate) &'a TermGraph,
    pub(crate) Literal,
);

impl fmt::Display for FmtLiteral<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let FmtLiteral(variable_map, symbol_table, term_graph, literal) = self;
        match (literal.polarity, literal.atom) {
            (true, Atom::Predicate(p)) => write!(
                f,
                "{}",
                FmtTerm(variable_map, symbol_table, term_graph, p)
            ),
            (false, Atom::Predicate(p)) => write!(
                f,
                "~{}",
                FmtTerm(variable_map, symbol_table, term_graph, p)
            ),
            (true, Atom::Equality(left, right)) => write!(
                f,
                "{} = {}",
                FmtTerm(variable_map, symbol_table, term_graph, left),
                FmtTerm(variable_map, symbol_table, term_graph, right)
            ),
            (false, Atom::Equality(left, right)) => write!(
                f,
                "{} != {}",
                FmtTerm(variable_map, symbol_table, term_graph, left),
                FmtTerm(variable_map, symbol_table, term_graph, right)
            ),
        }
    }
}

struct FmtClause<'a>(
    pub(crate) &'a VariableMap,
    pub(crate) &'a SymbolTable,
    pub(crate) &'a TermGraph,
    pub(crate) &'a ClauseStorage,
    pub(crate) Clause,
);

impl fmt::Display for FmtClause<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let FmtClause(
            variable_map,
            symbol_table,
            term_graph,
            clause_storage,
            clause,
        ) = self;
        let mut literals = clause.literals(clause_storage);
        if let Some(literal) = literals.next() {
            write!(
                f,
                "{}",
                FmtLiteral(variable_map, symbol_table, term_graph, literal)
            )?;
        } else {
            write!(f, "$false")?;
        }
        for literal in literals {
            write!(
                f,
                " | {}",
                FmtLiteral(variable_map, symbol_table, term_graph, literal)
            )?;
        }
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct TPTPProof {
    variable_map: VariableMap,
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
        clause: Clause,
    ) {
        println!(
            "cnf({}, negated_conjecture, {}).",
            self.clause_number,
            FmtClause(
                &self.variable_map,
                symbol_table,
                term_graph,
                clause_storage,
                clause
            )
        );
        self.clause_stack.push(self.clause_number);
        self.clause_number += 1;
    }

    fn reduction(
        &mut self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        clause: Clause,
        left: Id<Term>,
        right: Id<Term>,
    ) {
        let parent = self
            .clause_stack
            .pop()
            .expect("printing reduction on empty clause stack");
        println!(
            "cnf({}, assumption, {} = {}).",
            self.clause_number,
            FmtTerm(&self.variable_map, symbol_table, term_graph, left),
            FmtTerm(&self.variable_map, symbol_table, term_graph, right)
        );
        self.assumptions_list.push(self.clause_number);
        self.clause_number += 1;
        println!(
            "cnf({}, plain, {}, inference(reduction, [], [{}])).",
            self.clause_number,
            FmtClause(
                &self.variable_map,
                symbol_table,
                term_graph,
                clause_storage,
                clause
            ),
            parent
        );
        if !clause.is_empty() {
            self.clause_stack.push(self.clause_number);
        }
        self.clause_number += 1;
    }

    fn lemma(
        &mut self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        clause: Clause,
        left: Id<Term>,
        right: Id<Term>,
    ) {
        let parent = self
            .clause_stack
            .pop()
            .expect("printing lemma on empty clause stack");
        println!(
            "cnf({}, assumption, {} = {}).",
            self.clause_number,
            FmtTerm(&self.variable_map, symbol_table, term_graph, left),
            FmtTerm(&self.variable_map, symbol_table, term_graph, right)
        );
        self.assumptions_list.push(self.clause_number);
        self.clause_number += 1;
        println!(
            "cnf({}, plain, {}, inference(lemma, [assumptions([{}])], [{}])).",
            self.clause_number,
            FmtClause(
                &self.variable_map,
                symbol_table,
                term_graph,
                clause_storage,
                clause
            ),
            self.clause_number - 1,
            parent
        );
        if !clause.is_empty() {
            self.clause_stack.push(self.clause_number);
        }
        self.clause_number += 1;
    }

    fn equality_reduction(
        &mut self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        clause: Clause,
        left: Id<Term>,
        right: Id<Term>,
    ) {
        let parent = self
            .clause_stack
            .pop()
            .expect("printing equality reduction on empty stack");
        println!(
            "cnf({}, assumption, {} = {}).",
            self.clause_number,
            FmtTerm(&self.variable_map, symbol_table, term_graph, left),
            FmtTerm(&self.variable_map, symbol_table, term_graph, right)
        );
        self.assumptions_list.push(self.clause_number);
        self.clause_number += 1;
        println!(
            "cnf({}, plain, {}, inference(equality_reduction, [assumptions([{}])], [{}])).",
            self.clause_number,
            FmtClause(
                &self.variable_map,
                symbol_table,
                term_graph,
                clause_storage,
                clause
            ),
            self.clause_number - 1,
            parent
        );
        if !clause.is_empty() {
            self.clause_stack.push(self.clause_number);
        }
        self.clause_number += 1;
    }

    fn extension(
        &mut self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        clause: Clause,
        extension_clause: Clause,
    ) {
        let parent = self
            .clause_stack
            .pop()
            .expect("printing extension on empty stack");
        println!(
            "cnf({}, plain, {}, inference(extension, [], [{}])).",
            self.clause_number,
            FmtClause(
                &self.variable_map,
                symbol_table,
                term_graph,
                clause_storage,
                clause
            ),
            parent
        );
        if !clause.is_empty() {
            self.clause_stack.push(self.clause_number);
        }
        self.clause_number += 1;
        println!(
            "cnf({}, plain, {}, inference(extension, [], [{}])).",
            self.clause_number,
            FmtClause(
                &self.variable_map,
                symbol_table,
                term_graph,
                clause_storage,
                extension_clause
            ),
            parent
        );
        self.clause_stack.push(self.clause_number);
        self.clause_number += 1;
    }

    fn constraint_solving(
        &mut self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        bindings: &IdMap<Variable, Option<Id<Term>>>,
    ) {
        print!(
            "cnf({}, plain, $false, inference(constraint_solving, [",
            self.clause_number,
        );
        let mut after_first_bind = false;
        for id in bindings.into_iter() {
            if let Some(bound) = bindings[id] {
                if after_first_bind {
                    print!(", ");
                }
                print!(
                    "bind(X{}, {})",
                    self.variable_map.get(id),
                    FmtTerm(
                        &self.variable_map,
                        symbol_table,
                        term_graph,
                        bound
                    )
                );
                after_first_bind = true;
            }
        }
        println!("], {:?})).", self.assumptions_list);
        self.clause_stack.clear();
        self.assumptions_list.clear();
    }
}
