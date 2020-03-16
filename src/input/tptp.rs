use crate::output::exit;
use crate::output::szs;
use crate::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::mem;
use tptp::parsers;
use tptp::syntax;
use tptp::syntax::Visitor;

type FunctionKey = (Id<Symbol>, Vec<Id<Term>>);

#[derive(Default)]
struct Builder {
    symbol_list: SymbolList,
    symbols: HashMap<(String, u32), Id<Symbol>>,
    term_list: TermList,
    saved_terms: Vec<Id<Term>>,
    clause_variables: HashMap<String, Id<Term>>,
    clause_functions: HashMap<FunctionKey, Id<Term>>,
    clause_literals: Vec<Literal>,
    clauses: Vec<(Clause, TermList)>,
}

impl Builder {
    fn finish(self) -> Problem {
        Problem::new(self.symbol_list, self.clauses)
    }
}

impl Visitor for Builder {
    fn visit_variable(&mut self, variable: syntax::Variable) {
        let terms = &mut self.term_list;
        let saved_terms = &mut self.saved_terms;
        let clause_variables = &mut self.clause_variables;
        let id = *clause_variables
            .entry(format!("{}", variable))
            .or_insert_with(|| terms.add_variable());
        saved_terms.push(id);
    }

    fn visit_fof_plain_term(&mut self, term: syntax::FofPlainTerm) {
        let (functor, arguments) = match term {
            syntax::FofPlainTerm::Constant(c) => (c, vec![]),
            syntax::FofPlainTerm::Function(f, args) => (f, args.0),
        };
        let arity = arguments.len();
        for argument in arguments {
            self.visit_fof_term(argument);
        }

        let symbols = &mut self.symbols;
        let symbol_list = &mut self.symbol_list;
        let symbol_id = *symbols
            .entry((format!("{}", &functor), arity as u32))
            .or_insert_with(|| {
                symbol_list.add(format!("{}", &functor), arity as u32)
            });

        let term_list = &mut self.term_list;
        let saved_terms = &mut self.saved_terms;
        let clause_functions = &mut self.clause_functions;
        let args = saved_terms.split_off(saved_terms.len() - arity);
        let id = *clause_functions
            .entry((symbol_id, args.clone()))
            .or_insert_with(|| {
                let function_id = term_list.add_symbol(symbol_id);
                for arg in args {
                    term_list.add_reference(arg);
                }
                function_id
            });
        saved_terms.push(id);
    }

    fn visit_literal(&mut self, literal: syntax::Literal) {
        let (atom, polarity) = match literal {
            syntax::Literal::Atomic(syntax::FofAtomicFormula::Plain(p)) => {
                self.visit_fof_plain_atomic_formula(p);
                let term = self.saved_terms.pop().unwrap();
                (true, Atom::Predicate(term))
            }
            syntax::Literal::Atomic(syntax::FofAtomicFormula::Defined(
                syntax::FofDefinedAtomicFormula::Infix(infix),
            )) => {
                self.visit_fof_term(infix.left);
                let left = self.saved_terms.pop().unwrap();
                self.visit_fof_term(infix.right);
                let right = self.saved_terms.pop().unwrap();
                (true, Atom::Equality(left, right))
            }
            syntax::Literal::NegatedAtomic(
                syntax::FofAtomicFormula::Plain(p),
            ) => {
                self.visit_fof_plain_atomic_formula(p);
                let term = self.saved_terms.pop().unwrap();
                (false, Atom::Predicate(term))
            }
            syntax::Literal::NegatedAtomic(
                syntax::FofAtomicFormula::Defined(
                    syntax::FofDefinedAtomicFormula::Infix(infix),
                ),
            ) => {
                self.visit_fof_term(infix.left);
                let left = self.saved_terms.pop().unwrap();
                self.visit_fof_term(infix.right);
                let right = self.saved_terms.pop().unwrap();
                (false, Atom::Equality(left, right))
            }
            syntax::Literal::Infix(infix) => {
                self.visit_fof_term(infix.left);
                let left = self.saved_terms.pop().unwrap();
                self.visit_fof_term(infix.right);
                let right = self.saved_terms.pop().unwrap();
                (false, Atom::Equality(left, right))
            }
            _ => unimplemented!(),
        };
        self.clause_literals.push(Literal::new(atom, polarity));
    }

    fn visit_disjunction(&mut self, disjunction: syntax::Disjunction) {
        assert!(self.term_list.is_empty());
        assert!(self.saved_terms.is_empty());
        assert!(self.clause_variables.is_empty());
        assert!(self.clause_functions.is_empty());
        assert!(self.clause_literals.is_empty());

        for literal in disjunction.0 {
            self.visit_literal(literal);
        }

        assert!(self.saved_terms.is_empty());
        let term_list = mem::take(&mut self.term_list);
        self.clause_variables.clear();
        self.clause_functions.clear();
        let literals = mem::take(&mut self.clause_literals);
        self.clauses.push((Clause::new(literals), term_list));
    }
}

fn report_inappropriate<T: fmt::Display>(t: T) -> ! {
    error!("non-CNF input:\n{}", t);
    szs::inappropriate();
    exit::failure()
}

pub fn parse(bytes: &[u8]) -> Problem {
    let mut builder = Builder::default();
    let mut inputs = parsers::tptp_input_iterator::<()>(bytes);

    for input in &mut inputs {
        if let syntax::TPTPInput::Annotated(formula) = input {
            if let syntax::AnnotatedFormula::Cnf(cnf) = *formula {
                builder.visit_cnf_annotated(cnf);
            } else {
                report_inappropriate(formula)
            }
        } else {
            report_inappropriate(input)
        }
    }

    if let Ok((bytes, _)) = inputs.finish() {
        if let Ok((b"", _)) = parsers::ignored::<()>(bytes) {
            return builder.finish();
        }
    }
    error!("unsupported syntax in input");
    szs::input_error();
    exit::failure()
}
