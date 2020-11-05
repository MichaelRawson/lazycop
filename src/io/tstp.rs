use crate::clausify;
use crate::equation_solver::EquationSolver;
use crate::occurs::SkipCheck;
use crate::prelude::*;
use crate::record::{Inference, Record};
use crate::util::fresh::Fresh;
use std::cell::RefCell;
use std::fmt::Display;

pub(crate) struct TSTPInference {
    name: &'static str,
    axiom: Option<(Id<ProblemClause>, Range<Literal>)>,
    lemmas: Vec<Id<Literal>>,
    equations: Vec<(Id<Term>, Id<Term>)>,
    deductions: Vec<Range<Literal>>,
}

impl Inference for TSTPInference {
    fn new(name: &'static str) -> Self {
        let axiom = None;
        let lemmas = vec![];
        let equations = vec![];
        let deductions = vec![];
        Self {
            name,
            axiom,
            lemmas,
            equations,
            deductions,
        }
    }

    fn axiom(
        &mut self,
        id: Id<ProblemClause>,
        literals: Range<Literal>,
    ) -> &mut Self {
        self.axiom = Some((id, literals));
        self
    }

    fn lemma(&mut self, lemma: Id<Literal>) -> &mut Self {
        self.lemmas.push(lemma);
        self
    }

    fn equation(&mut self, left: Id<Term>, right: Id<Term>) -> &mut Self {
        self.equations.push((left, right));
        self
    }

    fn deduction(&mut self, deduction: Range<Literal>) -> &mut Self {
        self.deductions.push(deduction);
        self
    }
}

#[derive(Default)]
pub(crate) struct TSTP {
    bindings: Bindings,
    solver: EquationSolver,
    variable_map: RefCell<Fresh>,
    clause_stack: Vec<usize>,
    premise_list: Vec<usize>,
    clause_number: usize,
}

impl TSTP {
    fn print_symbol(symbols: &Symbols, symbol: Id<Symbol>) {
        match &symbols[symbol].name {
            Name::Regular(word) => print!("{}", word),
            Name::Quoted(quoted) => print!("'{}'", quoted),
            Name::Skolem(skolem) => print!("sK{}", skolem),
            Name::Definition(definition) => print!("sP{}", definition),
        }
    }

    fn print_clausifier_term(symbols: &Symbols, term: &clausify::Term) {
        match term {
            clausify::Term::Var(clausify::Variable(n)) => print!("X{}", n),
            clausify::Term::Fun(f, args) => {
                Self::print_symbol(symbols, *f);
                let mut args = args.iter();
                if let Some(first) = args.next() {
                    print!("(");
                    Self::print_clausifier_term(symbols, first);
                    for rest in args {
                        print!(",");
                        Self::print_clausifier_term(symbols, rest);
                    }
                    print!(")");
                }
            }
        }
    }

    fn print_clausifier_literal(symbols: &Symbols, literal: &clausify::Literal) {
        let clausify::Literal(polarity, atom) = literal;
        match atom {
            clausify::Atom::Pred(term) => {
                if !*polarity {
                    print!("~");
                }
                Self::print_clausifier_term(symbols, term);
            }
            clausify::Atom::Eq(left, right) => {
                Self::print_clausifier_term(symbols, left);
                if *polarity {
                    print!(" = ");
                } else {
                    print!(" != ");
                }
                Self::print_clausifier_term(symbols, right);
            }
        }
    }

    pub(crate) fn print_clausifier_clause(
        symbols: &Symbols,
        number: usize,
        conjecture: bool,
        cnf: &clausify::CNF,
    ) {
        let role = if conjecture {
            "negated_conjecture"
        } else {
            "axiom"
        };
        print!("cnf(c{}, {}, ", number, role);
        let mut literals = cnf.0.iter();
        if let Some(first) = literals.next() {
            Self::print_clausifier_literal(symbols, first);
        } else {
            print!("$false");
        }
        for rest in literals {
            print!(" | ");
            Self::print_clausifier_literal(symbols, rest);
        }
        println!(").");
    }

    fn print_variable(&self, x: Id<Variable>) {
        let mapped = self.variable_map.borrow_mut().get(x);
        print!("X{}", mapped);
    }

    fn print_term(&self, symbols: &Symbols, terms: &Terms, term: Id<Term>) {
        match terms.view(symbols, term) {
            TermView::Variable(x) => self.print_variable(x),
            TermView::Function(symbol, args) => {
                Self::print_symbol(symbols, symbol);
                let mut args = args.into_iter();
                if let Some(first) = args.next() {
                    print!("(");
                    let first = terms.resolve(first);
                    self.print_term(symbols, terms, first);
                    for arg in args {
                        print!(",");
                        let term = terms.resolve(arg);
                        self.print_term(symbols, terms, term);
                    }
                    print!(")");
                }
            }
        }
    }

    pub(crate) fn print_literal(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        literal: &Literal,
    ) {
        if literal.is_predicate() {
            if !literal.polarity {
                print!("~");
            }
            let p = literal.get_predicate();
            self.print_term(symbols, terms, p);
        } else {
            let (left, right) = literal.get_equality();
            self.print_term(symbols, terms, left);
            print!(" ");
            if !literal.polarity {
                print!("!");
            }
            print!("= ");
            self.print_term(symbols, terms, right);
        }
    }

    fn print_clause<I: Iterator<Item = Id<Literal>>>(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        literals: &Literals,
        mut clause: I,
    ) {
        if let Some(id) = clause.next() {
            let literal = &literals[id];
            self.print_literal(symbols, terms, literal);
        } else {
            print!("$false");
            return;
        }
        for id in clause {
            let literal = &literals[id];
            print!(" | ");
            self.print_literal(symbols, terms, literal);
        }
    }
}

impl Record for TSTP {
    type Inference = TSTPInference;

    fn inference(
        &mut self,
        problem: &Problem,
        terms: &Terms,
        literals: &Literals,
        inference: &TSTPInference,
    ) {
        self.variable_map
            .borrow_mut()
            .resize(terms.len().transmute());
        self.bindings.resize(terms.len());
        self.bindings.save();
        let symbols = &problem.symbols;
        if let Some((id, axiom)) = inference.axiom {
            let origin = &problem.clauses[id].origin;
            let role = if origin.conjecture {
                "negated_conjecture"
            } else {
                "axiom"
            };
            print!("cnf(c{}, {},\n\t", self.clause_number, role);
            self.premise_list.push(self.clause_number);
            if self.clause_stack.is_empty() {
                self.clause_stack.push(self.clause_number);
                self.clause_stack.push(self.clause_number);
            }
            self.clause_number += 1;
            self.print_clause(symbols, terms, literals, axiom.into_iter());
            println!(
                "\n\tinference(clausify, [status(esa)], [file('{}', {})])).",
                origin.path.display(),
                origin.name,
            );
        }

        for lemma in &inference.lemmas {
            print!("cnf(c{}, lemma,\n\t", self.clause_number);
            self.premise_list.push(self.clause_number);
            self.clause_number += 1;
            self.print_literal(symbols, terms, &literals[*lemma]);
            println!(").");
        }

        self.solver.solve::<SkipCheck, _>(
            symbols,
            terms,
            &mut self.bindings,
            inference.equations.iter().copied(),
        );

        let parent = self.clause_stack.pop();
        let mut remaining_deductions = inference.deductions.len();
        for deduction in &inference.deductions {
            remaining_deductions -= 1;
            if Range::is_empty(*deduction) {
                if remaining_deductions > 0 {
                    continue;
                }
            } else {
                self.clause_stack.push(self.clause_number);
            }

            print!("cnf(c{}, plain,\n\t", self.clause_number);
            self.clause_number += 1;
            self.print_clause(symbols, terms, literals, deduction.into_iter());
            print!(",\n\tinference({}, [status(thm)", inference.name);

            for (variable, binding) in self.bindings.new_bindings() {
                print!(", bind(");
                self.print_variable(variable);
                print!(",");
                self.print_term(symbols, terms, binding);
                print!(")");
            }
            print!("], [");

            let mut first_premise = true;
            let premises = parent.iter().chain(self.premise_list.iter());
            for premise in premises {
                if !first_premise {
                    print!(", ");
                }
                first_premise = false;
                print!("c{}", premise);
            }
            println!("])).")
        }
        self.premise_list.clear();
        println!();
    }

    fn statistic<T: Display>(&mut self, key: &'static str, value: T) {
        println!("% {}: {}", key, value);
    }
}
