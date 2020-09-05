use crate::cnf;
use crate::prelude::*;
use crate::record::{Inference, Record};
use crate::util::fresh::Fresh;
use std::fmt::Display;

pub(crate) struct TSTPInference {
    name: &'static str,
    axioms: Vec<(Id<ProblemClause>, Range<Literal>)>,
    lemmas: Vec<Id<Literal>>,
    equations: Vec<(Id<Term>, Id<Term>)>,
    deductions: Vec<Range<Literal>>,
}

impl Inference for TSTPInference {
    fn new(name: &'static str) -> Self {
        let axioms = vec![];
        let lemmas = vec![];
        let equations = vec![];
        let deductions = vec![];
        Self {
            name,
            axioms,
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
        self.axioms.push((id, literals));
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
    variable_map: Fresh,
    clause_stack: Vec<usize>,
    premise_list: Vec<usize>,
    assumption_number: usize,
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

    fn print_cnf_term(symbols: &Symbols, term: &cnf::Term) {
        match term {
            cnf::Term::Var(cnf::Variable(n)) => print!("X{}", n),
            cnf::Term::Fun(f, args) => {
                Self::print_symbol(symbols, *f);
                let mut args = args.iter();
                if let Some(first) = args.next() {
                    print!("(");
                    Self::print_cnf_term(symbols, first);
                    for rest in args {
                        print!(",");
                        Self::print_cnf_term(symbols, rest);
                    }
                    print!(")");
                }
            }
        }
    }

    fn print_cnf_literal(symbols: &Symbols, literal: &cnf::Literal) {
        let cnf::Literal(polarity, atom) = literal;
        match atom {
            cnf::Atom::Pred(term) => {
                if !*polarity {
                    print!("~");
                }
                Self::print_cnf_term(symbols, term);
            }
            cnf::Atom::Eq(left, right) => {
                Self::print_cnf_term(symbols, left);
                if *polarity {
                    print!(" = ");
                } else {
                    print!(" != ");
                }
                Self::print_cnf_term(symbols, right);
            }
        }
    }

    pub(crate) fn print_cnf(
        symbols: &Symbols,
        number: usize,
        conjecture: bool,
        cnf: &cnf::CNF,
    ) {
        let role = if conjecture {
            "negated_conjecture"
        } else {
            "axiom"
        };
        print!("cnf(c{}, {}, ", number, role);
        let mut literals = cnf.0.iter();
        if let Some(first) = literals.next() {
            Self::print_cnf_literal(symbols, first);
        } else {
            print!("$false");
        }
        for rest in literals {
            print!(" | ");
            Self::print_cnf_literal(symbols, rest);
        }
        println!(").");
    }

    fn print_variable(&mut self, x: Id<Variable>) {
        print!("X{}", self.variable_map.get(x));
    }

    fn print_term(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        term: Id<Term>,
    ) {
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

    fn print_literal(
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
        let symbols = &problem.symbols;
        for (id, axiom) in &inference.axioms {
            let origin = &problem.clauses[*id].origin;
            let role = if origin.conjecture {
                "negated_conjecture"
            } else {
                "axiom"
            };
            print!("cnf(c{}, {},\n\t", self.clause_number, role);
            self.premise_list.push(self.clause_number);
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

        let parent = self.clause_stack.pop();
        let assumption_start = self.assumption_number;
        for (left, right) in &inference.equations {
            print!("cnf(a{}, assumption,\n\t", self.assumption_number);
            self.print_term(symbols, terms, *left);
            print!(" = ");
            self.print_term(symbols, terms, *right);
            println!(").");
            self.assumption_number += 1;
        }

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

            print!(",\n\tinference({}, [", inference.name);
            print!("assumptions([");
            let mut assumptions = assumption_start..self.assumption_number;
            if let Some(first) = assumptions.next() {
                print!("a{}", first);
            }
            for rest in assumptions {
                print!(", a{}", rest);
            }
            print!("]), status(thm)");
            print!("], [");

            let mut first_premise = true;
            if let Some(parent) = parent {
                print!("c{}", parent);
                first_premise = false;
            }

            for premise in self.premise_list.iter() {
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

    fn unification<I: Iterator<Item = (Id<Variable>, Id<Term>)>>(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: I,
    ) {
        if self.assumption_number == 0 {
            return;
        }

        print!("cnf(c{}, plain,\n\t$false", self.clause_number);
        print!(",\n\tinference(constraint_solving, [\n\t\t");
        let mut first_bind = true;
        for (x, term) in bindings {
            if !first_bind {
                print!(",\n\t\t");
            }
            print!("bind(");
            self.print_variable(x);
            print!(", ");
            self.print_term(symbols, terms, term);
            print!(")");
            first_bind = false;
        }

        print!("\n\t],\n\t[");
        print!("a0");
        for number in 1..self.assumption_number {
            print!(", a{}", number);
        }
        println!("])).");
        println!();
    }

    fn statistic<T: Display>(&mut self, key: &'static str, value: T) {
        println!("% {}: {}", key, value);
    }
}
