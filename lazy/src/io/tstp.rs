use crate::clause::Clause;
use crate::prelude::*;
use crate::record::Record;
use crate::util::fresh::Fresh;
use std::fmt::Display;

#[derive(Default)]
pub struct TSTP {
    variable_map: Fresh,
    clause_stack: Vec<usize>,
    premise_list: Vec<usize>,
    assumption_number: usize,
    clause_number: usize,
}

impl TSTP {
    fn print_symbol(symbols: &Symbols, symbol: Id<Symbol>) {
        print!("{}", &symbols[symbol].name);
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
            TermView::Function(symbol, mut args) => {
                Self::print_symbol(symbols, symbol);
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

    fn print_clause(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        literals: &Literals,
        clause: Clause,
    ) {
        let mut range = clause.open();
        if let Some(id) = range.next() {
            let literal = &literals[id];
            self.print_literal(symbols, terms, literal);
        } else {
            print!("$false");
            return;
        }
        for id in range {
            let literal = &literals[id];
            print!(" | ");
            self.print_literal(symbols, terms, literal);
        }
    }
}

impl Record for TSTP {
    fn axiom(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        literals: &Literals,
        axiom: Clause,
    ) {
        print!("cnf(c{}, axiom,\n\t", self.clause_number);
        self.premise_list.push(self.clause_number);
        self.clause_number += 1;
        self.print_clause(symbols, terms, literals, axiom);
        println!(").");
    }

    fn inference<I: IntoIterator<Item = (Id<Term>, Id<Term>)>>(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        literals: &Literals,
        inference: &'static str,
        equations: I,
        literal: Option<&Literal>,
        deductions: &[Clause],
    ) {
        if let Some(literal) = literal {
            print!("cnf(c{}, plain,\n\t", self.clause_number);
            self.premise_list.push(self.clause_number);
            self.clause_number += 1;
            self.print_literal(symbols, terms, literal);
            println!(").");
        }

        let parent = self.clause_stack.pop();
        let assumption_start = self.assumption_number;
        for (left, right) in equations {
            print!("cnf(a{}, assumption,\n\t", self.assumption_number);
            self.print_term(symbols, terms, left);
            print!(" = ");
            self.print_term(symbols, terms, right);
            println!(").");
            self.assumption_number += 1;
        }

        for deduction in deductions {
            print!("cnf(c{}, plain,\n\t", self.clause_number);
            if !deduction.is_empty() {
                self.clause_stack.push(self.clause_number);
            }
            self.clause_number += 1;
            self.print_clause(symbols, terms, literals, *deduction);

            print!(",\n\tinference({}, [", inference);
            if assumption_start != self.assumption_number {
                print!("assumptions([");
                let mut assumptions = assumption_start..self.assumption_number;
                let first = some(assumptions.next());
                print!("a{}", first);
                for rest in assumptions {
                    print!(", a{}", rest);
                }
                print!("])");
            }
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

pub fn trivial_proof() {
    println!("cnf(c0, axiom,\n\t$false).");
}
