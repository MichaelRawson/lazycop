use crate::clause::Clause;
use crate::prelude::*;
use crate::record::Record;
use crate::util::fresh::Fresh;

fn print_symbol(symbols: &Block<Symbol>, symbol: Id<Symbol>) {
    print!("{}", &symbols[symbol].name);
}

fn print_term(
    variable_map: &mut Fresh,
    symbols: &Block<Symbol>,
    terms: &Terms,
    term: Id<Term>,
) {
    match terms.view(term) {
        (_, TermView::Variable(x)) => {
            print!("X{}", variable_map.get(x));
        }
        (_, TermView::Function(symbol, mut args)) => {
            print_symbol(symbols, symbol);
            if let Some(first) = args.next() {
                print!("(");
                print_term(variable_map, symbols, terms, first);
                for arg in args {
                    print!(",");
                    print_term(variable_map, symbols, terms, arg);
                }
                print!(")");
            }
        }
    }
}

fn print_literal(
    variable_map: &mut Fresh,
    symbols: &Block<Symbol>,
    terms: &Terms,
    literal: Literal,
) {
    if literal.is_predicate() {
        if !literal.polarity {
            print!("~");
        }
        let p = literal.get_predicate();
        print_term(variable_map, symbols, terms, p);
    } else {
        let (left, right) = literal.get_equality();
        print_term(variable_map, symbols, terms, left);
        print!(" ");
        if !literal.polarity {
            print!("!");
        }
        print!("= ");
        print_term(variable_map, symbols, terms, right);
    }
}

fn print_literals(
    variable_map: &mut Fresh,
    symbols: &Block<Symbol>,
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
pub(crate) struct TSTP {
    variable_map: Fresh,
    clause_stack: Vec<usize>,
    assumption_number: usize,
    clause_number: usize,
}

impl TSTP {
    fn assumption(&mut self) {
        self.assumption_number += 1;
    }

    fn push(&mut self, clause: &Clause) {
        if !clause.is_empty() {
            self.clause_stack.push(self.clause_number);
            self.clause_number += 1;
        }
    }

    fn pop(&mut self) -> usize {
        self.clause_stack.pop().expect("empty clause stack")
    }
}

impl Record for TSTP {
    fn copy(
        &mut self,
        symbols: &Block<Symbol>,
        terms: &Terms,
        literals: &Block<Literal>,
        clause: &Clause,
    ) {
        print!("cnf(c{}, axiom, ", self.clause_number);
        print_literals(
            &mut self.variable_map,
            symbols,
            terms,
            literals,
            clause.open(),
        );
        println!(").");
        self.push(clause);
    }

    fn predicate_reduction(
        &mut self,
        symbols: &Block<Symbol>,
        terms: &Terms,
        literals: &Block<Literal>,
        clause: &Clause,
        left: Id<Term>,
        right: Id<Term>,
    ) {
        let parent = self.pop();
        print!("cnf(a{}, assumption, ", self.assumption_number);
        print_term(&mut self.variable_map, symbols, terms, left);
        print!(" = ");
        print_term(&mut self.variable_map, symbols, terms, right);
        println!(").");
        self.assumption();
        print!("cnf(c{}, plain, ", self.clause_number);
        print_literals(
            &mut self.variable_map,
            symbols,
            terms,
            literals,
            clause.open(),
        );
        self.push(clause);
        println!(
            ", inference(predicate_reduction, [assumptions([a{}])], [c{}])).",
            self.assumption_number - 1,
            parent
        );
    }

    fn predicate_extension(
        &mut self,
        symbols: &Block<Symbol>,
        terms: &Terms,
        literals: &Block<Literal>,
        clause: &Clause,
        new_clause: &Clause,
    ) {
        let copy = self.pop();
        let parent = self.pop();
        print!("cnf(c{}, plain, ", self.clause_number);
        print_literals(
            &mut self.variable_map,
            symbols,
            terms,
            literals,
            clause.remaining(),
        );
        if !Range::is_empty(clause.remaining()) {
            self.push(clause);
        } else {
            self.clause_number += 1;
        }
        println!(
            ", inference(predicate_extension, [], [c{}, c{}])).",
            parent, copy
        );

        print!("cnf(c{}, plain, ", self.clause_number);
        print_literals(
            &mut self.variable_map,
            symbols,
            terms,
            literals,
            new_clause.open(),
        );
        println!(
            ", inference(predicate_extension, [], [c{}, c{}])).",
            parent, copy
        );
        self.push(new_clause);
    }

    fn reflexivity(
        &mut self,
        symbols: &Block<Symbol>,
        terms: &Terms,
        literals: &Block<Literal>,
        clause: &Clause,
        left: Id<Term>,
        right: Id<Term>,
    ) {
        let parent = self.pop();
        print!("cnf(a{}, assumption, ", self.assumption_number);
        print_term(&mut self.variable_map, symbols, terms, left);
        print!(" = ");
        print_term(&mut self.variable_map, symbols, terms, right);
        println!(").");
        self.assumption();

        print!("cnf(c{}, plain, ", self.clause_number);
        print_literals(
            &mut self.variable_map,
            symbols,
            terms,
            literals,
            clause.open(),
        );
        self.push(clause);
        println!(
            ", inference(reflexivity, [assumptions([a{}])], [c{}])).",
            self.assumption_number - 1,
            parent
        );
    }
}
