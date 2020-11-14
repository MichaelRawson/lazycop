use crate::clausify;
use crate::prelude::*;
use crate::statistics::Statistics;

#[derive(Default)]
pub(crate) struct TSTP {
    clause_number: usize,
}

impl TSTP {
    fn print_cnf_prologue(&mut self, origin: &Origin) {
        let role = if origin.conjecture {
            "negated_conjecture"
        } else {
            "axiom"
        };
        print!("cnf({}, {}", self.clause_number, role);
        self.clause_number += 1;
    }

    fn print_cnf_epilogue(origin: &Origin) {
        println!("file('{}', {})).", origin.path.display(), origin.name);
    }

    fn print_symbol(symbols: &Symbols, symbol: Id<Symbol>) {
        match &symbols[symbol].name {
            Name::Regular(word) => print!("{}", word),
            Name::Quoted(quoted) => print!("'{}'", quoted),
            Name::Distinct(distinct) => print!("\"{}\"", distinct),
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

    fn print_clausifier_literal(
        symbols: &Symbols,
        literal: &clausify::Literal,
    ) {
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
        &mut self,
        symbols: &Symbols,
        origin: &Origin,
        cnf: &clausify::CNF,
    ) {
        self.print_cnf_prologue(origin);
        print!(",\n\t");
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
        print!(",\n\t");
        Self::print_cnf_epilogue(origin);
    }

    fn print_term(
        &self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
        term: Id<Term>,
    ) {
        let term = bindings.resolve(terms, term);
        match terms.view(symbols, term) {
            TermView::Variable(_) => print!("sG0"),
            TermView::Function(symbol, args) => {
                Self::print_symbol(symbols, symbol);
                let mut args = args.into_iter();
                if let Some(first) = args.next() {
                    print!("(");
                    let first = terms.resolve(first);
                    self.print_term(symbols, terms, bindings, first);
                    for arg in args {
                        print!(",");
                        let term = terms.resolve(arg);
                        self.print_term(symbols, terms, bindings, term);
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
        bindings: &Bindings,
        literal: &Literal,
    ) {
        if literal.is_predicate() {
            if !literal.polarity {
                print!("~");
            }
            let p = literal.get_predicate();
            self.print_term(symbols, terms, bindings, p);
        } else {
            let (left, right) = literal.get_equality();
            self.print_term(symbols, terms, bindings, left);
            print!(" ");
            if !literal.polarity {
                print!("!");
            }
            print!("= ");
            self.print_term(symbols, terms, bindings, right);
        }
    }

    fn print_clause<I: IntoIterator<Item = Id<Literal>>>(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        literals: &Literals,
        bindings: &Bindings,
        clause: I,
    ) {
        let mut clause = clause.into_iter();
        if let Some(id) = clause.next() {
            let literal = &literals[id];
            self.print_literal(symbols, terms, bindings, literal);
        } else {
            print!("$false");
            return;
        }
        for id in clause {
            let literal = &literals[id];
            print!(" | ");
            self.print_literal(symbols, terms, bindings, literal);
        }
    }

    pub(crate) fn print_proof_clause(
        &mut self,
        problem: &Problem,
        terms: &Terms,
        literals: &Literals,
        bindings: &Bindings,
        clause: Clause,
    ) {
        let origin = &problem.clauses[clause.axiom].origin;
        self.print_cnf_prologue(origin);
        print!(",\n\t");
        self.print_clause(
            &problem.symbols,
            terms,
            literals,
            bindings,
            clause.original(),
        );
        print!(",\n\t");
        Self::print_cnf_epilogue(origin);
    }

    pub(crate) fn print_statistics(&mut self, statistics: &Statistics) {
        println!("% problem symbols\t: {}", statistics.problem_symbols);
        println!("% problem clauses\t: {}", statistics.problem_clauses);
        println!(
            "% eliminated leaves\t: {}",
            statistics.load_eliminated_leaves()
        );
        println!("% retained leaves\t: {}", statistics.load_retained_leaves());
        println!("% expanded leaves\t: {}", statistics.load_expanded_leaves());
        #[cfg(feature = "cudann")]
        println!("% expanded leaves\t: {}", statistics.load_evaluated_leaves());
    }
}
