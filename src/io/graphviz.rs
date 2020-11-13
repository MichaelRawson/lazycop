use crate::prelude::*;
use crate::record::{Inference, Record};
use crate::util::fresh::Fresh;

pub(crate) struct GraphvizInference {
    axiom: Option<Range<Literal>>,
    lemma: Option<Id<Literal>>,
    deductions: Vec<Range<Literal>>,
}

impl Inference for GraphvizInference {
    fn new(_name: &str) -> Self {
        let axiom = None;
        let lemma = None;
        let deductions = vec![];
        Self {
            axiom,
            lemma,
            deductions,
        }
    }

    fn axiom(
        &mut self,
        _id: Id<ProblemClause>,
        literals: Range<Literal>,
    ) -> &mut Self {
        self.axiom = Some(literals);
        self
    }

    fn lemma(&mut self, lemma: Id<Literal>) -> &mut Self {
        self.lemma = Some(lemma);
        self
    }

    fn deduction(&mut self, deduction: Range<Literal>) -> &mut Self {
        self.deductions.push(deduction);
        self
    }
}

#[derive(Default)]
pub(crate) struct Graphviz {
    todo: Vec<Id<Literal>>,
    lemmas: LUT<Literal, Option<Id<Literal>>>,
    fresh: Fresh,
}

impl Graphviz {
    pub(crate) fn start(&self) {
        println!("digraph proof {{");
        println!("\tnode [fontname=monospace, shape=rectangle];");
        println!("\tstart [label=\"*\"];");
    }

    pub(crate) fn finish(self) {
        println!("}}");
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

    fn print_variable(&mut self, x: Id<Variable>) {
        print!("X{}", self.fresh.get(x));
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

    fn get_parent(&mut self) -> String {
        self.todo
            .last()
            .map(|literal| format!("{}", literal.index()))
            .unwrap_or_else(|| "start".to_string())
    }
}

impl Record for Graphviz {
    type Inference = GraphvizInference;

    fn inference(
        &mut self,
        problem: &Problem,
        terms: &Terms,
        literals: &Literals,
        inference: &Self::Inference,
    ) {
        self.lemmas.resize(literals.len());
        self.fresh.resize(terms.len().transmute());
        let is_start = self.todo.is_empty();
        let mut deductions = inference.deductions.iter();
        deductions.next();
        let mut parent = self.get_parent();
        self.todo.pop();

        if let Some(axiom) = inference.axiom {
            deductions.next();
            let mut first_literal = true;
            for literal in axiom {
                print!("\t{} [label=\"", literal.index());
                self.print_literal(
                    &problem.symbols,
                    terms,
                    &literals[literal],
                );
                println!("\"];");
                if first_literal {
                    println!(
                        "\t{} -> {} [style=bold];",
                        parent,
                        literal.index()
                    );
                    first_literal = false;
                } else {
                    println!("\t{} -> {};", parent, literal.index());
                }
            }
            self.todo.extend(axiom.into_iter().rev());
            parent = self.get_parent();
            if !is_start {
                self.todo.pop();
            }
        } else if let Some(lemma) = inference.lemma {
            if let Some(original) = self.lemmas[lemma] {
                println!(
                    "\t{} -> {} [constraint=false, style=dashed];",
                    parent,
                    original.index()
                );
            } else {
                println!("\t{} -> {} [style=bold];", parent, lemma.index());
            }
        } else {
            println!("\t{}:s -> {}:s [style=bold];", parent, parent);
        }

        if let Some(deduction) = deductions.next() {
            for literal in *deduction {
                print!("\t{} [label=\"", literal.index());
                self.print_literal(
                    &problem.symbols,
                    terms,
                    &literals[literal],
                );
                println!("\"];");
                println!("\t{} -> {};", parent, literal.index());
            }
            self.todo.extend(deduction.into_iter().rev());
        }
    }

    fn lemma(
        &mut self,
        literals: &Literals,
        original: Id<Literal>,
        lemma: Id<Literal>,
    ) {
        self.lemmas.resize(literals.len());
        self.lemmas[lemma] = Some(original);
    }
}
