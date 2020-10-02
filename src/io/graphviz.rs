use crate::io::tstp::TSTP;
use crate::prelude::*;
use crate::record::{Inference, Record};

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
    tstp: TSTP,
    todo: Vec<Id<Literal>>,
    lemmas: LUT<Literal, Option<Id<Literal>>>,
}

impl Graphviz {
    pub(crate) fn start(&self) {
        println!("digraph proof {{");
        println!("\tnode [fontname=monospace, shape=rectangle];");
        println!("\tstart;");
    }

    pub(crate) fn finish(self) {
        println!("}}");
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
                self.tstp.print_literal(
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
        }

        for deduction in deductions {
            let mut first_literal = true;
            for literal in *deduction {
                print!("\t{} [label=\"", literal.index());
                self.tstp.print_literal(
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
            self.todo.extend(deduction.into_iter().rev());
        }
        println!();
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
