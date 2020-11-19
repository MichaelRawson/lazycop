#![allow(clippy::too_many_arguments)]

mod atom;
mod binding;
mod clause;
mod clausify;
mod constraint;
mod disequation_solver;
mod equation_solver;
mod goal;
mod graph;
mod index;
mod infer;
mod io;
mod literal;
mod occurs;
mod options;
mod ordering_solver;
mod output;
mod prelude;
mod problem;
mod problem_builder;
mod rule;
mod search;
#[cfg(feature = "smt")]
mod smt;
mod statistics;
mod symbol;
mod tableau;
mod term;
mod tree;
mod util;

use crate::clausify::Clausifier;
use crate::io::tptp;
use crate::io::tstp::TSTP;
use crate::options::{Dump, Options};
use crate::problem::Problem;
use crate::problem_builder::ProblemBuilder;
use crate::symbol::Symbols;

fn load(options: &Options) -> Option<Problem> {
    let mut symbols = Symbols::default();
    let mut loader = tptp::Loader::new(&options.path);
    let mut clausifier = Clausifier::default();
    let mut builder = ProblemBuilder::default();
    let mut tstp = TSTP::default();

    while let Some((origin, formula)) = loader.next(&mut symbols) {
        match options.dump {
            Some(Dump::CNF) => {
                clausifier.formula(&mut symbols, formula);
                while let Some(cnf) = clausifier.next(&mut symbols) {
                    tstp.print_clausifier_clause(&symbols, &origin, &cnf);
                }
            }
            None => {
                clausifier.formula(&mut symbols, formula);
                while let Some(cnf) = clausifier.next(&mut symbols) {
                    builder.add_axiom(&symbols, origin.clone(), cnf);
                }
            }
        }
    }

    if options.dump.is_none() {
        Some(builder.finish(symbols))
    } else {
        None
    }
}

fn main() {
    #[cfg(feature = "cudann")]
    cudann::init();

    let options = Options::parse();
    if let Some(problem) = load(&options) {
        let (statistics, result) = search::search(&problem, &options);
        options
            .output
            .result(&options, &problem, result, &statistics)
    }
}
