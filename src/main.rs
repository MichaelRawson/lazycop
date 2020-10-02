mod atom;
mod binding;
mod clause;
mod cnf;
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
mod record;
mod rule;
mod search;
mod statistics;
mod symbol;
mod tableau;
mod term;
mod tree;
mod util;

use crate::cnf::Clausifier;
use crate::io::tptp;
use crate::io::tstp::TSTP;
use crate::options::Options;
use crate::output::OutputInfo;
use crate::problem_builder::ProblemBuilder;
use crate::symbol::Symbols;

fn main() {
    #[cfg(feature = "cudann")]
    cudann::init();

    let options = Options::parse();
    let name = options.problem_name();
    let mut symbols = Symbols::default();
    let mut loader = tptp::Loader::new(&options.path);
    let mut clausifier = Clausifier::default();

    if options.dump_clauses {
        let mut clause_number = 0;
        while let Some((_, origin, formula)) = loader.next(&mut symbols) {
            clausifier.formula(&mut symbols, formula);
            while let Some(cnf) = clausifier.next(&mut symbols) {
                TSTP::print_cnf(
                    &symbols,
                    clause_number,
                    origin.conjecture,
                    &cnf,
                );
                clause_number += 1;
            }
        }
        return;
    }

    let mut builder = ProblemBuilder::default();
    let mut info = OutputInfo::default();
    while let Some((is_cnf, origin, formula)) = loader.next(&mut symbols) {
        info.is_cnf |= is_cnf;
        info.has_axioms |= !origin.conjecture;
        info.has_conjecture |= origin.conjecture;
        clausifier.formula(&mut symbols, formula);
        while let Some(cnf) = clausifier.next(&mut symbols) {
            builder.add_axiom(&symbols, origin.clone(), cnf);
        }
    }

    let problem = builder.finish(symbols);
    std::mem::drop(loader);
    let (statistics, result) = search::search(&problem, &options);

    options
        .output
        .result(&name, &problem, info, result, &statistics)
}
