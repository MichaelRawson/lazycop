mod atom;
mod binding;
mod clause;
mod cnf;
mod constraint;
mod disequation_solver;
mod equation_solver;
mod goal;
mod index;
mod infer;
mod io;
mod literal;
mod occurs;
mod options;
mod ordering_solver;
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
mod uctree;
mod util;

use crate::cnf::Clausifier;
use crate::goal::Goal;
use crate::io::tstp::TSTP;
use crate::io::{exit, szs, tptp};
use crate::options::Options;
use crate::problem_builder::ProblemBuilder;
use crate::symbol::Symbols;

fn main() {
    let options = Options::parse();
    let name = options.problem_name();
    let mut symbols = Symbols::default();
    let mut loader = tptp::Loader::new(&options.path);
    let mut clausifier = Clausifier::default();
    let mut builder = ProblemBuilder::default();
    let mut clause_number = 0;

    while let Some((origin, formula)) = loader.next(&mut symbols) {
        clausifier.formula(&mut symbols, formula);
        while let Some(cnf) = clausifier.next(&mut symbols) {
            if options.clausify {
                TSTP::print_cnf(
                    &symbols,
                    clause_number,
                    origin.conjecture,
                    &cnf,
                );
                clause_number += 1;
            } else {
                builder.add_axiom(&symbols, origin.clone(), cnf);
            }
        }
    }
    if options.clausify {
        return;
    }

    let problem = builder.finish(symbols);
    let (statistics, result) = search::search(&problem, &options);

    let mut record = TSTP::default();
    if let Some(proof) = result {
        szs::unsatisfiable(&name);
        szs::begin_cnf_refutation(&name);
        let mut goal = Goal::new(&problem);
        for rule in proof {
            goal.apply_rule(&mut record, &rule);
        }
        let ok = goal.is_closed() && goal.solve_constraints();
        debug_assert!(ok);
        goal.record_unification(&mut record);
        szs::end_cnf_refutation(&name);
        statistics.record(&mut record);
        exit::success()
    } else {
        szs::gave_up(&name);
        statistics.record(&mut record);
        exit::failure()
    }
}
