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
mod prelude;
mod problem;
//mod problem_builder;
mod record;
mod rule;
mod search;
mod statistics;
mod symbol;
mod tableau;
mod term;
mod uctree;
mod util;

use crate::goal::Goal;
use crate::io::tstp::TSTP;
use crate::io::{exit, szs, tptp};

fn main() {
    let mut symbols = symbol::Symbols::default();
    let options = options::Options::parse();
    let mut loader = tptp::Loader::new(&options.path);
    let mut clausifier = cnf::Clausifier::default();

    while let Some((_origin, formula)) = loader.next(&mut symbols) {
        clausifier.clausify(&mut symbols, formula);
    }

    let problem = todo!();
    let (statistics, result) = search::search(&problem, &options);

    if let options::Output::TSTP = options.output {
        let mut record = TSTP::default();
        if let Some(proof) = result {
            szs::unsatisfiable();
            szs::begin_incomplete_proof();
            let mut goal = Goal::new(&problem);
            for rule in proof {
                goal.apply_rule(&mut record, &rule);
            }
            let ok = goal.is_closed() && goal.solve_constraints();
            debug_assert!(ok);
            goal.record_unification(&mut record);
            szs::end_incomplete_proof();
            statistics.record(&mut record);
            exit::success()
        } else {
            szs::unknown();
            statistics.record(&mut record);
            exit::failure()
        }
    }
}
