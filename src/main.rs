mod atom;
mod binding;
mod clause;
mod constraint;
mod disequation_solver;
mod equation_solver;
mod goal;
mod index;
mod infer;
mod io;
mod literal;
mod occurs;
mod ordering_solver;
mod prelude;
mod problem;
mod problem_builder;
mod record;
mod rule;
mod rule_store;
mod search;
mod statistics;
mod symbol;
mod tableau;
mod term;
mod util;

use crate::goal::Goal;
use crate::io::tstp;
use crate::io::{exit, szs, tptp};
use tstp::TSTP;

fn main() {
    let problem = tptp::load_from_stdin();
    let (statistics, result) = search::search(&problem);

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
