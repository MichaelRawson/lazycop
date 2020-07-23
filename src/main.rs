#![type_length_limit = "10000000"]

pub(crate) mod atom;
pub(crate) mod binding;
pub(crate) mod clause;
pub(crate) mod constraint;
pub(crate) mod disequation_solver;
pub(crate) mod equation_solver;
pub(crate) mod goal;
pub(crate) mod io;
pub(crate) mod literal;
pub(crate) mod occurs;
pub(crate) mod ordering_solver;
pub(crate) mod prelude;
pub(crate) mod problem;
pub(crate) mod record;
pub(crate) mod rule;
pub(crate) mod symbol;
pub(crate) mod tableau;
pub(crate) mod term;
pub(crate) mod util;

mod priority;
mod rule_store;
mod search;
mod statistics;

use crate::io::tstp;
use crate::io::tstp::TSTP;
use crate::io::{exit, szs, tptp};
use crate::tableau::Tableau;

fn main() {
    //heuristic::init();

    let problem = tptp::load_from_stdin();
    if problem.is_trivial() {
        szs::unsatisfiable();
        szs::begin_incomplete_proof();
        tstp::trivial_proof();
        szs::end_incomplete_proof();
        exit::success()
    }

    let (statistics, result) = search::search(&problem);

    let mut record = TSTP::default();
    if let Some(proof) = result {
        szs::unsatisfiable();
        szs::begin_incomplete_proof();
        let mut tableau = Tableau::new(&problem);
        for rule in proof {
            tableau.apply_rule(&mut record, &rule);
        }
        assert!(tableau.is_closed());
        assert!(tableau.solve_constraints());
        tableau.record_unification(&mut record);
        szs::end_incomplete_proof();
        statistics.record(&mut record);
        exit::success()
    } else {
        szs::unknown();
        statistics.record(&mut record);
        exit::failure()
    }
}
