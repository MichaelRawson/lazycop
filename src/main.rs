#![type_length_limit = "10000000"]

mod atom;
mod binding;
mod clause;
mod constraint;
mod disequation_solver;
mod equation_solver;
mod goal;
mod io;
mod literal;
mod occurs;
mod ordering_solver;
mod prelude;
mod problem;
mod record;
mod rule;
mod search;
mod statistics;
mod symbol;
mod tableau;
mod term;
mod util;

fn main() {
    let problem = io::tptp::load_from_stdin();
    let (statistics, result) = search::search(&problem);

    let mut record = io::tstp::TSTP::default();
    if let Some(proof) = result {
        io::szs::unsatisfiable();
        io::szs::begin_incomplete_proof();
        let mut tableau = tableau::Tableau::new(&problem);
        for rule in proof {
            tableau.apply_rule(&mut record, &rule);
        }
        assert!(tableau.is_closed());
        assert!(tableau.solve_constraints());
        tableau.record_unification(&mut record);
        io::szs::end_incomplete_proof();
        statistics.record(&mut record);
        io::exit::success()
    } else {
        io::szs::unknown();
        statistics.record(&mut record);
        io::exit::failure()
    }
}
