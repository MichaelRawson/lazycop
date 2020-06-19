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
mod symbol;
mod tableau;
mod term;
mod util;

fn main() {
    let problem = io::tptp::load_from_stdin();

    if let Some(proof) = search::search(&problem) {
        io::szs::unsatisfiable();
        io::szs::begin_incomplete_proof();
        let mut record = io::tstp::TSTP::default();
        let mut tableau = tableau::Tableau::new(&problem);
        for rule in proof {
            tableau.apply_rule(&mut record, &rule);
        }
        assert!(tableau.is_closed());
        assert!(tableau.solve_constraints());
        tableau.record_unification(&mut record);
        io::szs::end_incomplete_proof();
        io::exit::success()
    } else {
        io::szs::unknown();
        io::exit::failure()
    }
}
