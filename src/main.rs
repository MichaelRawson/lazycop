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
    let problem = std::sync::Arc::new(problem);

    if let Some(proof) = search::Search::new(&problem).go() {
        io::szs::unsatisfiable();
        io::szs::begin_refutation();
        let mut record = io::tstp::TSTP::default();
        let mut tableau = tableau::Tableau::new(&problem);
        for rule in proof {
            tableau.apply_rule(&mut record, &rule);
        }
        assert!(tableau.is_closed());
        assert!(tableau.solve_constraints());
        tableau.record_unification(&mut record);
        io::szs::end_refutation();
        io::exit::success()
    } else {
        io::szs::incomplete();
        io::exit::failure()
    }
}
