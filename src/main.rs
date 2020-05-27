#![type_length_limit="10000000"]

mod atom;
mod clause;
mod goal;
mod io;
mod literal;
mod prelude;
mod problem;
mod record;
mod rule;
mod search;
mod solver;
mod symbol;
mod tableau;
mod term;
mod util;

fn main() {
    let problem = io::tptp::load_from_stdin();
    let mut queue = util::queue::Queue::default();
    if let Some(proof) = search::astar(&mut queue, &problem) {
        io::szs::unsatisfiable();
        io::szs::begin_refutation();
        let mut record = io::tstp::TSTP::default();
        let mut tableau = tableau::Tableau::new(&problem);
        for rule in proof {
            tableau.apply_rule(&mut record, rule);
        }
        assert!(tableau.is_closed());
        assert!(tableau.solve_constraints_correct());
        tableau.record_unification(&mut record);
        io::szs::end_refutation();
        io::exit::success()
    } else {
        io::szs::incomplete();
        io::exit::failure()
    }
}
