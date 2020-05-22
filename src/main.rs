#![allow(clippy::too_many_arguments)]

mod atom;
mod clause;
mod goal;
mod goal_stack;
mod literal;
mod problem;
mod rule;
mod solver;
mod symbol;
mod tableau;
mod term;
mod alloc;
mod io;
mod prelude;
mod search;
mod util;

fn main() {
    let problem = io::tptp::load_from_stdin();
    let mut queue = util::queue::Queue::default();
    if let Some(proof) = search::astar(&mut queue, &problem) {
        io::szs::unsatisfiable();
        io::szs::begin_refutation();
        let mut record = io::tptp::TPTPProof::default();
        let mut tableau = tableau::Tableau::new(&problem);
        for rule in proof {
            tableau.apply_rule(&mut record, rule);
        }
        assert!(tableau.is_closed());
        assert!(tableau.check_constraints());
        tableau.record_unification(&mut record);
        io::szs::end_refutation();
        io::exit::success()
    } else {
        io::szs::incomplete();
        io::exit::failure()
    }
}
