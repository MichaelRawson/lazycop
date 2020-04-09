mod core;
mod io;
mod prelude;
mod search;
mod util;

fn main() {
    let problem = io::tptp::load_from_stdin();
    if let Some(proof) = search::astar(&problem) {
        io::szs::unsatisfiable();
        io::szs::begin_refutation();
        let mut record = io::tptp::TPTPProof::default();
        let mut tableau = core::tableau::Tableau::new(&problem);
        for rule in proof {
            tableau.apply_rule(&mut record, rule);
        }
        assert!(tableau.solve_constraints(&mut record));
        io::szs::end_refutation();
        io::exit::success()
    } else {
        io::szs::incomplete();
        io::exit::failure()
    }
}
