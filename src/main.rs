mod core;
mod index;
mod input;
mod output;
mod prelude;
mod search;
mod util;

fn main() {
    let bytes = input::read_stdin();
    let problem = input::tptp::parse(&bytes);
    if let Some(proof) = search::Search::new(&problem).search() {
        output::szs::unsatisfiable();
        output::szs::begin_refutation();
        let mut record = output::proof::Proof;
        let mut tableau = core::tableau::Tableau::default();
        tableau.reconstruct(&mut record, &problem, &proof);
        output::szs::end_refutation();
    } else {
        output::szs::incomplete();
        output::exit::failure()
    }
}
