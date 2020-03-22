mod core;
mod index;
mod input;
mod output;
mod prelude;
mod search;
mod util;

fn main() {
    let problem = input::load_problem();
    if let Some(proof) = search::Search::new(&problem).search() {
        output::szs::unsatisfiable();
        output::szs::begin_refutation();
        let mut record = output::record::PrintProof::default();
        let mut tableau = core::tableau::Tableau::new(&problem);
        tableau.reconstruct(&mut record, &proof);
        output::szs::end_refutation();
    } else {
        output::szs::incomplete();
        output::exit::failure()
    }
}
