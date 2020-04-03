mod core;
mod io;
mod prelude;
//mod search;
mod util;

fn main() {
    let problem = io::tptp::load_from_stdin();
    /*
    if let Some(proof) = search::Search::default().search(&problem) {
        io::szs::unsatisfiable();
        io::szs::begin_refutation();
        //let mut record = output::record::PrintProof::default();
        //let mut tableau = core::tableau::Tableau::new(&problem);
        //tableau.reconstruct(&mut record, &proof);
        io::szs::end_refutation();
        io::exit::success()
    } else {
        io::szs::incomplete();
        io::exit::failure()
    }
    */
}
