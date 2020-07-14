use lazy::io::tstp::TSTP;
use lazy::io::{exit, szs, tptp};
use lazy::tableau::Tableau;

mod priority;
mod rule_store;
mod search;
mod statistics;

fn main() {
    heuristic::init();

    let problem = tptp::load_from_stdin();
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
