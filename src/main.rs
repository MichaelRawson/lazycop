#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate log;

mod core;
mod index;
mod input;
mod output;
mod prelude;
mod util;

fn main() {
    output::log::start_logging();
    let bytes = input::read_stdin();
    input::tptp::parse(&bytes);

    let mut rules = util::rule_store::RuleStore::default();
    let mut tableau = core::tableau::Tableau::default();
    info!("lazyCoP OK, proving...");
    tableau.clear();
}
