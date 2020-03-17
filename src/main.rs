#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate log;

mod core;
mod index;
mod input;
mod output;
mod prelude;
mod problem;
mod util;

fn main() {
    output::log::start_logging();
    let bytes = input::read_stdin();
    input::tptp::parse(&bytes);
    info!("lazyCoP OK, proving...");
}
