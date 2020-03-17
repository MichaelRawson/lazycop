#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate log;

mod core;
mod index;
mod input;
mod output;
mod prelude;
mod search;
mod util;

fn main() {
    output::log::start_logging();
    let bytes = input::read_stdin();
    let problem = input::tptp::parse(&bytes);
    if let Some(_proof) = search::Search::new(&problem).search() {
        println!("proved it");
    }
    else {
        output::szs::incomplete();
        output::exit::failure()
    }
}
