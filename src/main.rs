#[macro_use]
extern crate log;

mod core;
mod input;
mod output;
mod prelude;
mod util;

fn main() {
    output::log::start_logging();
    let input = input::read_stdin();
    info!("OK, start proving...");
    println!("{:?}", input);
}
