#[macro_use]
extern crate log;

mod input;
mod output;

fn main() {
    output::log::start_logging();
    let input = input::read_stdin();
    info!("OK, start proving...");
    println!("{:?}", input);
}
