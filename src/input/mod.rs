pub mod tptp;

use crate::output::exit;
use crate::output::szs;
use std::io::Read;

pub fn read_stdin() -> Box<[u8]> {
    let mut buffer = vec![];
    std::io::stdin()
        .read_to_end(&mut buffer)
        .unwrap_or_else(|err| {
            error!("failed to read from stdin: {}", err);
            szs::os_error();
            exit::failure()
        });
    buffer.into_boxed_slice()
}
