mod builder;

use crate::output::exit;
use crate::output::szs;
use crate::prelude::*;
use std::fmt;
use std::io::Read;
use tptp::parsers;
use tptp::syntax;
use tptp::syntax::Visitor;

fn read_stdin() -> Box<[u8]> {
    let mut buffer = vec![];
    std::io::stdin()
        .read_to_end(&mut buffer)
        .unwrap_or_else(|err| {
            println!("% failed to read from stdin: {}", err);
            szs::os_error();
            exit::failure()
        });
    buffer.into_boxed_slice()
}

fn report_inappropriate<T: fmt::Display>(t: T) -> ! {
    println!("% non-CNF input:\n{}", t);
    szs::inappropriate();
    exit::failure()
}

fn parse(bytes: &[u8]) -> Problem {
    let mut builder = builder::Builder::default();
    let mut inputs = parsers::tptp_input_iterator::<()>(bytes);

    for input in &mut inputs {
        if let syntax::TPTPInput::Annotated(formula) = input {
            if let syntax::AnnotatedFormula::Cnf(cnf) = *formula {
                builder.visit_cnf_annotated(cnf);
            } else {
                report_inappropriate(formula)
            }
        } else {
            report_inappropriate(input)
        }
    }

    if let Ok((bytes, _)) = inputs.finish() {
        if let Ok((b"", _)) = parsers::ignored::<()>(bytes) {
            return builder.finish();
        }
    }

    println!("% unsupported or invalid syntax in input");
    szs::input_error();
    exit::failure()
}

pub fn load_problem() -> Problem {
    let bytes = read_stdin();
    parse(&bytes)
}
