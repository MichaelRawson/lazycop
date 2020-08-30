use crate::prelude::*;
use std::path::PathBuf;
use structopt::StructOpt;

const NAME: &str = "lazyCoP";

const ABOUT: &str = "
lazyCoP is an automatic theorem prover for first-order logic with equality.
The system reads formulae in TPTP CNF from stdin and may output a proof in TSTP on stdout.
For more information, see the project repository.
";

pub(crate) enum Output {
    TSTP,
    Silent,
}

impl Output {
    fn new(tag: &str) -> Self {
        match tag {
            "tstp" => Self::TSTP,
            "silent" => Self::Silent,
            _ => unreachable(),
        }
    }
}

#[derive(StructOpt)]
#[structopt(name = NAME, author, about = ABOUT)]
pub(crate) struct Options {
    #[structopt(parse(from_os_str), help = "path to input problem")]
    pub(crate) path: PathBuf,

    #[structopt(
        long,
        help="proof output",
        possible_values=&["tstp", "silent"],
        default_value="tstp",
        parse(from_str=Output::new)
    )]
    pub(crate) output: Output,

    #[structopt(long, help = "limit number of inference steps")]
    pub(crate) steps: Option<usize>,

    #[structopt(long, help = "dump training data on exit")]
    pub(crate) dump_training_data: bool,

    #[structopt(
        long,
        help = "training data visit minimum",
        default_value = "1000"
    )]
    pub(crate) visit_minimum: u32,
}

impl Options {
    pub(crate) fn parse() -> Self {
        Self::from_args()
    }
}

/*
pub(crate) fn parse() -> Options {
    let matches= App::new(NAME)
        .version(env!("CARGO_PKG_VERSION"))
        .author(env!("CARGO_PKG_AUTHORS"))
        .about(ABOUT)
        .arg(Arg::with_name("output")
            .long("output")
            .value_name("OUTPUT")
            .takes_value(true)
            .possible_value("tstp")
            .possible_value("silent")
            .default_value("tstp")
            .help("proof output: SZS/TSTP or silent")
        )
        .arg(Arg::with_name("steps")
            .long("steps")
            .value_name("STEPS")
            .takes_value(true)
            .default_value(usize::MAX)
            .help("maximum number of inferences")
        )
        .get_matches();

    let output = match matches.value_of("output") {
        Some("tstp") => Output::TSTP,
        Some("silent") => Output::Silent,
        _ => unreachable()
    };

    let steps = usize::MAX;

    Options { output, steps }
}
*/
