use std::path::PathBuf;
use structopt::StructOpt;

const NAME: &str = "lazyCoP";

const ABOUT: &str = "
lazyCoP is an automatic theorem prover for first-order logic with equality.
The system can read the TPTP FOF and CNF dialects and outputs SZS/TSTP.
For more information, see the project repository.
";

#[derive(StructOpt)]
#[structopt(name = NAME, author, about = ABOUT)]
pub(crate) struct Options {
    #[structopt(parse(from_os_str), help = "path to input problem")]
    pub(crate) path: PathBuf,

    #[structopt(long, help = "limit number of inference steps")]
    pub(crate) steps: Option<usize>,

    #[structopt(long, help = "convert input to clause normal form and exit")]
    pub(crate) clausify: bool,

    #[structopt(long, help = "dump training data on exit")]
    pub(crate) training_data: bool,

    #[structopt(
        long,
        help = "training data visit threshold",
        default_value = "1000"
    )]
    pub(crate) training_threshold: u32,
}

impl Options {
    pub(crate) fn parse() -> Self {
        Self::from_args()
    }

    pub(crate) fn problem_name(&self) -> String {
        self.path
            .file_stem()
            .expect("bad path, no file name")
            .to_string_lossy()
            .as_ref()
            .into()
    }
}
