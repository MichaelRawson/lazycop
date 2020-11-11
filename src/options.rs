use crate::output::Output;
use std::path::PathBuf;
use std::time::Instant;
use structopt::StructOpt;

const NAME: &str = "lazyCoP";

const ABOUT: &str = "
lazyCoP is an automatic theorem prover for first-order logic with equality.
The system can read the TPTP FOF and CNF dialects and outputs TSTP or Graphviz.
For more information, see the project repository.
";

#[derive(StructOpt)]
#[structopt(
    name = NAME,
    author,
    about = ABOUT,
    help_message = "print help information and exit",
    version_message = "print version information and exit"
)]
pub(crate) struct Options {
    /// input problem to attempt
    #[structopt(parse(from_os_str))]
    pub(crate) path: PathBuf,

    /// time limit (s)
    #[structopt(long)]
    pub(crate) time: Option<u64>,

    /// print clause normal form and exit
    #[structopt(long)]
    pub(crate) dump_clauses: bool,

    /// proof output
    #[structopt(
        long,
        default_value="tstp",
        possible_values=&["tstp", "training", "graphviz"],
    )]
    pub(crate) output: Output,

    #[structopt(skip = Instant::now())]
    start_time: Instant,
}

impl Options {
    pub(crate) fn parse() -> Self {
        Self::from_args()
    }

    pub(crate) fn within_time_limit(&self) -> bool {
        if let Some(time_limit) = self.time {
            let elapsed = self.start_time.elapsed().as_secs();
            elapsed < time_limit
        } else {
            true
        }
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
