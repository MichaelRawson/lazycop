use std::path::PathBuf;
use std::time::Instant;
use structopt::StructOpt;

const NAME: &str = "lazyCoP";

const ABOUT: &str = "
lazyCoP is an automatic theorem prover for first-order logic with equality.
The system can read the TPTP FOF and CNF dialects and outputs TSTP.
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

    /// step limit
    #[structopt(long)]
    pub(crate) steps: Option<usize>,

    /// print clause normal form and exit
    #[structopt(long)]
    pub(crate) clausify: bool,

    /// dump training data instead of TSTP
    #[structopt(long)]
    pub(crate) dump_training_data: bool,

    #[structopt(skip = Instant::now())]
    start_time: Instant,
}

impl Options {
    pub(crate) fn parse() -> Self {
        Self::from_args()
    }

    pub(crate) fn within_resource_limits(&self, steps: usize) -> bool {
        if let Some(max_step) = self.steps {
            if steps >= max_step {
                return false;
            }
        }

        if let Some(time_limit) = self.time {
            let elapsed = self.start_time.elapsed().as_secs();
            if elapsed >= time_limit {
                return false;
            }
        }

        true
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
