use crate::io::training;
use crate::io::tstp;
use crate::options::Options;
use crate::problem::Problem;
use crate::search::SearchResult;
use crate::statistics::Statistics;
use std::str::FromStr;

pub(crate) enum Output {
    TSTP,
    Training,
}

impl FromStr for Output {
    type Err = String;

    fn from_str(output: &str) -> Result<Self, Self::Err> {
        match output {
            "tstp" => Ok(Self::TSTP),
            "training" => Ok(Self::Training),
            _ => Err(format!("{}: not a valid proof output", output)),
        }
    }
}

impl Output {
    pub(crate) fn result(
        &self,
        options: &Options,
        problem: &Problem,
        result: SearchResult,
        statistics: &Statistics,
    ) -> ! {
        match self {
            Self::TSTP => tstp::output(options, problem, result, statistics),
            Self::Training => training::output(options, problem, result),
        }
    }
}
