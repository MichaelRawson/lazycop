use crate::goal::Goal;
use crate::io::tstp::TSTP;
use crate::io::{exit, szs};
use crate::problem::Problem;
use crate::search::SearchResult;
use crate::statistics::Statistics;
use std::str::FromStr;

#[derive(Default)]
pub(crate) struct OutputInfo {
    pub(crate) is_cnf: bool,
    pub(crate) has_axioms: bool,
    pub(crate) has_conjecture: bool,
}

pub(crate) enum Output {
    TSTP,
}

impl FromStr for Output {
    type Err = String;

    fn from_str(output: &str) -> Result<Self, Self::Err> {
        match output {
            "tstp" => Ok(Self::TSTP),
            _ => Err(format!("{}: not a valid proof output", output)),
        }
    }
}

fn tstp(
    name: &str,
    problem: &Problem,
    info: OutputInfo,
    result: SearchResult,
    statistics: &Statistics,
) -> ! {
    let mut record = TSTP::default();
    let OutputInfo {
        is_cnf,
        has_axioms,
        has_conjecture,
    } = info;
    match result {
        SearchResult::Proof(proof) => {
            if !is_cnf && has_conjecture {
                szs::theorem(&name);
            } else {
                szs::unsatisfiable(&name);
            }
            szs::begin_cnf_refutation(&name);
            let mut goal = Goal::new(&problem);
            for rule in proof {
                goal.apply_rule(&mut record, &rule);
            }
            let ok = goal.is_closed() && goal.solve_constraints();
            debug_assert!(ok);
            szs::end_cnf_refutation(&name);
            statistics.record(&mut record);
            exit::success()
        }
        SearchResult::Exhausted => {
            match (is_cnf, has_axioms, has_conjecture) {
                (false, false, true) => {
                    szs::counter_satisfiable(&name);
                }
                (_, true, true) => {
                    szs::unknown(&name);
                }
                (true, _, _) | (false, true, _) | (false, false, false) => {
                    szs::satisfiable(&name);
                }
            }
            statistics.record(&mut record);
            exit::failure()
        }
        SearchResult::ResourceOut => {
            szs::resource_out(&name);
            statistics.record(&mut record);
            exit::failure()
        }
    }
}

impl Output {
    pub(crate) fn result(
        &self,
        name: &str,
        problem: &Problem,
        info: OutputInfo,
        result: SearchResult,
        statistics: &Statistics,
    ) -> ! {
        match self {
            Self::TSTP => tstp(name, problem, info, result, statistics),
        }
    }
}
