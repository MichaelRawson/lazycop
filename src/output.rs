use crate::goal::Goal;
use crate::io::training;
use crate::io::tstp::TSTP;
use crate::io::{exit, szs};
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

fn tstp(
    options: &Options,
    problem: &Problem,
    result: SearchResult,
    statistics: &Statistics,
) -> ! {
    let name = options.problem_name();
    let mut tstp = TSTP::default();
    let info = &problem.info;
    match result {
        SearchResult::Proof(proof) => {
            let mut axioms = vec![];
            let mut goal = Goal::new(&problem);
            for rule in proof {
                axioms.extend(goal.apply_rule(rule));
            }
            let ok = goal.is_closed() && goal.solve_constraints();
            debug_assert!(ok);
            let (terms, literals, bindings) = goal.destruct();

            if !info.is_cnf && info.has_conjecture {
                szs::theorem(&name);
            } else {
                szs::unsatisfiable(&name);
            }
            szs::begin_proof(&name);
            for axiom in axioms {
                tstp.print_proof_clause(
                    problem, &terms, &literals, &bindings, axiom,
                );
            }
            szs::end_proof(&name);
            tstp.print_statistics(&statistics);
            exit::success()
        }
        SearchResult::Exhausted => {
            match (info.is_cnf, info.has_axioms, info.has_conjecture) {
                (false, false, true) => {
                    szs::counter_satisfiable(&name);
                }
                (_, true, true) => {
                    szs::gave_up(&name);
                }
                (true, _, _) | (false, true, _) | (false, false, false) => {
                    szs::satisfiable(&name);
                }
            }
            tstp.print_statistics(&statistics);
            exit::failure()
        }
        SearchResult::TimeOut => {
            szs::time_out(&name);
            tstp.print_statistics(&statistics);
            exit::failure()
        }
    }
}

fn training(options: &Options, problem: &Problem, result: SearchResult) -> ! {
    if let SearchResult::Proof(proof) = result {
        training::dump(options, problem, proof);
        exit::success()
    } else {
        exit::failure()
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
            Self::TSTP => tstp(options, problem, result, statistics),
            Self::Training => training(options, problem, result),
        }
    }
}
