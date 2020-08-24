use crate::problem::Problem;
use crate::record::Record;

pub(crate) struct Statistics {
    problem_equality: bool,
    problem_symbols: u32,
    problem_clauses: u32,
    start_clauses: u32,
    eliminated_goals: u32,
    retained_goals: u32,
    expanded_nodes: u32,
}

impl Statistics {
    pub(crate) fn new(problem: &Problem) -> Self {
        let problem_equality = problem.has_equality;
        let problem_symbols = problem.num_symbols();
        let problem_clauses = problem.num_clauses();
        let start_clauses = problem.num_start_clauses();
        let eliminated_goals = 0;
        let retained_goals = 0;
        let expanded_nodes = 0;
        Self {
            problem_equality,
            problem_symbols,
            problem_clauses,
            start_clauses,
            eliminated_goals,
            retained_goals,
            expanded_nodes,
        }
    }

    pub(crate) fn record<R: Record>(&self, record: &mut R) {
        record.statistic("equality problem", self.problem_equality);
        record.statistic("problem symbols", self.problem_symbols);
        record.statistic("problem clauses", self.problem_clauses);
        record.statistic("start clauses", self.start_clauses);
        record.statistic("eliminated goals", self.eliminated_goals);
        record.statistic("retained goals", self.retained_goals);
        record.statistic("expanded nodes", self.expanded_nodes);
    }

    pub(crate) fn increment_eliminated_goals(&mut self) {
        self.eliminated_goals += 1;
    }

    pub(crate) fn increment_retained_goals(&mut self) {
        self.retained_goals += 1;
    }

    pub(crate) fn increment_expanded_nodes(&mut self) {
        self.expanded_nodes += 1;
    }
}
