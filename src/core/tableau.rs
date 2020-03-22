use crate::output::record::Record;
use crate::prelude::*;

pub struct Tableau<'problem> {
    problem: &'problem Problem,
    pub blocked: bool,
    term_graph: TermGraph,
    subgoals: Vec<Subgoal>,
}

impl<'problem> Tableau<'problem> {
    pub fn new(problem: &'problem Problem) -> Self {
        let blocked = false;
        let term_graph = TermGraph::default();
        let subgoals = vec![];
        Self {
            problem,
            blocked,
            term_graph,
            subgoals,
        }
    }

    pub fn duplicate(&mut self, other: &Self) {
        self.blocked = other.blocked;
        self.term_graph.clear();
        self.term_graph.copy_from(&other.term_graph);
        self.subgoals.clear();
        self.subgoals.extend(other.subgoals.iter().cloned());
    }

    pub fn is_closed(&self) -> bool {
        self.subgoals.is_empty()
    }

    pub fn num_literals(&self) -> usize {
        self.subgoals
            .iter()
            .map(|subgoal| subgoal.num_literals())
            .sum()
    }

    pub fn reconstruct<R: Record>(&mut self, record: &mut R, script: &[Rule]) {
        self.blocked = false;
        self.term_graph.clear();
        self.subgoals.clear();
        for rule in script {
            self.apply_rule::<_, FastUnification>(record, *rule);
        }
    }

    pub fn apply_rule<R: Record, U: UnificationAlgorithm>(
        &mut self,
        record: &mut R,
        rule: Rule,
    ) {
        assert!(!self.blocked);
        match rule {
            Rule::Start(clause_id) => {
                assert!(self.subgoals.is_empty());
                assert!(self.term_graph.is_empty());

                let start_goal = Subgoal::start(
                    record,
                    &mut self.term_graph,
                    self.problem,
                    clause_id,
                );
                self.subgoals.push(start_goal);
            }
            Rule::Extension(coordinate) => {
                assert!(!self.subgoals.is_empty());
                let mut subgoal = self.subgoals.pop().unwrap();
                assert!(!subgoal.is_done());

                let new_goal = subgoal.apply_extension(
                    record,
                    &mut self.term_graph,
                    self.problem,
                    coordinate,
                );
                if !new_goal.is_done() {
                    self.subgoals.push(new_goal);
                }
                if !subgoal.is_done() {
                    self.subgoals.push(subgoal);
                }
            }
            Rule::Reduction(path_id) => {
                assert!(!self.subgoals.is_empty());
                let mut subgoal = self.subgoals.pop().unwrap();
                assert!(!subgoal.is_done());

                if !subgoal.apply_reduction::<_, U>(
                    record,
                    &mut self.term_graph,
                    self.problem,
                    path_id,
                ) {
                    self.blocked = true;
                    return;
                }
                if !subgoal.is_done() {
                    self.subgoals.push(subgoal);
                }
            }
            Rule::Symmetry => {
                assert!(!self.subgoals.is_empty());
                let mut subgoal = self.subgoals.pop().unwrap();
                assert!(!subgoal.is_done());

                if !subgoal.apply_symmetry::<_, U>(
                    record,
                    &mut self.term_graph,
                    self.problem,
                ) {
                    self.blocked = true;
                    return;
                }
                if !subgoal.is_done() {
                    self.subgoals.push(subgoal);
                }
            }
        }
    }

    pub fn fill_possible_rules(&self, possible: &mut Vec<Rule>) {
        assert!(!self.blocked);
        assert!(!self.subgoals.is_empty());
        let subgoal = self.subgoals.last().unwrap();
        assert!(!subgoal.is_done());
        subgoal.possible_rules(possible, &self.problem, &self.term_graph)
    }
}
