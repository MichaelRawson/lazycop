use crate::core::unification::{Fast, UnificationPolicy};
use crate::output::record::Record;
use crate::prelude::*;

#[derive(Default)]
pub struct Tableau {
    pub blocked: bool,
    term_graph: TermGraph,
    subgoals: Vec<Subgoal>,
}

impl Tableau {
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

    pub fn reconstruct<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        script: &[Rule],
    ) {
        self.blocked = false;
        self.term_graph.clear();
        self.subgoals.clear();
        for rule in script {
            self.apply_rule::<_, Fast>(record, problem, *rule);
        }
    }

    pub fn apply_rule<R: Record, U: UnificationPolicy>(
        &mut self,
        record: &mut R,
        problem: &Problem,
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
                    problem,
                    clause_id,
                );
                self.subgoals.push(start_goal);
            }
            Rule::LazyPredicateExtension(coordinate) => {
                assert!(!self.subgoals.is_empty());
                let mut subgoal = self.subgoals.pop().unwrap();
                assert!(!subgoal.is_done());

                let new_goal = subgoal.apply_lazy_predicate_extension(
                    record,
                    &mut self.term_graph,
                    problem,
                    coordinate,
                );
                if !new_goal.is_done() {
                    self.subgoals.push(new_goal);
                }
                if !subgoal.is_done() {
                    self.subgoals.push(subgoal);
                }
            }
            Rule::EqualityReduction => {
                assert!(!self.subgoals.is_empty());
                let mut subgoal = self.subgoals.pop().unwrap();
                assert!(!subgoal.is_done());

                if !subgoal.apply_equality_reduction::<_, U>(
                    record,
                    &mut self.term_graph,
                    problem,
                ) {
                    self.blocked = true;
                    return;
                }
                if !subgoal.is_done() {
                    self.subgoals.push(subgoal);
                }
            }
            Rule::PredicateReduction(path_id) => {
                assert!(!self.subgoals.is_empty());
                let mut subgoal = self.subgoals.pop().unwrap();
                assert!(!subgoal.is_done());

                if !subgoal.apply_predicate_reduction::<_, U>(
                    record,
                    &mut self.term_graph,
                    problem,
                    path_id,
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

    pub fn fill_possible_rules(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
    ) {
        assert!(!self.blocked);
        assert!(!self.subgoals.is_empty());
        let subgoal = self.subgoals.last().unwrap();
        assert!(!subgoal.is_done());
        subgoal.possible_rules(possible, &problem, &self.term_graph)
    }
}
