use crate::output::record::Record;
use crate::prelude::*;

#[derive(Default)]
pub struct Tableau {
    pub blocked: bool,
    term_list: TermList,
    subgoals: Vec<Subgoal>,
}

impl Tableau {
    pub fn duplicate(&mut self, other: &Self) {
        self.blocked = other.blocked;
        self.term_list.clear();
        self.term_list.copy_from(&other.term_list);
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
        self.term_list.clear();
        self.subgoals.clear();
        for rule in script {
            self.apply_rule(record, problem, *rule);
        }
    }

    pub fn apply_rule<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        rule: Rule,
    ) {
        assert!(!self.blocked);
        match rule {
            Rule::Start(clause_id) => {
                assert!(self.subgoals.is_empty());
                assert!(self.term_list.is_empty());

                let start_goal = Subgoal::start(
                    record,
                    &mut self.term_list,
                    problem,
                    clause_id,
                );
                self.subgoals.push(start_goal);
            }
            Rule::LazyPredicateExtension(clause_id, literal_id) => {
                assert!(!self.subgoals.is_empty());
                let mut subgoal = self.subgoals.pop().unwrap();
                assert!(!subgoal.is_done());

                let new_goal = subgoal.apply_lazy_predicate_extension(
                    record,
                    &mut self.term_list,
                    problem,
                    clause_id,
                    literal_id,
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

                if !subgoal.apply_equality_reduction(
                    record,
                    &mut self.term_list,
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

                if !subgoal.apply_predicate_reduction(
                    record,
                    &mut self.term_list,
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
        subgoal.possible_rules(possible, &problem, &self.term_list)
    }
}
