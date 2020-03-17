mod queue;
mod rule_store;

use crate::core::tableau::Tableau;
use crate::output::proof::NoProof;
use crate::prelude::*;
use queue::Queue;
use rule_store::RuleStore;

pub struct Search<'problem> {
    problem: &'problem Problem,
    queue: Queue,
    rule_store: RuleStore,
    tableau: Tableau,
}

impl<'problem> Search<'problem> {
    pub fn new(problem: &'problem Problem) -> Self {
        let mut queue = Queue::default();
        let mut rule_store = RuleStore::default();
        for rule in problem.start_rules() {
            let rule_id = rule_store.add_start_rule(rule);
            queue.enqueue(rule_id, 0);
        }

        let tableau = Tableau::default();
        Self {
            problem,
            queue,
            rule_store,
            tableau,
        }
    }

    pub fn search(&mut self) -> Option<Vec<Rule>> {
        let mut record = NoProof;
        while let Some(rule_id) = self.queue.deque() {
            let script = self.rule_store.get_script(rule_id);
            self.tableau.reconstruct(&mut record, self.problem, &script);
            if self.tableau.is_closed() {
                return Some(script);
            }

            let priority = self.tableau.num_subgoals();
            for possible in self.tableau.possible_rules() {
                let possible_id = self.rule_store.add_rule(rule_id, possible);
                self.queue.enqueue(possible_id, priority);
            }
            self.tableau.clear();
        }
        None
    }
}