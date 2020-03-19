mod queue;
mod rule_store;

use crate::core::tableau::Tableau;
use crate::output::record::Silent;
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
        let mut next_rules = vec![];
        let mut record = Silent;
        while let Some(script_id) = self.queue.deque() {
            let script = self.rule_store.get_script(script_id);
            self.tableau.reconstruct(&mut record, self.problem, &script);
            if self.tableau.blocked {
                continue;
            }
            if self.tableau.is_closed() {
                return Some(script);
            }

            let priority = (self.tableau.num_subgoals() + script.len()) as u32;
            next_rules.clear();
            self.tableau
                .fill_possible_rules(&mut next_rules, &self.problem);
            for next_rule in &next_rules {
                let next = self.rule_store.add_rule(script_id, *next_rule);
                self.queue.enqueue(next, priority);
            }
        }
        None
    }
}
