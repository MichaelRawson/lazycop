mod queue;
mod rule_store;

use crate::core::tableau::Tableau;
use crate::core::unification::Safe;
use crate::output::record::Silent;
use crate::prelude::*;
use queue::Queue;
use rule_store::RuleStore;

pub struct Search<'problem> {
    problem: &'problem Problem,
    queue: Queue,
    rule_store: RuleStore,
    next_rules: Vec<Rule>,
    reconstruction: Tableau,
    next_step: Tableau,
}

impl<'problem> Search<'problem> {
    pub fn new(problem: &'problem Problem) -> Self {
        let mut queue = Queue::default();
        let mut rule_store = RuleStore::default();
        for rule in problem.start_rules() {
            let rule_id = rule_store.add_start_rule(rule);
            queue.enqueue(rule_id, 0);
        }

        let next_rules = vec![];
        let reconstruction = Tableau::default();
        let next_step = Tableau::default();
        Self {
            problem,
            queue,
            rule_store,
            next_rules,
            reconstruction,
            next_step,
        }
    }

    pub fn search(&mut self) -> Option<Vec<Rule>> {
        while let Some(parent_id) = self.queue.dequeue() {
            let script = self.rule_store.get_script(parent_id);
            let distance = script.len();
            self.reconstruction.reconstruct(
                &mut Silent,
                self.problem,
                &script,
            );
            assert!(!self.reconstruction.blocked);
            assert!(!self.reconstruction.is_closed());

            if let Some(rule) = self.expand(parent_id, distance) {
                let proof_id = self.rule_store.add_rule(parent_id, rule);
                let proof = self.rule_store.get_script(proof_id);
                return Some(proof);
            }
        }
        None
    }

    fn expand(
        &mut self,
        parent_id: Id<Rule>,
        distance: usize,
    ) -> Option<Rule> {
        self.next_rules.clear();
        self.reconstruction
            .fill_possible_rules(&mut self.next_rules, &self.problem);

        for next_rule in &self.next_rules {
            self.next_step.duplicate(&self.reconstruction);
            self.next_step.apply_rule::<_, Safe>(
                &mut Silent,
                &self.problem,
                *next_rule,
            );
            if self.next_step.blocked {
                continue;
            }
            if self.next_step.is_closed() {
                return Some(*next_rule);
            }

            let estimate = self.heuristic();
            let priority = (distance + estimate) as u32;
            let next = self.rule_store.add_rule(parent_id, *next_rule);
            self.queue.enqueue(next, priority);
        }
        None
    }

    fn heuristic(&self) -> usize {
        self.next_step.num_literals()
    }
}
