mod queue;
mod script;

use queue::Queue;
use script::Script;

use crate::output::record::Silent;
use crate::prelude::*;
use std::rc::Rc;

fn heuristic(tableau: &Tableau) -> usize {
    tableau.num_literals()
}

pub struct Search<'problem> {
    problem: &'problem Problem,
    queue: Queue,
}

impl<'problem> Search<'problem> {
    pub fn new(problem: &'problem Problem) -> Self {
        let mut queue = Queue::default();
        for rule in problem.start_rules() {
            queue.enqueue(Script::start(rule), 0);
        }
        Self { problem, queue }
    }

    pub fn search(&mut self) -> Option<Vec<Rule>> {
        let mut reconstruction = Tableau::new(self.problem);
        let mut copy = Tableau::new(self.problem);
        while let Some(script) = self.queue.dequeue() {
            if let Some(proof) =
                self.step(script, &mut reconstruction, &mut copy)
            {
                return Some(proof);
            }
        }
        None
    }

    fn step(
        &mut self,
        script: Rc<Script>,
        reconstruction: &mut Tableau<'problem>,
        copy: &mut Tableau<'problem>,
    ) -> Option<Vec<Rule>> {
        let rules = script.rules();
        reconstruction.reconstruct(&mut Silent, &rules);
        assert!(!reconstruction.blocked);
        assert!(!reconstruction.is_closed());

        let mut next_rules = vec![];
        reconstruction.fill_possible_rules(&mut next_rules);

        for next_rule in next_rules {
            copy.duplicate(&reconstruction);
            copy.apply_rule::<_, CorrectUnification>(&mut Silent, next_rule);
            if copy.blocked {
                continue;
            }
            if copy.is_closed() {
                return Some(Script::new(script, next_rule).rules());
            }

            let estimate = heuristic(&copy);
            let distance = rules.len();
            let priority = (distance + estimate) as u32;
            let next_script = Script::new(script.clone(), next_rule);
            self.queue.enqueue(next_script, priority);
        }
        None
    }
}
