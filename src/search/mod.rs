mod queue;
mod script;

use queue::Queue;
use script::Script;

use crate::output::record::Silent;
use crate::prelude::*;

fn heuristic(tableau: &Tableau) -> usize {
    tableau.num_literals()
}

#[derive(Default)]
pub struct Search {
    queue: Queue,
}

impl Search {
    pub fn search(&mut self, problem: &Problem) -> Option<Vec<Rule>> {
        self.queue.clear();
        for rule in problem.start_rules() {
            self.queue.enqueue(Script::start(rule), 0);
        }

        let mut reconstruction = Tableau::new(problem);
        let mut copy = Tableau::new(problem);
        let mut steps = 0;
        while let Some(script) = self.queue.dequeue() {
            if let Some(proof) =
                self.step(script, &mut reconstruction, &mut copy)
            {
                dbg!(steps);
                return Some(proof);
            }
            steps += 1;
        }
        None
    }

    fn step<'problem>(
        &mut self,
        script: Rc<Script>,
        reconstruction: &mut Tableau<'problem>,
        copy: &mut Tableau<'problem>,
    ) -> Option<Vec<Rule>> {
        let rules = script.rules();
        reconstruction.reconstruct(&mut Silent, &rules);
        assert!(!reconstruction.is_blocked());
        assert!(!reconstruction.is_closed());

        let mut next_rules = vec![];
        reconstruction.fill_possible_rules(&mut next_rules);

        for next_rule in next_rules {
            copy.duplicate(&reconstruction);
            copy.apply_rule::<Checked, _>(&mut Silent, next_rule);
            if copy.is_blocked() {
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
