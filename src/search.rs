use crate::core::tableau::Tableau;
use crate::io::record::Silent;
use crate::prelude::*;
use crate::util::queue::Queue;
use crate::util::rc_stack::RcStack;
use std::collections::VecDeque;

pub(crate) fn astar(problem: &Problem) -> Option<VecDeque<Rule>> {
    let mut queue = Queue::default();
    queue.enqueue(RcStack::default(), 0);

    let mut script = VecDeque::new();
    let mut possible = vec![];
    let mut tableau = Tableau::new(problem);
    let mut record = Silent; //crate::io::tptp::TPTPProof::default();
    let mut steps = 0 as usize;
    while let Some(rule_stack) = queue.dequeue() {
        steps += 1;
        script.clear();
        tableau.clear();
        for rule in rule_stack.items() {
            script.push_front(*rule);
        }
        for rule in &script {
            tableau.apply_rule(&mut record, *rule);
        }

        possible.clear();
        tableau.possible_rules(&mut possible);
        tableau.mark();
        for rule in &possible {
            tableau.apply_rule(&mut record, *rule);
            if tableau.solve_constraints(&mut record) {
                if tableau.is_closed() {
                    println!("% proof found in {} steps", steps);
                    script.push_back(*rule);
                    return Some(script);
                }
                let estimate = tableau.open_branches() + (script.len() as u32);
                queue.enqueue(rule_stack.push(*rule), estimate);
            }
            tableau.undo();
        }
    }
    None
}
