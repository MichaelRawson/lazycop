use crate::core::tableau::Tableau;
use crate::io::record::Silent;
use crate::prelude::*;
use crate::util::imstack::ImStack;
use crate::util::queue::Queue;
use std::collections::VecDeque;

pub(crate) fn astar(
    queue: &mut Queue<ImStack<Rule>>,
    problem: &Problem,
) -> Option<VecDeque<Rule>> {
    queue.clear();
    queue.enqueue(ImStack::default(), 0);

    let mut tableau = Tableau::new(problem);
    let mut script = VecDeque::new();
    let mut possible = vec![];
    let mut record = Silent; //crate::io::tptp::TPTPProof::default();
    while let Some(rule_stack) = queue.dequeue() {
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
        tableau.solve_constraints();
        tableau.mark();
        for rule in &possible {
            tableau.apply_rule(&mut record, *rule);
            if tableau.check_constraints() {
                if tableau.is_closed() {
                    script.push_back(*rule);
                    return Some(script);
                }
                let estimate =
                    tableau.num_open_branches() + (script.len() as u32);
                queue.enqueue(rule_stack.push(*rule), estimate);
            }
            tableau.undo_to_mark();
        }
    }
    None
}
