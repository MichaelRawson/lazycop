use crate::prelude::*;
use crate::record::Silent;
use crate::tableau::Tableau;
use crate::util::list::List;
use crate::util::queue::Queue;
use std::collections::VecDeque;

pub(crate) fn astar(
    queue: &mut Queue<List<Rule>>,
    problem: &Problem,
) -> Option<VecDeque<Rule>> {
    let mut saved = Tableau::new(problem);
    let mut tableau = Tableau::new(problem);
    let mut possible = vec![];
    tableau.possible_rules(&mut possible);
    for rule in possible.drain(..) {
        queue.enqueue(List::new(rule), 0);
    }

    let mut script = VecDeque::new();
    let mut record = Silent; //crate::io::tptp::TPTPProof::default();
    while let Some(rule_list) = queue.dequeue() {
        script.clear();
        saved.clear();
        tableau.clear();
        for rule in rule_list.items() {
            script.push_front(*rule);
        }
        for rule in &script {
            tableau.apply_rule(&mut record, *rule);
        }

        saved.clone_from(&tableau);
        possible.clear();
        tableau.possible_rules(&mut possible);
        tableau.solve_constraints();
        for rule in &possible {
            tableau.apply_rule(&mut record, *rule);
            if tableau.check_constraints() {
                if tableau.is_closed() {
                    script.push_back(*rule);
                    return Some(script);
                }
                let estimate =
                    tableau.num_open_branches() + (script.len() as u32);
                queue.enqueue(List::cons(&rule_list, *rule), estimate);
            }
            tableau.clone_from(&saved);
        }
    }
    None
}
