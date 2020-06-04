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
    let mut tableau = Tableau::new(problem);

    let mut possible = vec![];
    let mut script = VecDeque::new();
    while let Some(rule_list) = queue.dequeue() {
        let mut record = Silent; //crate::io::tstp::TSTP::default();
        script.clear();
        tableau.clear();
        for rule in rule_list.items() {
            script.push_front(*rule);
        }
        for rule in &script {
            tableau.apply_rule(&mut record, rule);
        }

        assert!(tableau.simplify_constraints());
        tableau.save();

        tableau.possible_rules(&mut possible);
        possible.sort();
        possible.dedup();
        for rule in possible.drain(..) {
            tableau.apply_rule(&mut Silent, &rule);
            if tableau.solve_constraints() {
                if tableau.is_closed() {
                    script.push_back(rule);
                    return Some(script);
                }
                let estimate =
                    tableau.num_open_branches() + (script.len() as u32);
                queue.enqueue(List::cons(&rule_list, rule), estimate);
            }
            tableau.restore();
        }
    }
    None
}
