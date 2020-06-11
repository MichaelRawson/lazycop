use crate::prelude::*;
use crate::record::Silent;
use crate::tableau::Tableau;
use crate::util::queue::{Priority, Queue};

pub(crate) fn astar(
    rules: &mut Rules,
    queue: &mut Queue<Id<RuleList>>,
    problem: &Problem,
) -> Option<Vec<Rule>> {
    let mut tableau = Tableau::new(problem);

    let mut possible = vec![];
    let mut script = vec![];
    while let Some(id) = queue.dequeue() {
        let mut record = Silent; //crate::io::tstp::TSTP::default();
        script.clear();
        tableau.clear();

        script.extend(rules.get_list(id));
        for rule in script.iter().rev() {
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
                    script.reverse();
                    script.push(rule);
                    return Some(script);
                }
                let estimate =
                    tableau.num_open_branches() + (script.len() as u16);
                let precedence = rule.precedence();
                let priority = Priority {
                    estimate,
                    precedence,
                };

                let new = rules.add(Some(id), rule);
                queue.enqueue(new, priority);
            }
            tableau.restore();
        }
        rules.mark_done(id);
    }
    None
}
