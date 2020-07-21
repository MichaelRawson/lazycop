use crate::rule_store::RuleStore;
use lazy::io::tptp;
use lazy::prelude::*;
use lazy::record::Silent;
use lazy::tableau::Tableau;
use lazy::util::queue::Queue;
use serde::Serialize;
use serde_json::to_writer;
use std::io::Write;

mod rule_store;

const EXPANSIONS: usize = 1_000_000;

#[derive(Serialize)]
struct Item<'a> {
    heuristic: u16,
    estimate: u16,
    nodes: &'a [u32],
    from: &'a [u32],
    to: &'a [u32],
}

fn main() {
    let problem = tptp::load_from_stdin();
    if problem.is_trivial() {
        return;
    }
    let mut tableau = Tableau::new(&problem);
    let mut record = Silent;
    let mut rules = RuleStore::default();
    let mut rule_list = vec![];
    let mut possible = vec![];
    let mut queue = Queue::default();
    let mut graph = Graph::default();
    let mut nodes = vec![];
    let mut expanded = 0;
    let mut limit = std::u16::MAX;
    queue.enqueue(0, None);

    while expanded < EXPANSIONS {
        let (priority, id) = if let Some(item) = queue.dequeue() {
            item
        } else {
            break;
        };
        if priority > limit {
            break;
        }

        rule_list.clear();
        rule_list.extend(rules.get_list(id));
        for rule in rule_list.iter().rev() {
            tableau.apply_rule(&mut record, rule);
        }
        tableau.simplify_constraints();
        tableau.possible_rules(&mut possible);
        tableau.save();

        for rule in possible.drain(..) {
            tableau.apply_rule(&mut Silent, &rule);
            if tableau.solve_constraints() {
                if tableau.is_closed() {
                    rules.add(id, rule, 0);
                    limit = priority;
                } else {
                    let heuristic = tableau.num_open_branches();
                    let id = rules.add(id, rule, heuristic);
                    let distance = rule_list.len() as u16;
                    let priority = distance + heuristic;
                    queue.enqueue(priority, Some(id));
                }
            }
            tableau.restore();
        }
        if let Some(id) = id {
            rules.mark_expanded(id);
        }
        expanded += 1;
        tableau.clear();
    }

    rules.bubble_up();
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    for id in rules.examples() {
        let heuristic = rules.get_heuristic(id);
        let estimate = rules.get_estimate(id);

        rule_list.clear();
        rule_list.extend(rules.get_list(Some(id)));
        for rule in rule_list.iter().rev() {
            tableau.apply_rule(&mut record, rule);
        }
        tableau.simplify_constraints();
        tableau.graph(&mut graph);

        nodes.extend(graph.nodes.range().map(|id| graph.nodes[id] as u32));
        let item = Item {
            heuristic,
            estimate,
            nodes: nodes.as_ref(),
            from: graph.from.as_ref(),
            to: graph.to.as_ref(),
        };

        to_writer(&mut lock, &item).expect("failed to write record");
        writeln!(&mut lock).expect("failed to write newline");

        tableau.clear();
        graph.clear();
        nodes.clear();
    }
}
