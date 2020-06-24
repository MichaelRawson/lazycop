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

const MAXIMUM: usize = 100_000;

#[derive(Serialize)]
struct Item<'a> {
    heuristic: u16,
    actual: u16,
    nodes: &'a [u32],
    from: &'a [u32],
    to: &'a [u32],
}

fn main() {
    let problem = tptp::load_from_stdin();
    let mut tableau = Tableau::new(&problem);
    let mut record = Silent;
    let mut rules = RuleStore::default();
    let mut rule_list = vec![];
    let mut possible = vec![];
    let mut queue = Queue::default();
    let mut graph = Graph::default();
    let mut nodes = vec![];
    let mut from = vec![];
    let mut to = vec![];
    let mut expanded = 0;
    queue.enqueue(0, None);

    while let Some(id) = queue.dequeue() {
        if expanded > MAXIMUM {
            break;
        }

        rule_list.clear();
        rule_list.extend(rules.get_list(id));
        for rule in rule_list.iter().rev() {
            tableau.apply_rule(&mut record, rule);
        }
        if !tableau.simplify_constraints() {
            unreachable()
        }

        tableau.possible_rules(&mut possible);
        tableau.save();
        for rule in possible.drain(..) {
            tableau.apply_rule(&mut Silent, &rule);
            if tableau.solve_constraints() {
                if tableau.is_closed() {
                    rules.add(id, rule, 0);
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

    rules.punish_expanded();
    rules.recompute_heuristics();
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    for id in rules.examples() {
        rule_list.clear();
        rule_list.extend(rules.get_list(Some(id)));
        for rule in rule_list.iter().rev() {
            tableau.apply_rule(&mut record, rule);
        }
        if !tableau.simplify_constraints() {
            unreachable()
        }

        let heuristic = if tableau.is_closed() {
            continue;
        } else {
            tableau.num_open_branches()
        };
        let actual = rules.get_heuristic(id);

        tableau.graph(&mut graph);
        nodes.extend(graph.nodes.range().map(|id| graph.nodes[id] as u32));
        from.extend(graph.from.iter().map(|id| id.as_usize() as u32 - 1));
        to.extend(graph.to.iter().map(|id| id.as_usize() as u32 - 1));
        let item = Item {
            heuristic,
            actual,
            nodes: nodes.as_ref(),
            from: from.as_ref(),
            to: to.as_ref(),
        };

        to_writer(&mut lock, &item).expect("failed to write record");
        writeln!(&mut lock).expect("failed to write newline");

        tableau.clear();
        graph.clear();
        from.clear();
        to.clear();
        nodes.clear();
    }
}
