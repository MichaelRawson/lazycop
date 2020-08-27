use crate::goal::Goal;
use crate::options::Options;
use crate::prelude::*;
use crate::record::Silent;
use crate::statistics::Statistics;
use crate::uctree::{UCTNode, UCTree};
use crossbeam_utils::thread;
use parking_lot::Mutex;

const STACK_SIZE: usize = 0x10_00000;

fn take(tree: &mut UCTree, rules: &mut Vec<Rule>) -> Option<Id<UCTNode>> {
    rules.clear();
    if tree.is_closed() {
        return None;
    }
    Some(tree.take(rules))
}

fn search_task(
    problem: &Problem,
    statistics: &mut Statistics,
    tree: &Mutex<UCTree>,
    steps: usize,
) -> Option<Vec<Rule>> {
    let mut rules = vec![];
    let mut proof = None;
    let mut possible = vec![];
    let mut goal = Goal::new(problem);
    let mut data = vec![];
    let mut leaf = take(&mut tree.lock(), &mut rules)?;

    for _ in 0..steps {
        let mut record = Silent; //crate::io::tstp::TSTP::default();
        for rule in &rules {
            goal.apply_rule(&mut record, rule);
        }
        let constraints_ok = goal.simplify_constraints();
        debug_assert!(constraints_ok);
        goal.possible_rules(&mut possible);
        goal.save();

        for rule in possible.drain(..) {
            goal.apply_rule(&mut Silent, &rule);
            if goal.solve_constraints() {
                if goal.is_closed() {
                    let mut script = rules.clone();
                    script.push(rule);
                    proof = Some(script);
                }

                let score = goal.num_open_branches();
                data.push((rule, score));
                statistics.increment_retained_goals();
            } else {
                statistics.increment_eliminated_goals();
            }
            goal.restore();
        }
        statistics.increment_expanded_nodes();
        goal.clear();

        let mut tree = tree.lock();
        tree.give(leaf, &*data);
        data.clear();

        if proof.is_some() {
            return proof;
        }

        leaf = take(&mut tree, &mut rules)?;
    }

    None
}

pub(crate) fn search(
    problem: &Problem,
    options: &Options,
) -> (Statistics, Option<Vec<Rule>>) {
    let mut statistics = Statistics::new(problem);
    let mut proof = None;
    let tree = Mutex::new(UCTree::default());
    let steps = options.steps.unwrap_or(usize::MAX);

    thread::scope(|scope| {
        scope
            .builder()
            .name("search".into())
            .stack_size(STACK_SIZE)
            .spawn(|_| {
                proof = search_task(problem, &mut statistics, &tree, steps)
            })
            .expect("failed to spawn search thread");
    })
    .unwrap_or_else(|_| panic!("worker thread crashed"));

    if options.dump_training_data {
        let tree = tree.into_inner();
        dump_training_data(problem, &tree, options.visit_minimum);
    }
    (statistics, proof)
}

fn dump_array<T: std::fmt::Display>(data: &[T]) {
    let mut data = data.iter();
    print!("[");
    if let Some(first) = data.next() {
        print!("{}", first);
    }
    for rest in data {
        print!(",{}", rest);
    }
    print!("]");
}

fn dump_training_data(problem: &Problem, tree: &UCTree, limit: u32) {
    let mut rules = vec![];
    let mut scores = vec![];
    let mut goal = Goal::new(problem);
    let mut graph = Graph::default();

    for id in tree.eligible_training_nodes(limit) {
        tree.rules_for_node(id, &mut rules);
        for rule in &rules {
            goal.apply_rule(&mut Silent, rule);
        }
        let constraints_ok = goal.simplify_constraints();
        debug_assert!(constraints_ok);
        goal.save();

        for (rule, score) in tree.child_rule_scores(id) {
            scores.push(score);
            goal.apply_rule(&mut Silent, &rule);
            let constraints_ok = goal.solve_constraints();
            debug_assert!(constraints_ok);
            debug_assert!(!goal.is_closed());
            goal.graph(&mut graph);
            graph.finish_subgraph();
            goal.restore();
        }

        if scores.len() > 1 {
            print!("{{");
            print!("\"nodes\":");
            dump_array(graph.node_labels());
            print!(",\"sources\":");
            dump_array(&graph.sources);
            print!(",\"targets\":");
            dump_array(&graph.targets);
            print!(",\"batch\":");
            dump_array(&graph.batch);
            print!(",\"scores\":");
            dump_array(&scores);
            println!("}}");
        }

        goal.clear();
        graph.clear();
        scores.clear();
    }
}
