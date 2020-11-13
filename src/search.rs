use crate::goal::Goal;
use crate::options::Options;
use crate::prelude::*;
use crate::record::Silent;
use crate::statistics::Statistics;
use crate::tree::Tree;
use crossbeam_utils::thread;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

const STACK_SIZE: usize = 0x10_00000;

pub(crate) enum SearchResult {
    Proof(Vec<Rule>),
    Exhausted,
    TimeOut,
}

fn expansion_task(
    problem: &Problem,
    options: &Options,
    statistics: &Statistics,
    tree: &Mutex<Tree>,
    stop: &AtomicBool,
) -> SearchResult {
    let mut rules = vec![];
    let mut possible = vec![];
    let mut goal = Goal::new(problem);
    let mut data = vec![];

    while !stop.load(Ordering::Relaxed) && options.within_time_limit() {
        let leaf = {
            let mut tree = tree.lock();
            if tree.is_closed() {
                stop.store(true, Ordering::Relaxed);
                return SearchResult::Exhausted;
            }
            if let Some(leaf) = tree.select_for_expansion(&mut rules) {
                leaf
            } else {
                rules.clear();
                std::thread::yield_now();
                continue;
            }
        };

        let mut record = Silent;
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
                    stop.store(true, Ordering::Relaxed);
                    rules.push(rule);
                    return SearchResult::Proof(rules);
                }

                data.push((rule, goal.size()));
                statistics.increment_retained_leaves();
            } else {
                statistics.increment_eliminated_leaves();
            }
            goal.restore();
        }

        tree.lock().expand(leaf, &*data);
        statistics.increment_expanded_leaves();
        goal.clear();
        rules.clear();
        data.clear();
    }
    SearchResult::TimeOut
}

#[cfg(feature = "cudann")]
fn evaluation_task(
    problem: &Problem,
    statistics: &Statistics,
    tree: &Mutex<Tree>,
    stop: &AtomicBool,
) {
    let mut rules = vec![];
    let mut inferences = vec![];
    let mut scores = vec![];
    let mut goal = Goal::new(problem);
    let mut graph = Graph::new(problem);

    while !stop.load(Ordering::Relaxed) {
        let node = if let Some(node) =
            tree.lock().select_for_evaluation(&mut rules)
        {
            node
        } else {
            rules.clear();
            std::thread::yield_now();
            continue;
        };
        for rule in rules.drain(..) {
            goal.apply_rule(&mut Silent, &rule);
        }
        let constraints_ok = goal.simplify_constraints();
        debug_assert!(constraints_ok);
        goal.save();

        inferences.extend(tree.lock().child_rules(node));
        debug_assert!(!inferences.is_empty());

        if inferences.len() == 1 {
            scores.clear();
            scores.push(0.0);
        } else {
            goal.graph(&mut graph, &inferences);
            let input = cudann::Input {
                nodes: graph.node_labels(),
                sources: &graph.sources,
                targets: &graph.targets,
                rules: &graph.rules,
            };
            cudann::model(input, &mut scores);
        }

        tree.lock().evaluate(node, &scores);
        inferences.clear();
        goal.clear();
        graph.clear();
        statistics.increment_evaluated_goals();
    }
}

pub(crate) fn search(
    problem: &Problem,
    options: &Options,
) -> (Statistics, SearchResult) {
    let statistics = Statistics::new(problem);
    let result = Mutex::new(SearchResult::TimeOut);
    let tree = Mutex::new(Tree::default());
    let stop = AtomicBool::new(false);

    thread::scope(|scope| {
        for cpu in 0..num_cpus::get() {
            scope
                .builder()
                .name(format!("search-{}", cpu))
                .stack_size(STACK_SIZE)
                .spawn(|_| {
                    let task_result = expansion_task(
                        problem,
                        options,
                        &statistics,
                        &tree,
                        &stop,
                    );

                    let mut result = result.lock();
                    if let SearchResult::TimeOut = &*result {
                        *result = task_result;
                    }
                })
                .expect("failed to spawn expansion thread");
        }

        #[cfg(feature = "cudann")]
        scope
            .builder()
            .name("evaluation".into())
            .stack_size(STACK_SIZE)
            .spawn(|_| {
                evaluation_task(problem, &statistics, &tree, &stop);
            })
            .expect("failed to spawn evaluation thread");
    })
    .unwrap_or_else(|_| panic!("worker thread crashed"));
    let result = result.into_inner();
    (statistics, result)
}
