use crate::goal::Goal;
use crate::options::Options;
use crate::prelude::*;
use crate::record::Silent;
use crate::statistics::Statistics;
use crate::tree::Tree;
use crossbeam_utils::thread;
use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const STACK_SIZE: usize = 0x10_00000;

pub(crate) enum SearchResult {
    Proof(Vec<Rule>),
    Exhausted,
    ResourceOut,
}

fn expansion_task(
    problem: &Problem,
    options: &Options,
    statistics: &Statistics,
    tree: &RwLock<Tree>,
    stop: &AtomicBool,
    steps: &AtomicUsize,
) -> SearchResult {
    let mut rules = vec![];
    let mut proof = None;
    let mut possible = vec![];
    let mut goal = Goal::new(problem);
    let mut data = vec![];

    loop {
        let should_stop = stop.load(Ordering::Relaxed);
        let steps_so_far = steps.fetch_add(1, Ordering::Relaxed);
        if should_stop || !options.within_resource_limits(steps_so_far) {
            return SearchResult::ResourceOut;
        }

        let leaf = {
            let tree = tree.read();

            if tree.is_closed() {
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

                data.push((rule, goal.size()));
                statistics.increment_retained_goals();
            } else {
                statistics.increment_eliminated_goals();
            }
            goal.restore();
        }

        tree.write().expand(leaf, &*data);
        statistics.increment_expanded_goals();
        goal.clear();
        rules.clear();
        data.clear();

        if let Some(proof) = proof {
            return SearchResult::Proof(proof);
        }
    }
}

#[cfg(feature = "cudann")]
fn evaluation_task(
    problem: &Problem,
    statistics: &Statistics,
    tree: &RwLock<Tree>,
    stop: &AtomicBool,
) {
    let mut rules = vec![];
    let mut inferences = vec![];
    let mut scores = vec![];
    let mut goal = Goal::new(problem);
    let mut graph = Graph::new(problem);

    while !stop.load(Ordering::Relaxed) {
        let node = if let Some(node) =
            tree.read().select_for_evaluation(&mut rules)
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

        inferences.extend(tree.read().child_rules(node));
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

        tree.write().evaluate(node, &scores);
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
    let result = Mutex::new(SearchResult::ResourceOut);
    let tree = RwLock::new(Tree::default());
    let steps = AtomicUsize::default();
    let stop = AtomicBool::new(false);

    thread::scope(|scope| {
        scope
            .builder()
            .name("search".into())
            .stack_size(STACK_SIZE)
            .spawn(|_| {
                let task_result = expansion_task(
                    problem,
                    options,
                    &statistics,
                    &tree,
                    &stop,
                    &steps,
                );
                stop.store(true, Ordering::Relaxed);

                let mut result = result.lock();
                match (&task_result, &*result) {
                    (SearchResult::Proof(_), _)
                    | (SearchResult::Exhausted, SearchResult::ResourceOut) => {
                        *result = task_result;
                    }
                    (SearchResult::ResourceOut, _)
                    | (SearchResult::Exhausted, _) => {}
                }
            })
            .expect("failed to spawn expansion thread");

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
