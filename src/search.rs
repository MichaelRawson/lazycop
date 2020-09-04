use crate::goal::Goal;
use crate::options::Options;
use crate::prelude::*;
use crate::record::Silent;
use crate::statistics::Statistics;
use crate::training;
use crate::uctree::UCTree;
use crossbeam_utils::thread;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const STACK_SIZE: usize = 0x10_00000;

pub(crate) enum SearchResult {
    Proof(Vec<Rule>),
    Exhausted,
    ResourceOut,
}

fn task(
    problem: &Problem,
    options: &Options,
    statistics: &mut Statistics,
    tree: &Mutex<UCTree>,
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

        rules.clear();
        let leaf = {
            let mut tree = tree.lock();
            if tree.is_closed() {
                return SearchResult::Exhausted;
            }
            tree.take(&mut rules)
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

        if let Some(proof) = proof {
            return SearchResult::Proof(proof);
        }
    }
}

pub(crate) fn search(
    problem: &Problem,
    options: &Options,
) -> (Statistics, SearchResult) {
    let mut statistics = Statistics::new(problem);
    let result = Mutex::new(SearchResult::ResourceOut);
    let tree = Mutex::new(UCTree::default());
    let steps = AtomicUsize::default();
    let stop = AtomicBool::new(false);

    thread::scope(|scope| {
        scope
            .builder()
            .name("search".into())
            .stack_size(STACK_SIZE)
            .spawn(|_| {
                let task_result = task(
                    problem,
                    options,
                    &mut statistics,
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
            .expect("failed to spawn search thread");
    })
    .unwrap_or_else(|_| panic!("worker thread crashed"));
    let result = result.into_inner();
    let tree = tree.into_inner();

    if options.dump_training_data {
        training::dump(problem, &tree, options.max_training_data);
    }
    (statistics, result)
}
