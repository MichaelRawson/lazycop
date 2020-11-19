use crate::goal::Goal;
use crate::options::Options;
use crate::prelude::*;
#[cfg(feature = "smt")]
use crate::smt;
use crate::statistics::Statistics;
use crate::tree::Tree;
use crossbeam_utils::thread;
use spin::mutex::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

const STACK_SIZE: usize = 0x10_00000;
#[cfg(feature = "smt")]
const SMT_INTERVAL: usize = 1024;

pub(crate) enum SearchResult {
    Proof(Vec<Rule>),
    #[cfg(feature = "smt")]
    Unsat(Vec<Vec<Rule>>),
    Exhausted,
    TimeOut,
}

fn search_task(
    problem: &Problem,
    options: &Options,
    statistics: &Statistics,
    tree: &Mutex<Tree>,
    stop: &AtomicBool,
) -> SearchResult {
    let mut goal = Goal::new(problem);
    let mut rules = vec![];
    let mut possible = vec![];
    let mut children = vec![];

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
                continue;
            }
        };
        for rule in &rules {
            goal.apply_rule(*rule);
        }
        let constraints_ok = goal.simplify_constraints();
        debug_assert!(constraints_ok);
        goal.possible_rules(&mut possible);
        goal.save();

        for rule in possible.drain(..) {
            goal.apply_rule(rule);
            if goal.solve_constraints() {
                if goal.is_closed() {
                    stop.store(true, Ordering::Relaxed);
                    rules.push(rule);
                    return SearchResult::Proof(rules);
                }

                children.push((rule, goal.tableau.size()));
                statistics.increment_retained_leaves();
            } else {
                statistics.increment_eliminated_leaves();
            }
            goal.restore();
        }
        tree.lock().expand(leaf, &*children);

        statistics.increment_expanded_leaves();
        goal.clear();
        rules.clear();
        children.clear();
    }
    stop.store(true, Ordering::Relaxed);
    SearchResult::TimeOut
}

#[cfg(feature = "smt")]
fn smt_task(
    problem: &Problem,
    statistics: &Statistics,
    tree: &Mutex<Tree>,
    stop: &AtomicBool,
) -> SearchResult {
    let mut goal = Goal::new(problem);
    let mut progress = Id::default() + Offset::new(1);
    let mut rules = vec![];
    let mut axioms = vec![];
    let mut count = 0;

    let mut solver = smt::Solver::new(&problem.symbols);
    while !stop.load(Ordering::Relaxed) {
        let id = if let Some(id) =
            tree.lock().select_for_smt(&mut rules, progress)
        {
            progress = id + Offset::new(1);
            id
        } else {
            std::thread::yield_now();
            continue;
        };
        for rule in rules.drain(..).rev() {
            axioms.extend(goal.apply_rule(rule));
        }
        let constraints_ok = goal.simplify_constraints();
        debug_assert!(constraints_ok);

        let assertion = solver.ground(
            &problem.symbols,
            &goal.terms,
            &goal.tableau.literals,
            &goal.bindings,
            axioms.drain(..),
        );
        solver.assert(id.transmute(), assertion);
        statistics.increment_smt_assertions();
        count += 1;
        if count % SMT_INTERVAL == 0 && solver.check() {
            stop.store(true, Ordering::Relaxed);
            let core = solver.unsat_core();
            let mut unsat = vec![];
            let tree = tree.lock();
            for assertion in core {
                unsat.push(tree.derivation(assertion.transmute()));
            }
            return SearchResult::Unsat(unsat);
        }

        goal.clear();
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
            goal.apply_rule(rule);
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
        statistics.increment_evaluated_leaves();
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

    let available = num_cpus::get();
    #[cfg(feature = "smt")]
    let available = available - 1;
    let num_search_threads =
        std::cmp::max(options.cores.unwrap_or(available), 1);

    thread::scope(|scope| {
        for thread in 0..num_search_threads {
            scope
                .builder()
                .name(format!("search-{}", thread))
                .stack_size(STACK_SIZE)
                .spawn(|_| {
                    let task_result = search_task(
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
                .expect("failed to spawn search thread");
        }

        #[cfg(feature = "smt")]
        scope
            .builder()
            .name("smt".into())
            .stack_size(STACK_SIZE)
            .spawn(|_| {
                let task_result = smt_task(problem, &statistics, &tree, &stop);
                let mut result = result.lock();
                if let SearchResult::TimeOut = &*result {
                    *result = task_result;
                }
            })
            .expect("failed to spawn SMT solver thread");

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
