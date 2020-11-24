use crate::goal::Goal;
use crate::options::Options;
use crate::prelude::*;
#[cfg(feature = "smt")]
use crate::smt;
use crate::statistics::Statistics;
use crate::tree::Tree;
#[cfg(feature = "smt")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "smt")]
use std::sync::mpsc::{channel, Receiver, Sender};
#[cfg(feature = "smt")]
use std::sync::Arc;
#[cfg(feature = "smt")]
use std::thread;

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
    tree: &mut Tree,
    #[cfg(feature = "smt")] context: &smt::Context,
    #[cfg(feature = "smt")] sender: Sender<(
        Id<smt::Assertion>,
        smt::Assertion,
    )>,
    #[cfg(feature = "smt")] stop: &AtomicBool,
) -> Option<SearchResult> {
    let mut goal = Goal::new(problem);
    let mut rules = vec![];
    let mut possible = vec![];
    let mut children = vec![];

    #[cfg(feature = "smt")]
    let mut axioms = vec![];
    #[cfg(feature = "smt")]
    let mut grounder = smt::Grounder::new(&context, &problem.symbols);

    loop {
        if !options.within_time_limit() {
            #[cfg(feature = "smt")]
            stop.store(true, Ordering::Relaxed);
            return Some(SearchResult::TimeOut);
        }
        #[cfg(feature = "smt")]
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        if tree.is_closed() {
            #[cfg(feature = "smt")]
            stop.store(true, Ordering::Relaxed);
            return Some(SearchResult::Exhausted);
        }
        let leaf = tree.select_for_expansion(&mut rules);
        for rule in rules.drain(..) {
            #[cfg(feature = "smt")]
            axioms.extend(goal.apply_rule(rule));
            #[cfg(not(feature = "smt"))]
            goal.apply_rule(rule);
        }
        let constraints_ok = goal.simplify_constraints();
        debug_assert!(constraints_ok);
        goal.possible_rules(&mut possible);
        goal.save();

        for rule in possible.drain(..) {
            goal.apply_rule(rule);
            if goal.solve_constraints() {
                if goal.is_closed() {
                    #[cfg(feature = "smt")]
                    stop.store(true, Ordering::Relaxed);
                    let mut proof = tree.derivation(leaf);
                    proof.push(rule);
                    return Some(SearchResult::Proof(proof));
                }

                children.push((rule, goal.tableau.size()));
                statistics.increment_retained_leaves();
            } else {
                statistics.increment_eliminated_leaves();
            }
            goal.restore();
        }
        tree.expand(leaf, children.drain(..));
        statistics.increment_expanded_leaves();

        #[cfg(feature = "smt")]
        if sender
            .send((
                leaf.transmute(),
                grounder.ground(
                    &problem.symbols,
                    &goal.terms,
                    &goal.bindings,
                    &goal.tableau.literals,
                    axioms.drain(..),
                ),
            ))
            .is_err()
        {
            return None;
        }
        goal.clear();
    }
}

#[cfg(feature = "smt")]
fn smt_task(
    statistics: Arc<Statistics>,
    from: Arc<smt::Context>,
    receiver: Receiver<(Id<smt::Assertion>, smt::Assertion)>,
    stop: Arc<AtomicBool>,
) -> Option<Vec<Id<smt::Assertion>>> {
    let mut context = smt::Context::default();
    let mut solver = smt::Solver::new(&context);
    while !stop.load(Ordering::Relaxed) {
        for (id, assertion) in receiver.try_iter() {
            let assertion = context.translate(&from, assertion);
            statistics.increment_smt_assertions();
            solver.assert(id, assertion);

            if stop.load(Ordering::Relaxed) {
                return None;
            }
        }

        if solver.check() {
            stop.store(true, Ordering::Relaxed);
            return Some(solver.unsat_core());
        }
        statistics.increment_smt_checks();
    }
    None
}

pub(crate) fn search(
    problem: &Problem,
    options: &Options,
) -> (Statistics, SearchResult) {
    let statistics = Statistics::new(&problem);
    #[cfg(feature = "smt")]
    let statistics = Arc::new(statistics);
    #[cfg(feature = "smt")]
    let context = Arc::new(smt::Context::default());
    #[cfg(feature = "smt")]
    let stop = Arc::new(AtomicBool::new(false));

    #[cfg(feature = "smt")]
    let (sender, receiver) = channel();
    #[cfg(feature = "smt")]
    let smt = {
        let smt_statistics = statistics.clone();
        let smt_from = context.clone();
        let smt_stop = stop.clone();
        thread::spawn(move || {
            smt_task(smt_statistics, smt_from, receiver, smt_stop)
        })
    };

    let mut tree = Tree::default();
    let search_result = search_task(
        &problem,
        options,
        &statistics,
        &mut tree,
        #[cfg(feature = "smt")]
        &context,
        #[cfg(feature = "smt")]
        sender,
        #[cfg(feature = "smt")]
        &stop,
    );

    #[cfg(feature = "smt")]
    let smt_result = smt.join().expect("SMT thread crashed");
    let result = search_result;
    #[cfg(feature = "smt")]
    let result = result.or_else(|| {
        let mut unsat = vec![];
        let core = smt_result?;
        for assertion in core {
            unsat.push(tree.derivation(assertion.transmute()));
        }
        Some(SearchResult::Unsat(unsat))
    });
    let result = result.unwrap_or(SearchResult::TimeOut);

    #[cfg(feature = "smt")]
    let statistics = Arc::try_unwrap(statistics)
        .unwrap_or_else(|_| panic!("thread should have joined"));
    (statistics, result)
}
