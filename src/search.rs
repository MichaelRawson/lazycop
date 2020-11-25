use crate::goal::Goal;
use crate::options::Options;
use crate::prelude::*;
#[cfg(feature = "smt")]
use crate::smt;
use crate::statistics::Statistics;
use crate::tree::Tree;
use crossbeam_utils::thread;
use spin::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "smt")]
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Duration;

pub(crate) enum SearchResult {
    Unsat(Vec<Vec<Rule>>),
    Exhausted,
    TimeOut,
}

pub(crate) struct Search<'a> {
    problem: &'a Problem,
    options: &'a Options,
    tree: Mutex<Tree>,
    stop: AtomicBool,
    statistics: Statistics,
    result: Mutex<SearchResult>,
    #[cfg(feature = "smt")]
    grounding_context: smt::Context,
}

impl<'a> Search<'a> {
    pub(crate) fn new(problem: &'a Problem, options: &'a Options) -> Self {
        let tree = Mutex::new(Tree::default());
        let stop = AtomicBool::new(false);
        let statistics = Statistics::new(problem);
        let result = Mutex::new(SearchResult::TimeOut);
        #[cfg(feature = "smt")]
        let grounding_context = smt::Context::default();
        Self {
            problem,
            options,
            tree,
            stop,
            statistics,
            result,
            #[cfg(feature = "smt")]
            grounding_context,
        }
    }

    pub(crate) fn go(self) -> (SearchResult, Statistics) {
        #[cfg(feature = "smt")]
        let (sender, receiver) = channel();

        thread::scope(|scope| {
            if self.options.time.is_some() {
                scope.spawn(|_| self.sleep_task());
            }
            #[cfg(feature = "smt")]
            scope.spawn(|_| self.smt_task(receiver));
            self.search_task(
                #[cfg(feature = "smt")]
                sender,
            );
        })
        .expect("thread panicked");
        (self.result.into_inner(), self.statistics)
    }

    fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    fn set_stop(&self) {
        self.stop.store(true, Ordering::Relaxed)
    }

    fn sleep_task(&self) {
        while !self.should_stop() {
            if self.options.within_time_limit() {
                std::thread::sleep(Duration::from_millis(10));
            } else {
                self.set_stop();
            }
        }
    }

    fn search_task(
        &self,
        #[cfg(feature = "smt")] sender: Sender<(
            Id<smt::Assertion>,
            smt::Assertion,
        )>,
    ) {
        let mut goal = Goal::new(self.problem);
        let mut rules = vec![];
        let mut possible = vec![];
        let mut children = vec![];

        #[cfg(feature = "smt")]
        let mut axioms = vec![];
        #[cfg(feature = "smt")]
        let mut grounder =
            smt::Grounder::new(&self.grounding_context, &self.problem.symbols);

        while !self.should_stop() {
            let leaf = {
                let tree = self.tree.lock();
                if tree.is_closed() {
                    self.set_stop();
                    *self.result.lock() = SearchResult::Exhausted;
                    return;
                }
                tree.select_for_expansion(&mut rules)
            };
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
                        self.set_stop();
                        let tree = self.tree.lock();
                        let mut proof = tree.derivation(leaf);
                        proof.push(rule);
                        let core = vec![proof];
                        *self.result.lock() = SearchResult::Unsat(core);
                        return;
                    }
                    children.push((rule, goal.tableau.size()));
                    self.statistics.increment_retained_leaves();
                } else {
                    self.statistics.increment_eliminated_leaves();
                }
                goal.restore();
            }
            self.tree.lock().expand(leaf, children.drain(..));
            self.statistics.increment_expanded_leaves();

            #[cfg(feature = "smt")]
            if sender
                .send((
                    leaf.transmute(),
                    grounder.ground(
                        &self.problem.symbols,
                        &goal.terms,
                        &goal.bindings,
                        &goal.tableau.literals,
                        axioms.drain(..),
                    ),
                ))
                .is_err()
            {
                self.set_stop();
                return;
            }
            goal.clear();
        }
    }

    #[cfg(feature = "smt")]
    fn smt_task(
        &self,
        receiver: Receiver<(Id<smt::Assertion>, smt::Assertion)>,
    ) {
        let context = smt::Context::default();
        let mut solver = smt::Solver::new(&context);
        loop {
            for (id, assertion) in receiver.try_iter() {
                if self.should_stop() {
                    return;
                }
                let assertion =
                    solver.translate(&self.grounding_context, assertion);
                self.statistics.increment_smt_assertions();
                solver.assert(id, assertion);
            }
            if self.should_stop() {
                return;
            }
            if solver.check() {
                self.set_stop();
                let tree = self.tree.lock();
                let mut core = vec![];
                for assertion in solver.unsat_core() {
                    core.push(tree.derivation(assertion.transmute()));
                }
                *self.result.lock() = SearchResult::Unsat(core);
                return;
            }
            self.statistics.increment_smt_checks();
        }
    }
}
