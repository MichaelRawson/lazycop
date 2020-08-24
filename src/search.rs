use crate::goal::Goal;
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
) -> Option<Vec<Rule>> {
    let mut rules = vec![];
    let mut possible = vec![];
    let mut goal = Goal::new(problem);
    let mut data = vec![];
    let mut leaf = take(&mut tree.lock(), &mut rules)?;

    loop {
        let mut record = Silent; //crate::io::tstp::TSTP::default();
        for rule in rules.iter() {
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
                    rules.push(rule);
                    return Some(rules);
                }

                let score = goal.num_literals();
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
        leaf = take(&mut tree, &mut rules)?;
    }
}

pub(crate) fn search(problem: &Problem) -> (Statistics, Option<Vec<Rule>>) {
    let mut statistics = Statistics::new(problem);
    let mut proof = None;
    let tree = Mutex::new(UCTree::default());

    thread::scope(|scope| {
        scope
            .builder()
            .name("search".into())
            .stack_size(STACK_SIZE)
            .spawn(|_| proof = search_task(problem, &mut statistics, &tree))
            .expect("failed to spawn search thread");
    })
    .unwrap_or_else(|_| panic!("worker thread crashed"));
    (statistics, proof)
}
