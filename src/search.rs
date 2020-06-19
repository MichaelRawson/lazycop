use crate::prelude::*;
use crate::record::Silent;
use crate::statistics::Statistics;
use crate::tableau::Tableau;
use crate::util::queue::{Priority, Queue};
use crossbeam_utils::thread;
use parking_lot::Mutex;

const STACK_SIZE: usize = 0x10_00000;

struct Attempt {
    rules: Rules,
    queue: Queue<Option<Id<RuleList>>>,
    proof: Option<Vec<Rule>>,
    in_flight: u16,
}

impl Default for Attempt {
    fn default() -> Self {
        let rules = Rules::default();
        let proof = None;
        let in_flight = 0;
        let mut queue = Queue::default();
        let priority = Priority {
            estimate: 0,
            precedence: 0,
        };
        queue.enqueue(None, priority);
        Self {
            rules,
            queue,
            proof,
            in_flight,
        }
    }
}

fn dequeue(
    attempt: &Mutex<Attempt>,
    leaf: &mut Option<Id<RuleList>>,
    rules: &mut Vec<Rule>,
) -> bool {
    rules.clear();
    loop {
        let mut attempt = attempt.lock();
        if attempt.proof.is_some() {
            return false;
        }

        if let Some(id) = attempt.queue.dequeue() {
            *leaf = id;
            rules.extend(attempt.rules.get_list(id));
            attempt.in_flight += 1;
            return true;
        } else if attempt.in_flight == 0 {
            return false;
        } else {
            std::mem::drop(attempt);
            std::thread::yield_now();
        }
    }
}

fn enqueue(
    attempt: &Mutex<Attempt>,
    leaf: Option<Id<RuleList>>,
    rule: Rule,
    priority: Priority,
) {
    let mut attempt = attempt.lock();
    let new = attempt.rules.add(leaf, rule);
    attempt.queue.enqueue(Some(new), priority);
}

fn finish(attempt: &Mutex<Attempt>, leaf: Option<Id<RuleList>>) -> u16 {
    let mut attempt = attempt.lock();
    attempt.in_flight -= 1;
    attempt.rules.mark_done(leaf)
}

fn found_proof(attempt: &Mutex<Attempt>, proof: Vec<Rule>) {
    let mut attempt = attempt.lock();
    attempt.proof = Some(proof);
}

fn heuristic(rule: &Rule, rules: &[Rule], tableau: &Tableau) -> Priority {
    let estimate = tableau.num_open_branches() + (rules.len() as u16);
    let precedence = rule.precedence();
    Priority {
        estimate,
        precedence,
    }
}

fn task(problem: &Problem, statistics: &Statistics, attempt: &Mutex<Attempt>) {
    let mut leaf = None;
    let mut rules = vec![];
    let mut possible = vec![];
    let mut tableau = Tableau::new(&*problem);

    while dequeue(attempt, &mut leaf, &mut rules) {
        let mut record = Silent; //crate::io::tstp::TSTP::default();
        for rule in rules.iter().rev() {
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
                    rules.reverse();
                    rules.push(rule);
                    found_proof(attempt, rules);
                    return;
                }
                let priority = heuristic(&rule, &rules, &tableau);
                enqueue(attempt, leaf, rule, priority);
                statistics.increment_enqueued_tableaux();
            } else {
                statistics.increment_discarded_tableaux();
            }
            tableau.restore();
            statistics.increment_total_tableaux();
        }
        statistics.increment_expanded_tableaux();
        let closed = finish(attempt, leaf);
        statistics.exhausted_tableaux(closed);
        tableau.clear();
    }
}

pub(crate) fn search(problem: &Problem) -> (Statistics, Option<Vec<Rule>>) {
    let statistics = Statistics::new(problem);
    let attempt = Mutex::new(Attempt::default());
    thread::scope(|scope| {
        for index in 0..num_cpus::get() {
            scope
                .builder()
                .name(format!("search-{}", index))
                .stack_size(STACK_SIZE)
                .spawn(|_| task(problem, &statistics, &attempt))
                .expect("failed to spawn search thread");
        }
    })
    .unwrap_or_else(|_| panic!("thread crashed"));
    (statistics, attempt.into_inner().proof)
}
