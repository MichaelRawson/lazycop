use crate::goal::Goal;
use crate::prelude::*;
use crate::priority::Priority;
use crate::record::Silent;
use crate::rule_store::*;
use crate::statistics::Statistics;
use crate::util::queue::Queue;
use crossbeam_utils::thread;
use parking_lot::Mutex;

const STACK_SIZE: usize = 0x10_00000;

struct Attempt {
    rule_store: RuleStore,
    queue: Queue<Priority, Option<Id<RuleList>>>,
    proof: Option<Vec<Rule>>,
    in_flight: u16,
}

impl Default for Attempt {
    fn default() -> Self {
        let rule_store = RuleStore::default();
        let proof = None;
        let in_flight = 0;
        let mut queue = Queue::default();
        let priority = Priority::new(0.0);
        queue.enqueue(priority, None);
        Self {
            rule_store,
            queue,
            proof,
            in_flight,
        }
    }
}

fn add_rule(
    attempt: &Mutex<Attempt>,
    leaf: Option<Id<RuleList>>,
    rule: Rule,
) -> Id<RuleList> {
    let mut attempt = attempt.lock();
    attempt.rule_store.add(leaf, rule)
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

        if let Some((_priority, id)) = attempt.queue.dequeue() {
            *leaf = id;
            rules.extend(attempt.rule_store.get_list(id));
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

fn enqueue(attempt: &Mutex<Attempt>, new: Id<RuleList>, priority: Priority) {
    let mut attempt = attempt.lock();
    attempt.queue.enqueue(priority, Some(new));
}

fn finish(attempt: &Mutex<Attempt>, leaf: Option<Id<RuleList>>) -> u16 {
    let mut attempt = attempt.lock();
    attempt.in_flight -= 1;
    attempt.rule_store.mark_done(leaf)
}

fn found_proof(attempt: &Mutex<Attempt>, proof: Vec<Rule>) {
    let mut attempt = attempt.lock();
    attempt.proof = Some(proof);
}

fn heuristic(rules: &[Rule], goal: &Goal) -> f32 {
    let open_branches = goal.num_open_branches();
    let depth = rules.len() as u16;
    let heuristic = open_branches + depth;
    heuristic as f32
}

fn task(problem: &Problem, statistics: &Statistics, attempt: &Mutex<Attempt>) {
    let mut leaf = None;
    let mut rules = vec![];
    let mut possible = vec![];
    let mut goal = Goal::new(problem);
    let mut leaves = vec![];
    let mut heuristics = vec![];

    while dequeue(attempt, &mut leaf, &mut rules) {
        let mut record = Silent; //crate::io::tstp::TSTP::default();
        for rule in rules.iter().rev() {
            goal.apply_rule(&mut record, rule);
        }
        if !goal.simplify_constraints() {
            unreachable()
        }

        goal.possible_rules(&mut possible);
        goal.save();
        for rule in possible.drain(..) {
            goal.apply_rule(&mut Silent, &rule);
            if goal.solve_constraints() {
                if goal.is_closed() {
                    rules.reverse();
                    rules.push(rule);
                    found_proof(attempt, rules);
                    return;
                }
                leaves.push(add_rule(attempt, leaf, rule));
                heuristics.push(heuristic(&rules, &goal));
            } else {
                statistics.increment_discarded_goals();
            }
            goal.restore();
            statistics.increment_total_goals();
        }

        for (new, heuristic) in leaves.drain(..).zip(heuristics.drain(..)) {
            enqueue(attempt, new, Priority::new(heuristic));
            statistics.increment_enqueued_goals();
        }

        statistics.increment_expanded_goals();
        let closed = finish(attempt, leaf);
        statistics.exhausted_goals(closed);
        goal.clear();
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
