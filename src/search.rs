use crate::goal::Goal;
use crate::prelude::*;
use crate::record::Silent;
use crate::rule::Start;
use crate::rule_store::*;
use crate::statistics::Statistics;
use crate::util::queue::Queue;
use crossbeam_utils::thread;
use parking_lot::Mutex;

const STACK_SIZE: usize = 0x10_00000;

#[derive(Default)]
struct Attempt {
    rule_store: RuleStore,
    queue: Queue<u16, Id<RuleList>>,
    proof: Option<Vec<Rule>>,
    in_flight: u16,
}

fn dequeue(
    attempt: &Mutex<Attempt>,
    rules: &mut Vec<Rule>,
) -> Option<Id<RuleList>> {
    rules.clear();
    loop {
        let mut attempt = attempt.lock();
        if attempt.proof.is_some() {
            return None;
        }

        if let Some((_priority, id)) = attempt.queue.dequeue() {
            rules.extend(attempt.rule_store.get_list(id));
            attempt.in_flight += 1;
            return Some(id);
        } else if attempt.in_flight == 0 {
            return None;
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
    priority: u16,
) {
    let mut attempt = attempt.lock();
    let new = attempt.rule_store.add(leaf, rule);
    attempt.queue.enqueue(priority, new);
}

fn finish(attempt: &Mutex<Attempt>, leaf: Id<RuleList>) -> u16 {
    let mut attempt = attempt.lock();
    attempt.in_flight -= 1;
    attempt.rule_store.mark_done(leaf)
}

fn found_proof(attempt: &Mutex<Attempt>, proof: Vec<Rule>) {
    let mut attempt = attempt.lock();
    attempt.proof = Some(proof);
}

fn heuristic(rules: &[Rule], goal: &Goal) -> u16 {
    let open_branches = goal.num_open_branches();
    let depth = rules.len() as u16;
    open_branches + depth
}

fn task(problem: &Problem, statistics: &Statistics, attempt: &Mutex<Attempt>) {
    let mut rules = vec![];
    let mut possible = vec![];
    let mut goal = Goal::new(problem);

    while let Some(leaf) = dequeue(attempt, &mut rules) {
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
                let heuristic = heuristic(&rules, &goal);
                enqueue(attempt, Some(leaf), rule, heuristic);
                statistics.increment_enqueued_goals();
            } else {
                statistics.increment_discarded_goals();
            }
            goal.restore();
            statistics.increment_total_goals();
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
    for start in problem
        .start_clauses
        .iter()
        .copied()
        .map(|clause| Start { clause })
        .map(Rule::Start)
    {
        enqueue(&attempt, None, start, 0);
    }

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
