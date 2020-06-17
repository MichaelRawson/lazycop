use crate::prelude::*;
use crate::record::Silent;
use crate::tableau::Tableau;
use crate::util::queue::{Priority, Queue};
use parking_lot::Mutex;
use std::sync::Arc;
use std::thread::Builder;

const STACK_SIZE: usize = 0x10_00000;

struct Attempt {
    rules: Rules,
    queue: Queue<Id<RuleList>>,
    proof: Option<Vec<Rule>>,
    in_flight: u16,
}

impl Attempt {
    pub(crate) fn new(problem: &Problem) -> Self {
        let mut rules = Rules::default();
        let mut queue = Queue::default();
        let proof = None;
        let in_flight = 0;
        for start in problem
            .start_clauses()
            .map(|clause| Start { clause })
            .map(Rule::Start)
        {
            let id = rules.add(None, start);
            let estimate = 0;
            let precedence = start.precedence();
            let priority = Priority {
                estimate,
                precedence,
            };
            queue.enqueue(id, priority);
        }

        Self {
            rules,
            queue,
            proof,
            in_flight,
        }
    }
}

fn dequeue(attempt: &Mutex<Attempt>) -> Option<Id<RuleList>> {
    loop {
        let mut attempt = attempt.lock();
        if attempt.proof.is_some() {
            return None;
        }

        if let Some(id) = attempt.queue.dequeue() {
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

fn get_rules(
    attempt: &Mutex<Attempt>,
    rules: &mut Vec<Rule>,
    id: Id<RuleList>,
) {
    let attempt = attempt.lock();
    rules.extend(attempt.rules.get_list(id));
}

fn enqueue(
    attempt: &Mutex<Attempt>,
    leaf: Id<RuleList>,
    rule: Rule,
    priority: Priority,
) {
    let mut attempt = attempt.lock();
    let new = attempt.rules.add(Some(leaf), rule);
    attempt.queue.enqueue(new, priority);
}

fn finish(attempt: &Mutex<Attempt>, leaf: Id<RuleList>) {
    let mut attempt = attempt.lock();
    attempt.rules.mark_done(leaf);
    attempt.in_flight -= 1;
}

fn found_proof(attempt: &Mutex<Attempt>, proof: Vec<Rule>) {
    let mut attempt = attempt.lock();
    attempt.proof = Some(proof);
}

fn search(problem: Arc<Problem>, attempt: Arc<Mutex<Attempt>>) {
    let mut rules = vec![];
    let mut possible = vec![];
    let mut tableau = Tableau::new(&*problem);

    while let Some(leaf) = dequeue(&*attempt) {
        rules.clear();
        get_rules(&*attempt, &mut rules, leaf);

        let mut record = Silent; //crate::io::tstp::TSTP::default();
        for rule in rules.iter().rev() {
            tableau.apply_rule(&mut record, rule);
        }
        assert!(tableau.simplify_constraints());
        tableau.save();

        tableau.possible_rules(&mut possible);
        possible.sort();
        possible.dedup();

        for rule in possible.drain(..) {
            tableau.apply_rule(&mut Silent, &rule);
            if tableau.solve_constraints() {
                if tableau.is_closed() {
                    rules.reverse();
                    rules.push(rule);
                    found_proof(&*attempt, rules);
                    return;
                }
                let estimate =
                    tableau.num_open_branches() + (rules.len() as u16);
                let precedence = rule.precedence();
                let priority = Priority {
                    estimate,
                    precedence,
                };
                enqueue(&*attempt, leaf, rule, priority);
            }
            tableau.restore();
        }
        finish(&*attempt, leaf);
        tableau.clear();
    }
}

pub(crate) struct Search {
    problem: Arc<Problem>,
    attempt: Arc<Mutex<Attempt>>,
}

impl Search {
    pub(crate) fn new(problem: &Arc<Problem>) -> Self {
        let attempt = Arc::new(Mutex::new(Attempt::new(&*problem)));
        let problem = problem.clone();
        Self { problem, attempt }
    }

    pub(crate) fn go(self) -> Option<Vec<Rule>> {
        let mut handles = vec![];
        for index in 0..num_cpus::get() {
            let problem = self.problem.clone();
            let attempt = self.attempt.clone();
            let handle = Builder::new()
                .name(format!("search-{}", index))
                .stack_size(STACK_SIZE)
                .spawn(move || search(problem, attempt))
                .expect("failed to spawn search thread");
            handles.push(handle);
        }
        for handle in handles.drain(..) {
            handle.join().unwrap_or_else(|_| panic!("thread crashed"));
        }
        self.attempt.lock().proof.clone()
    }
}
