use crate::priority::Priority;
use crate::rule_store::*;
use crate::statistics::Statistics;
use crossbeam_utils::thread;
use lazy::prelude::*;
use lazy::record::Silent;
use lazy::tableau::Tableau;
use lazy::util::queue::Queue;
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

        if let Some(id) = attempt.queue.dequeue() {
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

fn heuristic(rules: &[Rule], tableau: &Tableau) -> f32 {
    (tableau.num_open_branches() + (rules.len() as u16)) as f32
}

fn task(problem: &Problem, statistics: &Statistics, attempt: &Mutex<Attempt>) {
    let mut leaf = None;
    let mut rules = vec![];
    let mut possible = vec![];
    let mut tableau = Tableau::new(&*problem);
    let mut leaves = vec![];
    let mut heuristics = vec![];
    let mut residuals = vec![];
    let mut graph = Graph::default();

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
                leaves.push(add_rule(attempt, leaf, rule));
                heuristics.push(heuristic(&rules, &tableau));
                residuals.push(0.0);
                tableau.graph(&mut graph);
                graph.finish_subgraph();
            } else {
                statistics.increment_discarded_tableaux();
            }
            tableau.restore();
            statistics.increment_total_tableaux();
        }

        heuristic::model(&graph, &mut residuals);
        for (i, residual) in residuals.drain(..).enumerate() {
            heuristics[i] += residual;
        }
        for (new, heuristic) in leaves.drain(..).zip(heuristics.drain(..)) {
            enqueue(attempt, new, Priority::new(heuristic));
            statistics.increment_enqueued_tableaux();
        }

        statistics.increment_expanded_tableaux();
        let closed = finish(attempt, leaf);
        statistics.exhausted_tableaux(closed);
        tableau.clear();
        graph.clear();
    }
}

pub fn search(problem: &Problem) -> (Statistics, Option<Vec<Rule>>) {
    let statistics = Statistics::new(problem);
    let attempt = Mutex::new(Attempt::default());
    thread::scope(|scope| {
        for index in 0..2 * num_cpus::get() {
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
