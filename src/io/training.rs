use crate::goal::Goal;
use crate::io::exit;
use crate::options::Options;
use crate::prelude::*;
use crate::search::SearchResult;

fn array<T: std::fmt::Display>(data: &[T]) {
    let mut data = data.iter();
    print!("[");
    if let Some(first) = data.next() {
        print!("{}", first);
    }
    for rest in data {
        print!(",{}", rest);
    }
    print!("]");
}

pub(crate) fn dump(options: &Options, problem: &Problem, route: Vec<Rule>) {
    let mut goal = Goal::new(problem);
    let mut graph = Graph::new(problem);
    let mut possible = vec![];

    for step in route {
        goal.save();
        goal.possible_rules(&mut possible);
        possible.retain(|possible| {
            goal.apply_rule(*possible);
            let constraints_ok = goal.solve_constraints();
            goal.restore();
            constraints_ok
        });
        if possible.len() > 1 {
            let y = unwrap(possible.iter().position(|rule| *rule == step));
            goal.graph(&mut graph, &possible);
            goal.restore();
            print!("{{");
            print!("\"problem\":{:?}", options.problem_name());
            print!(",\"nodes\":");
            array(graph.node_labels());
            print!(",\"sources\":");
            array(&graph.sources);
            print!(",\"targets\":");
            array(&graph.targets);
            print!(",\"rules\":");
            array(&graph.rules);
            print!(",\"y\":{}", y);
            println!("}}");
        }

        graph.clear();
        possible.clear();
        goal.apply_rule(step);
        let constraints_ok = goal.simplify_constraints();
        debug_assert!(constraints_ok);
    }
    debug_assert!(goal.is_closed());
    let constraints_ok = goal.solve_constraints();
    debug_assert!(constraints_ok);
}

pub(crate) fn output(
    options: &Options,
    problem: &Problem,
    result: SearchResult,
) -> ! {
    if let SearchResult::Unsat(core) = result {
        for route in core {
            dump(options, problem, route);
        }
        exit::success()
    } else {
        exit::failure()
    }
}
