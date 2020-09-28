use crate::goal::Goal;
use crate::prelude::*;
use crate::record::Silent;

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

pub(crate) fn dump(name: &str, problem: &Problem, proof: &[Rule]) {
    let mut goal = Goal::new(problem);
    let mut graph = Graph::default();
    let mut possible = vec![];

    for step in proof.iter() {
        goal.save();
        goal.possible_rules(&mut possible);
        possible.retain(|possible| {
            goal.apply_rule(&mut Silent, &possible);
            let constraints_ok = goal.solve_constraints();
            goal.restore();
            constraints_ok
        });
        if possible.len() > 1 {
            let y = some(possible.iter().position(|rule| rule == step));
            goal.graph(&mut graph, &possible);
            goal.restore();
            print!("{{");
            print!("\"problem\":{:?}", name);
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
        goal.apply_rule(&mut Silent, step);
        let constraints_ok = goal.simplify_constraints();
        debug_assert!(constraints_ok);
    }
    debug_assert!(goal.is_closed());
    let constraints_ok = goal.solve_constraints();
    debug_assert!(constraints_ok);
}
