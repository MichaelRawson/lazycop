use crate::goal::Goal;
use crate::options::Options;
use crate::prelude::*;
use crate::record::Silent;
use crate::tree::Tree;

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

pub(crate) fn dump(problem: &Problem, tree: &Tree, options: &Options) {
    let mut scores: Vec<f32> = vec![];
    let mut goal = Goal::new(problem);
    let mut graph = Graph::default();
    let mut rules = vec![];

    for id in tree.eligible_training_nodes(options) {
        rules.extend(tree.backward_derivation(id));
        for rule in rules.drain(..).rev() {
            goal.apply_rule(&mut Silent, &rule);
        }
        let constraints_ok = goal.simplify_constraints();
        debug_assert!(constraints_ok);
        goal.save();

        let mut empty_goal = false;
        for rule in tree.child_rules(id) {
            goal.apply_rule(&mut Silent, &rule);
            empty_goal |= goal.is_closed();
            let constraints_ok = goal.solve_constraints();
            debug_assert!(constraints_ok);
            goal.graph(&mut graph);
            graph.finish_subgraph();
            goal.restore();
        }

        if !empty_goal && scores.len() > 1 {
            print!("{{");
            print!("\"nodes\":");
            array(graph.node_labels());
            print!(",\"sources\":");
            array(&graph.sources);
            print!(",\"targets\":");
            array(&graph.targets);
            print!(",\"batch\":");
            array(&graph.batch);
            print!(",\"scores\":");
            array(&scores);
            println!("}}");
        }

        goal.clear();
        graph.clear();
        scores.clear();
    }
}
