use crate::goal::Goal;
use crate::prelude::*;
use crate::record::Silent;
use crate::uctree::UCTree;

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

pub(crate) fn dump(problem: &Problem, tree: &UCTree, threshold: u32) {
    let mut rules = vec![];
    let mut scores = vec![];
    let mut goal = Goal::new(problem);
    let mut graph = Graph::default();

    for id in tree.eligible_training_nodes(threshold) {
        tree.rules_for_node(id, &mut rules);
        for rule in &rules {
            goal.apply_rule(&mut Silent, rule);
        }
        let constraints_ok = goal.simplify_constraints();
        debug_assert!(constraints_ok);
        goal.save();

        for (rule, score) in tree.child_rule_scores(id) {
            scores.push(score);
            goal.apply_rule(&mut Silent, &rule);
            let constraints_ok = goal.solve_constraints();
            debug_assert!(constraints_ok);
            debug_assert!(!goal.is_closed());
            goal.graph(&mut graph);
            graph.finish_subgraph();
            goal.restore();
        }

        if scores.len() > 1 {
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
