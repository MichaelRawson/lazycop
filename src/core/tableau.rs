use crate::output::record::Record;
use crate::prelude::*;

pub struct Tableau<'problem> {
    problem: &'problem Problem,
    blocked: bool,
    term_graph: TermGraph,
    subgoals: Vec<Subgoal>,
}

impl<'problem> Tableau<'problem> {
    pub fn new(problem: &'problem Problem) -> Self {
        let blocked = false;
        let term_graph = TermGraph::default();
        let subgoals = vec![];
        Self {
            problem,
            blocked,
            term_graph,
            subgoals,
        }
    }

    pub fn duplicate(&mut self, other: &Self) {
        self.blocked = other.blocked;
        self.term_graph.clear();
        self.term_graph.copy_from(&other.term_graph);
        self.subgoals.clear();
        self.subgoals.extend(other.subgoals.iter().cloned());
    }

    pub fn is_blocked(&self) -> bool {
        self.blocked
    }

    pub fn is_closed(&self) -> bool {
        self.subgoals.is_empty()
    }

    pub fn num_literals(&self) -> usize {
        self.subgoals
            .iter()
            .map(|subgoal| subgoal.num_literals())
            .sum()
    }

    pub fn reconstruct<R: Record>(&mut self, record: &mut R, script: &[Rule]) {
        self.blocked = false;
        self.term_graph.clear();
        self.subgoals.clear();
        for rule in script {
            self.apply_rule::<Unchecked, _>(record, *rule);
        }
    }

    pub fn apply_rule<P: Policy, R: Record>(
        &mut self,
        record: &mut R,
        rule: Rule,
    ) {
        assert!(!self.blocked);
        match rule {
            Rule::Start(clause_id) => {
                assert!(self.subgoals.is_empty());
                assert!(self.term_graph.is_empty());

                let start_goal = Subgoal::start(
                    record,
                    &mut self.term_graph,
                    self.problem,
                    clause_id,
                );
                self.subgoals.push(start_goal);
            }
            Rule::Extension(coordinate) => {
                assert!(!self.subgoals.is_empty());
                let mut subgoal = self.subgoals.pop().unwrap();
                assert!(!subgoal.is_done());

                if let Some((extension_goal, eq_goal)) = subgoal
                    .apply_extension::<P, _>(
                        record,
                        &mut self.term_graph,
                        self.problem,
                        coordinate,
                    )
                {
                    if !subgoal.is_done() {
                        self.subgoals.push(subgoal);
                    }
                    if !extension_goal.is_done() {
                        self.subgoals.push(extension_goal);
                    }
                    if !eq_goal.is_done() {
                        self.subgoals.push(eq_goal);
                    }
                } else {
                    self.blocked = true;
                }
            }
            Rule::Reduction(path_id) => {
                assert!(!self.subgoals.is_empty());
                let mut subgoal = self.subgoals.pop().unwrap();
                assert!(!subgoal.is_done());

                if subgoal.apply_reduction::<P, _>(
                    record,
                    &mut self.term_graph,
                    self.problem,
                    path_id,
                ) {
                    if !subgoal.is_done() {
                        self.subgoals.push(subgoal);
                    }
                } else {
                    self.blocked = true;
                }
            }
            Rule::Lemma(lemma_id) => {
                assert!(!self.subgoals.is_empty());
                let mut subgoal = self.subgoals.pop().unwrap();
                assert!(!subgoal.is_done());

                subgoal.apply_lemma::<P, _>(
                    record,
                    &mut self.term_graph,
                    self.problem,
                    lemma_id,
                );
            }
            Rule::Symmetry => {
                assert!(!self.subgoals.is_empty());
                let mut subgoal = self.subgoals.pop().unwrap();
                assert!(!subgoal.is_done());

                if subgoal.apply_symmetry::<P, _>(
                    record,
                    &mut self.term_graph,
                    self.problem,
                ) {
                    if !subgoal.is_done() {
                        self.subgoals.push(subgoal);
                    }
                } else {
                    self.blocked = true;
                }
            }
        }
        if !P::should_check() {
            return;
        }
        if let Some(subgoal) = self.subgoals.last() {
            if !subgoal
                .is_regular(&self.problem.symbol_table, &self.term_graph)
            {
                self.blocked = true;
            }
        }
    }

    pub fn fill_possible_rules(&self, possible: &mut Vec<Rule>) {
        assert!(!self.blocked);
        assert!(!self.subgoals.is_empty());
        let subgoal = self.subgoals.last().unwrap();
        assert!(!subgoal.is_done());
        possible.clear();
        subgoal.possible_rules(possible, &self.problem, &self.term_graph)
    }
}
