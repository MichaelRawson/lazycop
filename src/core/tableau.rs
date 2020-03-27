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

    pub fn fill_possible_rules(&self, possible: &mut Vec<Rule>) {
        let subgoal = self.subgoals.last().unwrap();
        possible.clear();
        subgoal.possible_rules(possible, &self.problem, &self.term_graph)
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
        match rule {
            Rule::Start(clause_id) => {
                self.apply_start::<P, _>(record, clause_id);
            }
            Rule::Extension(coordinate) => {
                self.apply_extension::<P, _>(record, coordinate);
                self.check_regularity::<P>();
            }
            Rule::Reduction(path_id) => {
                self.apply_reduction::<P, _>(record, path_id);
                self.check_regularity::<P>();
            }
            Rule::Lemma(lemma_id) => {
                self.apply_lemma::<P, _>(record, lemma_id);
                self.check_regularity::<P>();
            }
            Rule::Symmetry => {
                self.apply_symmetry::<P, _>(record);
                self.check_regularity::<P>();
            }
        }
    }

    fn check_regularity<P: Policy>(&mut self) {
        if P::should_check_regularity() {
            if let Some(subgoal) = self.subgoals.last() {
                if !subgoal
                    .is_regular(&self.problem.symbol_table, &self.term_graph)
                {
                    self.blocked = true;
                }
            }
        }
    }

    fn apply_start<P: Policy, R: Record>(
        &mut self,
        record: &mut R,
        clause_id: Id<Clause>,
    ) {
        let start_goal = Subgoal::start(
            record,
            &mut self.term_graph,
            self.problem,
            clause_id,
        );
        self.subgoals.push(start_goal);
    }

    fn apply_extension<P: Policy, R: Record>(
        &mut self,
        record: &mut R,
        coordinate: Id<(Clause, Literal)>,
    ) {
        let mut subgoal = self.subgoals.pop().unwrap();
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

    fn apply_reduction<P: Policy, R: Record>(
        &mut self,
        record: &mut R,
        path_id: Id<Literal>,
    ) {
        let mut subgoal = self.subgoals.pop().unwrap();
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

    fn apply_lemma<P: Policy, R: Record>(
        &mut self,
        record: &mut R,
        lemma_id: Id<Literal>,
    ) {
        let mut subgoal = self.subgoals.pop().unwrap();
        subgoal.apply_lemma::<P, _>(
            record,
            &mut self.term_graph,
            self.problem,
            lemma_id,
        );
        if !subgoal.is_done() {
            self.subgoals.push(subgoal);
        }
    }

    fn apply_symmetry<P: Policy, R: Record>(&mut self, record: &mut R) {
        let mut subgoal = self.subgoals.pop().unwrap();
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
