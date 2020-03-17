use crate::output::record::Record;
use crate::prelude::*;

#[derive(Default)]
pub struct Tableau {
    term_list: TermList,
    subgoals: Vec<Subgoal>,
}

impl Tableau {
    pub fn is_closed(&self) -> bool {
        true
        //self.subgoals.is_empty()
    }

    pub fn num_subgoals(&self) -> u32 {
        self.subgoals.len() as u32
    }

    pub fn clear(&mut self) {
        self.term_list.clear();
        self.subgoals.clear();
    }

    pub fn reconstruct<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        script: &[Rule],
    ) {
        for rule in script {
            match rule {
                Rule::Start(clause_id) => {
                    self.start(record, problem, *clause_id);
                }
            }
        }
    }

    pub fn possible_rules(&self) -> Vec<Rule> {
        vec![]
    }

    fn start<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        clause_id: Id<Clause>,
    ) {
        assert!(self.subgoals.is_empty());
        assert!(self.term_list.is_empty());
        let (clause, clause_term_list) = problem.get_clause(clause_id);
        self.term_list.copy_from(clause_term_list);
        record.start(&problem.symbol_list, &self.term_list, &clause);
        self.subgoals.push(Subgoal::start(clause));
    }
}
