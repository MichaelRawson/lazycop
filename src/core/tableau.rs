use crate::output::proof::Record;
use crate::prelude::*;

#[derive(Default)]
pub struct Tableau {
    term_list: TermList,
    subgoals: Vec<Subgoal>,
}

impl Tableau {
    pub fn is_closed(&self) -> bool {
        self.subgoals.is_empty()
    }

    pub fn num_subgoals(&self) -> u32 {
        self.subgoals.len() as u32
    }

    pub fn reconstruct<R: Record>(
        &mut self,
        _record: &mut R,
        _problem: &Problem,
        _script: &[Rule],
    ) {
    }

    pub fn possible_rules(&self) -> Vec<Rule> {
        vec![]
    }

    pub fn clear(&mut self) {
        self.subgoals.clear();
    }
}
