use crate::prelude::*;

#[derive(Default)]
pub struct Tableau {
    subgoals: Vec<Subgoal>,
}

impl Tableau {
    pub fn is_closed(&self) -> bool {
        false
    }

    pub fn num_subgoals(&self) -> u32 {
        self.subgoals.len() as u32
    }

    pub fn reconstruct(&mut self, _script: &[Rule]) {
    }

    pub fn possible_rules(&self) -> Vec<Rule> {
        vec![]
    }

    pub fn clear(&mut self) {
        self.subgoals.clear();
    }
}
