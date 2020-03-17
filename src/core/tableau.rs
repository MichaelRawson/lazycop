use crate::prelude::*;

#[derive(Default)]
pub struct Tableau {
    subgoals: Vec<Subgoal>
}

impl Tableau {
    pub fn clear(&mut self) {
        self.subgoals.clear();
    }
}
