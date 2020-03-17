use crate::prelude::*;

pub struct Subgoal {
    path: Vec<Literal>,
    clause: Clause,
}

impl Subgoal {
    pub fn start(clause: Clause) -> Self {
        let path = vec![];
        Self { path, clause }
    }
}
