use crate::prelude::*;

pub struct Subgoal {
    path: Vec<Literal>,
    clause: Clause,
}

impl Subgoal {
    pub fn with_path(other: &Self, clause: Clause) -> Self {
        let path = other.path.clone();
        Self { path, clause }
    }

    pub fn start(clause: Clause) -> Self {
        let path = vec![];
        Self { path, clause }
    }

    pub fn is_done(&self) -> bool {
        self.clause.is_empty()
    }

    pub fn current_literal(&self) -> Option<&Literal> {
        self.clause.literals.last()
    }

    pub fn pop_literal(&mut self) -> Literal {
        self.clause.pop_literal()
    }
}
