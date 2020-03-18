use crate::prelude::*;

pub struct Subgoal {
    pub path: Vec<Literal>,
    pub clause: Clause,
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

    pub fn push_path(&mut self, literal: Literal) {
        self.path.push(literal);
    }

    pub fn extend_clause<T: Iterator<Item = Literal>>(&mut self, t: T) {
        self.clause.extend(t)
    }

    pub fn is_done(&self) -> bool {
        self.clause.is_empty()
    }

    pub fn current_literal(&self) -> Option<&Literal> {
        self.clause.last_literal()
    }

    pub fn pop_literal(&mut self) -> Option<Literal> {
        self.clause.pop_literal()
    }
}
