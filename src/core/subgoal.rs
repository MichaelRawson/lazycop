use crate::prelude::*;

pub struct Subgoal {
    path: Vec<Literal>,
    clause: Clause,
}
