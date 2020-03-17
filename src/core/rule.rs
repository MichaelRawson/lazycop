use crate::prelude::*;

#[derive(Clone, Copy)]
pub enum Rule {
    Start(Id<Clause>),
}
