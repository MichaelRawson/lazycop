//pub mod fingerprint;
pub mod top_symbol;
use crate::prelude::*;

pub type PredicateIndex =
    [top_symbol::Index<Vec<(Id<Clause>, Id<Literal>)>>; 2];
