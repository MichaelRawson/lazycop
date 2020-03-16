use crate::prelude::*;
use std::fmt;

struct Context<'problem, 'value, T>(&'problem Problem, &'value T);

impl fmt::Display for Context<'_, '_, Id<Symbol>> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Context(problem, symbol_id) = self;
        write!(f, "{}", problem.symbol_list.name(**symbol_id))
    }
}
