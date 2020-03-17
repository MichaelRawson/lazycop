use crate::prelude::*;
use std::fmt;

struct Context<'context, 'value, C, T>(&'context C, &'value T);

impl fmt::Display for Context<'_, '_, SymbolList, Id<Symbol>> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Context(symbol_list, symbol_id) = self;
        write!(f, "{}", symbol_list.name(**symbol_id))
    }
}
