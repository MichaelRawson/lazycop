use crate::prelude::*;
use fnv::FnvHashMap;
use std::mem;
use std::ops::Index;

pub(crate) struct ProblemClause {
    pub(crate) literals: Block<Literal>,
    pub(crate) terms: Terms,
}

type Position = (Id<ProblemClause>, Id<Literal>);

#[derive(Default)]
pub(crate) struct Problem {
    clauses: Block<ProblemClause>,
    pub(crate) start_clauses: Vec<Id<ProblemClause>>,
    pub(crate) predicate_occurrences:
        [FnvHashMap<Id<Symbol>, Vec<Position>>; 2],
    pub(crate) symbols: Symbols,
    pub(crate) equality: bool,
}

impl Index<Id<ProblemClause>> for Problem {
    type Output = ProblemClause;

    fn index(&self, id: Id<ProblemClause>) -> &Self::Output {
        &self.clauses[id]
    }
}

type FunctionKey = (Id<Symbol>, Vec<Id<Term>>);
#[derive(Default)]
pub(crate) struct ProblemBuilder {
    problem: Problem,
    symbols: FnvHashMap<(String, u32), Id<Symbol>>,
    variable_map: FnvHashMap<String, Id<Term>>,
    function_map: FnvHashMap<FunctionKey, Id<Term>>,
    saved_terms: Vec<Id<Term>>,
    saved_literals: Block<Literal>,
    terms: Terms,
}

impl ProblemBuilder {
    pub(crate) fn finish(self) -> Problem {
        self.problem
    }

    pub(crate) fn variable(&mut self, variable: String) {
        let terms = &mut self.terms;
        let id = *self
            .variable_map
            .entry(variable)
            .or_insert_with(|| terms.add_variable());
        self.saved_terms.push(id);
    }

    pub(crate) fn function(&mut self, symbol: String, arity: u32) {
        let symbols = &mut self.problem.symbols;
        let symbol = *self
            .symbols
            .entry((symbol.clone(), arity))
            .or_insert_with(|| symbols.append(symbol));

        let args = self
            .saved_terms
            .split_off(self.saved_terms.len() - (arity as usize));
        let terms = &mut self.terms;
        let id = *self
            .function_map
            .entry((symbol, args.clone()))
            .or_insert_with(|| terms.add_function(symbol, &args));
        self.saved_terms.push(id);
    }

    pub(crate) fn predicate(&mut self, polarity: bool) {
        let term = self.saved_terms.pop().expect("predicate without a term");
        let atom = Atom::Predicate(term);
        let symbol = match self.terms.view(term) {
            (_, TermView::Function(symbol, _)) => symbol,
            _ => unreachable!(),
        };

        let clause = self.problem.clauses.len();
        let literal = self.saved_literals.len();
        self.problem.predicate_occurrences[polarity as usize]
            .entry(symbol)
            .or_default()
            .push((clause, literal));
        self.saved_literals.push(Literal::new(polarity, atom));
    }

    pub(crate) fn equality(&mut self, polarity: bool) {
        self.problem.equality = true;
        let right = self.saved_terms.pop().expect("equality without term");
        let left = self.saved_terms.pop().expect("equality without term");
        let atom = Atom::Equality(left, right);
        self.saved_literals.push(Literal::new(polarity, atom));
    }

    pub(crate) fn clause(&mut self, start_clause: bool) {
        let terms = mem::take(&mut self.terms);
        self.variable_map.clear();
        self.function_map.clear();
        let literals = mem::take(&mut self.saved_literals);
        let problem_clause = ProblemClause { literals, terms };
        let id = self.problem.clauses.push(problem_clause);
        if start_clause {
            self.problem.start_clauses.push(id);
        }
    }
}
