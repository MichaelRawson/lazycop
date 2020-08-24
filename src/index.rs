use crate::prelude::*;

pub(crate) struct PredicateOccurrence {
    pub(crate) clause: Id<ProblemClause>,
    pub(crate) literal: Id<Literal>,
}

pub(crate) struct EqualityOccurrence {
    pub(crate) clause: Id<ProblemClause>,
    pub(crate) literal: Id<Literal>,
    pub(crate) l2r: bool,
}

pub(crate) struct SubtermOccurrence {
    pub(crate) clause: Id<ProblemClause>,
    pub(crate) literal: Id<Literal>,
    pub(crate) subterm: Id<Term>,
}

#[derive(Default)]
pub(crate) struct Index {
    pub(crate) predicate_occurrences: Block<PredicateOccurrence>,
    pub(crate) equality_occurrences: Block<EqualityOccurrence>,
    pub(crate) subterm_occurrences: Block<SubtermOccurrence>,

    predicates: [LUT<Symbol, Vec<Id<PredicateOccurrence>>>; 2],
    variable_equalities: Vec<Id<EqualityOccurrence>>,
    function_equalities: LUT<Symbol, Vec<Id<EqualityOccurrence>>>,
    symbol_subterms: LUT<Symbol, Vec<Id<SubtermOccurrence>>>,
}

impl Index {
    pub(crate) fn query_predicates(
        &self,
        polarity: bool,
        symbol: Id<Symbol>,
    ) -> impl Iterator<Item = Id<PredicateOccurrence>> + '_ {
        self.predicates[polarity as usize][symbol].iter().copied()
    }

    pub(crate) fn query_variable_equalities(
        &self,
    ) -> impl Iterator<Item = Id<EqualityOccurrence>> + '_ {
        self.variable_equalities.iter().copied()
    }

    pub(crate) fn query_function_equalities(
        &self,
        symbol: Id<Symbol>,
    ) -> impl Iterator<Item = Id<EqualityOccurrence>> + '_ {
        self.function_equalities[symbol].iter().copied()
    }

    pub(crate) fn query_all_subterms(
        &self,
    ) -> impl Iterator<Item = Id<SubtermOccurrence>> + '_ {
        self.symbol_subterms
            .range()
            .into_iter()
            .flat_map(move |id| self.symbol_subterms[id].iter().copied())
    }

    pub(crate) fn query_subterms(
        &self,
        symbol: Id<Symbol>,
    ) -> impl Iterator<Item = Id<SubtermOccurrence>> + '_ {
        self.symbol_subterms[symbol].iter().copied()
    }

    pub(crate) fn set_signature(&mut self, symbols: &Symbols) {
        let max_symbol = symbols.len();
        self.predicates[0].resize(max_symbol);
        self.predicates[1].resize(max_symbol);
        self.function_equalities.resize(max_symbol);
        self.symbol_subterms.resize(max_symbol);
    }

    pub(crate) fn add_predicate_occurrence(
        &mut self,
        symbols: &Symbols,
        clause: Id<ProblemClause>,
        literal: Id<Literal>,
        polarity: bool,
        symbol: Id<Symbol>,
    ) {
        let occurrence = self
            .predicate_occurrences
            .push(PredicateOccurrence { clause, literal });
        let polarity_positions = &mut self.predicates[polarity as usize];
        polarity_positions.resize(symbols.len());
        polarity_positions[symbol].push(occurrence);
    }

    pub(crate) fn add_equality_occurrence(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        clause: Id<ProblemClause>,
        literal: Id<Literal>,
        from: Id<Term>,
        l2r: bool,
    ) {
        let occurrence = self.equality_occurrences.push(EqualityOccurrence {
            clause,
            literal,
            l2r,
        });
        match terms.view(&symbols, from) {
            TermView::Variable(_) => {
                self.variable_equalities.push(occurrence);
            }
            TermView::Function(f, _) => {
                self.function_equalities.resize(symbols.len());
                self.function_equalities[f].push(occurrence);
            }
        }
    }

    pub(crate) fn add_subterm_occurrences(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        clause: Id<ProblemClause>,
        literal: Id<Literal>,
        term: Id<Term>,
    ) {
        let subterm_occurrences = &mut self.subterm_occurrences;
        let symbol_subterms = &mut self.symbol_subterms;
        terms.subterms(symbols, term, &mut |subterm| {
            let symbol = terms.symbol(subterm);
            let occurrence = subterm_occurrences.push(SubtermOccurrence {
                clause,
                literal,
                subterm,
            });
            symbol_subterms.resize(symbols.len());
            symbol_subterms[symbol].push(occurrence);
        });
    }
}
