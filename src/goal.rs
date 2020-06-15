use crate::clause::Clause;
use crate::constraint::Constraints;
use crate::prelude::*;
use crate::record::Record;
use std::iter::once;

#[derive(Default)]
pub(crate) struct Goal {
    literals: Literals,
    stack: Block<Clause>,
    valid: Block<Id<Clause>>,
    lemmata: Block<Vec<Id<Literal>>>,

    save_literals: Id<Literal>,
    save_stack: Block<Clause>,
    save_valid: Block<Id<Clause>>,
    save_lemmata: Block<Vec<Id<Literal>>>,
}

impl Goal {
    pub(crate) fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.literals.clear();
        self.stack.clear();
        self.valid.clear();
        self.lemmata.clear();
    }

    pub(crate) fn save(&mut self) {
        self.save_literals = self.literals.len();
        self.save_stack.copy_from(&self.stack);
        self.save_valid.copy_from(&self.valid);
        self.save_lemmata.copy_from(&self.lemmata);
    }

    pub(crate) fn restore(&mut self) {
        self.literals.truncate(self.save_literals);
        self.stack.copy_from(&self.save_stack);
        self.valid.copy_from(&self.save_valid);
        self.lemmata.copy_from(&self.save_lemmata);
    }

    pub(crate) fn num_open_branches(&self) -> u16 {
        self.stack
            .range()
            .map(|id| Range::len(self.stack[id].remaining()) as u16)
            .sum::<u16>()
            + 1
    }

    pub(crate) fn apply_rule<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        constraints: &mut Constraints,
        rule: &Rule,
    ) {
        match *rule {
            Rule::Start(start) => {
                let start = Clause::start(
                    record,
                    problem,
                    terms,
                    &mut self.literals,
                    constraints,
                    start,
                );
                self.push(start);
            }
            Rule::Reflexivity => {
                some(self.stack.last_mut()).reflexivity(
                    record,
                    &problem.signature(),
                    terms,
                    &self.literals,
                    constraints,
                );
                self.close_branches();
            }
            Rule::PredicateReduction(reduction) => {
                some(self.stack.last_mut()).predicate_reduction(
                    record,
                    &problem.signature(),
                    terms,
                    &mut self.literals,
                    constraints,
                    reduction,
                );
                self.reduction_validity(reduction.literal);
                self.close_branches();
            }
            Rule::LREqualityReduction(reduction)
            | Rule::RLEqualityReduction(reduction) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                let consequence = some(self.stack.last_mut())
                    .equality_reduction(
                        record,
                        &problem.signature(),
                        terms,
                        &mut self.literals,
                        constraints,
                        reduction,
                        rule.lr(),
                    );
                self.push(consequence);
                self.reduction_validity(reduction.literal);
            }
            Rule::LRSubtermReduction(reduction)
            | Rule::RLSubtermReduction(reduction) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                let consequence = some(self.stack.last_mut())
                    .subterm_reduction(
                        record,
                        &problem.signature(),
                        terms,
                        &mut self.literals,
                        constraints,
                        reduction,
                        rule.lr(),
                    );
                self.push(consequence);
                self.reduction_validity(reduction.literal);
            }
            Rule::LazyPredicateExtension(extension) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                let (extension, consequence) = some(self.stack.last_mut())
                    .lazy_predicate_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        constraints,
                        extension,
                    );
                self.push(extension);
                self.extension_validity();
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                self.push(consequence);
                self.close_branches();
            }
            Rule::StrictPredicateExtension(extension) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                let extension = some(self.stack.last_mut())
                    .strict_predicate_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        constraints,
                        extension,
                    );
                self.push(extension);
                self.close_branches();
            }
            Rule::StrictFunctionExtension(extension) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                let (extension, consequence) = some(self.stack.last_mut())
                    .strict_function_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        constraints,
                        extension,
                    );
                self.push(extension);
                self.extension_validity();
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                self.push(consequence);
            }
            Rule::LazyFunctionExtension(extension) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                let (extension, consequence) = some(self.stack.last_mut())
                    .lazy_function_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        constraints,
                        extension,
                    );
                self.push(extension);
                self.extension_validity();
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                self.push(consequence);
            }
            Rule::VariableExtension(extension) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                let (extension, consequence) = some(self.stack.last_mut())
                    .variable_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        constraints,
                        extension,
                    );
                self.push(extension);
                self.extension_validity();
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                self.push(consequence);
            }
            Rule::LRStrictSubtermExtension(extension)
            | Rule::RLStrictSubtermExtension(extension) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                let (extension, consequence) = some(self.stack.last_mut())
                    .strict_subterm_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        constraints,
                        extension,
                        rule.lr(),
                    );
                self.push(extension);
                self.extension_validity();
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                self.push(consequence);
            }
            Rule::LRLazySubtermExtension(extension)
            | Rule::RLLazySubtermExtension(extension) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                let (extension, consequence) = some(self.stack.last_mut())
                    .lazy_subterm_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        constraints,
                        extension,
                        rule.lr(),
                    );
                self.push(extension);
                self.extension_validity();
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                self.push(consequence);
            }
        }
    }

    fn push(&mut self, clause: Clause) {
        let id = self.stack.push(clause);
        if self.stack.len().transmute() > self.lemmata.len() {
            self.lemmata.resize(self.stack.len().transmute());
        }
        self.lemmata[id.transmute()].clear();
    }

    fn extension_validity(&mut self) {
        self.valid.resize(self.literals.len().transmute());
        let valid_in = self.stack.len() + Offset::new(-1);
        let literal = some(self.stack.last()).current_literal();
        self.valid[literal.transmute()] = valid_in;
    }

    fn reduction_validity(&mut self, reduction: Id<Literal>) {
        self.valid.resize(self.literals.len().transmute());
        let valid_in = self
            .stack
            .range()
            .find(|id| self.stack[*id].current_literal() == reduction)
            .map(|id| id + Offset::new(1))
            .unwrap_or(self.valid[reduction.transmute()]);

        for affected in Range::new(valid_in, self.stack.len()).rev().skip(1) {
            let literal = self.stack[affected.transmute()].current_literal();
            let index = literal.transmute();
            let existing = self.valid[index];
            self.valid[index] = std::cmp::max(existing, valid_in);
        }
    }

    fn close_branches(&mut self) {
        self.valid.resize(self.literals.len().transmute());
        let last = some(self.stack.last());
        if !last.is_empty() {
            return;
        }
        self.stack.pop();
        while let Some(parent) = self.stack.last_mut() {
            let id = parent.close_literal();
            let valid_in = self.valid[id.transmute()];
            let mut lemma = self.literals[id];
            lemma.polarity = !lemma.polarity;
            let lemma = self.literals.push(lemma);
            self.valid.push(valid_in);
            self.lemmata[valid_in.transmute()].push(lemma);

            if !parent.is_empty() {
                return;
            }
            self.stack.pop();
        }
    }

    fn add_regularity_constraints(
        &self,
        constraints: &mut Constraints,
        terms: &Terms,
        literals: &Literals,
    ) {
        let current =
            &self.literals[some(self.stack.last()).current_literal()];

        if current.is_predicate() {
            for path in self
                .path_literals()
                .map(|id| &literals[id])
                .filter(|path| path.polarity == current.polarity)
            {
                current.add_disequation_constraints(constraints, terms, &path);
            }

            for reduction in self
                .reduction_literals()
                .map(|id| &literals[id])
                .filter(|reduction| reduction.polarity != current.polarity)
            {
                current.add_disequation_constraints(
                    constraints,
                    terms,
                    &reduction,
                );
            }
        }
        else if current.is_equality() && current.polarity {
            for reduction in self
                .reduction_literals()
                .map(|id| &literals[id])
                .filter(|reduction| reduction.polarity)
            {
                current.add_disequation_constraints(
                    constraints,
                    terms,
                    &reduction,
                );
            }
        }
        else if current.is_equality() && !current.polarity {
            current.add_reflexivity_constraints(constraints);
        }
    }

    pub(crate) fn possible_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
        terms: &Terms,
    ) {
        let clause = some(self.stack.last());
        let literal = &self.literals[clause.current_literal()];
        if literal.is_predicate() {
            self.possible_predicate_rules(possible, problem, terms, literal);
        } else if literal.is_equality() {
            self.possible_equality_rules(possible, problem, terms, literal);
        }
        self.possible_equality_reduction_rules(
            possible,
            problem.signature(),
            terms,
            literal,
        );
        self.possible_equality_extension_rules(
            possible, problem, terms, literal,
        );
    }

    fn possible_predicate_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
        terms: &Terms,
        literal: &Literal,
    ) {
        self.possible_predicate_reduction_rules(possible, terms, literal);
        self.possible_predicate_extension_rules(
            possible, problem, terms, literal,
        );
    }

    fn possible_predicate_reduction_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        terms: &Terms,
        literal: &Literal,
    ) {
        let polarity = literal.polarity;
        let symbol = literal.get_predicate_symbol(terms);
        possible.extend(
            self.reduction_literals()
                .filter(|id| {
                    let reduction = &self.literals[*id];
                    reduction.polarity != polarity
                        && reduction.is_predicate()
                        && reduction.get_predicate_symbol(terms) == symbol
                })
                .map(|literal| PredicateReduction { literal })
                .map(Rule::PredicateReduction),
        );
    }

    fn possible_predicate_extension_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
        terms: &Terms,
        literal: &Literal,
    ) {
        let polarity = !literal.polarity;
        let symbol = literal.get_predicate_symbol(terms);
        let extensions = problem
            .query_predicates(polarity, symbol)
            .map(|occurrence| PredicateExtension { occurrence });

        for extension in extensions {
            if problem.has_equality() {
                possible.extend(once(Rule::LazyPredicateExtension(extension)));
            }
            possible.extend(once(Rule::StrictPredicateExtension(extension)));
        }
    }

    fn possible_equality_reduction_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        symbols: &Symbols,
        terms: &Terms,
        literal: &Literal,
    ) {
        for id in self.reduction_literals() {
            let reduction = &self.literals[id];
            if !reduction.polarity || !reduction.is_equality() {
                continue;
            }
            let (left, right) = reduction.get_equality();
            literal.subterms(symbols, terms, &mut |target| {
                possible.extend(
                    Self::possible_equality_reduction(terms, id, target, left)
                        .map(Rule::LREqualityReduction),
                );
                possible.extend(
                    Self::possible_equality_reduction(
                        terms, id, target, right,
                    )
                    .map(Rule::RLEqualityReduction),
                );
            });
        }
    }

    fn possible_equality_extension_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
        terms: &Terms,
        literal: &Literal,
    ) {
        literal.subterms(problem.signature(), terms, &mut |target| {
            possible.extend(
                problem
                    .query_variable_equalities()
                    .map(|occurrence| EqualityExtension { target, occurrence })
                    .map(Rule::VariableExtension),
            );

            let symbol = terms.symbol(target);
            let function_extensions = problem
                .query_function_equalities(symbol)
                .map(|occurrence| EqualityExtension { target, occurrence });
            for extension in function_extensions {
                possible.extend(once(Rule::LazyFunctionExtension(extension)));
                possible
                    .extend(once(Rule::StrictFunctionExtension(extension)));
            }
        });
    }

    fn possible_equality_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
        terms: &Terms,
        literal: &Literal,
    ) {
        if !literal.polarity {
            possible.extend(once(Rule::Reflexivity));
        } else {
            self.possible_subterm_reductions(
                possible,
                problem.signature(),
                terms,
                literal,
            );
            self.possible_subterm_extensions(
                possible, problem, terms, literal,
            );
        }
    }

    fn possible_subterm_reductions<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        symbols: &Symbols,
        terms: &Terms,
        literal: &Literal,
    ) {
        let (left, right) = literal.get_equality();
        for id in self.reduction_literals() {
            let reduction = &self.literals[id];
            reduction.subterms(symbols, terms, &mut |target| {
                possible.extend(
                    Self::possible_equality_reduction(terms, id, target, left)
                        .map(Rule::LRSubtermReduction),
                );
                possible.extend(
                    Self::possible_equality_reduction(
                        terms, id, target, right,
                    )
                    .map(Rule::RLSubtermReduction),
                );
            });
        }
    }

    fn possible_subterm_extensions<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
        terms: &Terms,
        literal: &Literal,
    ) {
        let (left, right) = literal.get_equality();
        Self::possible_subterm_extensions_one_sided(
            possible, problem, terms, left, true,
        );
        Self::possible_subterm_extensions_one_sided(
            possible, problem, terms, right, false,
        );
    }

    fn possible_subterm_extensions_one_sided<E: Extend<Rule>>(
        possible: &mut E,
        problem: &Problem,
        terms: &Terms,
        from: Id<Term>,
        lr: bool,
    ) {
        if terms.is_variable(from) {
            for occurrence in problem.query_all_subterms() {
                Self::possible_subterm_extensions_single(
                    possible, occurrence, lr,
                );
            }
        } else {
            for occurrence in problem.query_subterms(terms.symbol(from)) {
                Self::possible_subterm_extensions_single(
                    possible, occurrence, lr,
                );
            }
        }
    }

    fn possible_subterm_extensions_single<E: Extend<Rule>>(
        possible: &mut E,
        occurrence: Id<SubtermOccurrence>,
        lr: bool,
    ) {
        let extension = SubtermExtension { occurrence };
        if lr {
            possible.extend(once(Rule::LRStrictSubtermExtension(extension)));
            possible.extend(once(Rule::LRLazySubtermExtension(extension)));
        } else {
            possible.extend(once(Rule::RLStrictSubtermExtension(extension)));
            possible.extend(once(Rule::RLLazySubtermExtension(extension)));
        }
    }

    fn possible_equality_reduction(
        terms: &Terms,
        literal: Id<Literal>,
        target: Id<Term>,
        from: Id<Term>,
    ) -> Option<EqualityReduction> {
        if terms.is_variable(from)
            || terms.symbol(from) == terms.symbol(target)
        {
            Some(EqualityReduction { literal, target })
        } else {
            None
        }
    }

    fn reduction_literals(&self) -> impl Iterator<Item = Id<Literal>> + '_ {
        self.path_literals().chain(
            self.stack.range().flat_map(move |id| {
                self.lemmata[id.transmute()].iter().copied()
            }),
        )
    }

    fn path_literals(&self) -> impl Iterator<Item = Id<Literal>> + '_ {
        self.stack
            .range()
            .rev()
            .skip(1)
            .map(move |id| self.stack[id].current_literal())
    }
}
