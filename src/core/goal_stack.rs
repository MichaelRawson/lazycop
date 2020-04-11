use crate::core::goal::Goal;
use crate::io::record::Record;
use crate::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct Lemma {
    valid_from: Id<Goal>,
    literal: Literal,
}

#[derive(Default)]
pub(crate) struct GoalStack {
    stack: Arena<Goal>,
    lemmata: Arena<Lemma>,
}

impl GoalStack {
    pub(crate) fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.stack.clear();
        self.lemmata.clear();
    }

    pub(crate) fn reset_to(&mut self, other: &GoalStack) {
        self.clear();
        self.stack.extend_from(&other.stack);
        self.lemmata.extend_from(&other.lemmata);
    }

    pub(crate) fn open_branches(&self) -> u32 {
        self.stack
            .into_iter()
            .map(|id| self.stack[id].num_open_branches())
            .sum()
    }

    pub(crate) fn apply_rule<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        term_graph: &mut TermGraph,
        clause_storage: &mut ClauseStorage,
        constraint_list: &mut ConstraintList,
        rule: Rule,
    ) {
        match rule {
            Rule::Start(start) => {
                let start = Goal::start(
                    record,
                    problem,
                    term_graph,
                    clause_storage,
                    start.start_clause,
                );
                self.stack.push(start);
            }
            Rule::Reduction(reduction) => {
                let literal =
                    self.stack[reduction.goal].current_literal(clause_storage);
                let mut valid_from = reduction.goal;
                valid_from.increment();
                let goal =
                    self.stack.last_mut().expect("reduction on empty tableau");
                goal.reduction(
                    record,
                    &problem.symbol_table,
                    term_graph,
                    clause_storage,
                    constraint_list,
                    literal,
                    valid_from,
                );
                self.close_branches(clause_storage);
            }
            Rule::Lemma(lemma) => {
                let lemma = self.lemmata[lemma.lemma];
                let goal = self
                    .stack
                    .last_mut()
                    .expect("lemma applies on empty tableau");
                goal.lemma(
                    record,
                    &problem.symbol_table,
                    term_graph,
                    clause_storage,
                    constraint_list,
                    lemma.literal,
                    lemma.valid_from,
                );
                self.close_branches(clause_storage);
            }
            Rule::EqualityReduction => {
                let goal = self
                    .stack
                    .last_mut()
                    .expect("equality reduction on empty tableau");
                goal.equality_reduction(
                    record,
                    &problem.symbol_table,
                    term_graph,
                    clause_storage,
                    constraint_list,
                );
                self.close_branches(clause_storage);
            }
            Rule::Extension(extension) => {
                let goal =
                    self.stack.last_mut().expect("extension on empty tableau");
                let new_goal = goal.extension(
                    record,
                    problem,
                    term_graph,
                    clause_storage,
                    constraint_list,
                    extension.position,
                );
                self.stack.push(new_goal);
                self.add_regularity_constraints(
                    constraint_list,
                    term_graph,
                    clause_storage,
                );
            }
        }
    }

    fn add_regularity_constraints(
        &mut self,
        constraint_list: &mut ConstraintList,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
    ) {
        let goal = self
            .stack
            .last()
            .expect("adding constraints to empty stack");
        let lemma_literals =
            self.available_lemmata().map(|id| self.lemmata[id].literal);
        let regularity_literals =
            self.path_literals(clause_storage).chain(lemma_literals);
        for path_literal in regularity_literals {
            for literal in goal.clause.literals(clause_storage) {
                if path_literal.polarity != literal.polarity {
                    continue;
                }
                if !path_literal.atom.possibly_equal(&literal.atom, term_graph)
                {
                    continue;
                }
                constraint_list
                    .add_disequality(path_literal.atom, literal.atom);
            }
        }
    }

    fn close_branches(&mut self, clause_storage: &ClauseStorage) {
        while let Some(solved) = self.stack.last() {
            if !solved.clause.is_empty() {
                return;
            }
            let valid_from = solved.valid_from;
            self.stack.pop();
            if let Some(parent) = self.stack.last_mut() {
                let (valid_from, mut literal) =
                    parent.solve_literal(clause_storage, valid_from);
                literal.polarity = !literal.polarity;
                let lemma = Lemma {
                    valid_from,
                    literal,
                };
                let id = self.lemmata.push(lemma);
                if valid_from < self.stack.len() {
                    self.stack[valid_from].lemmata =
                        self.stack[valid_from].lemmata.push(id);
                }
            }
        }
    }

    pub(crate) fn possible_rules(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
    ) {
        if let Some(goal) = self.stack.last() {
            self.possible_nonstart_rules(
                possible,
                problem,
                term_graph,
                clause_storage,
                goal,
            );
        } else {
            self.possible_start_rules(possible, problem);
        }
    }

    fn possible_start_rules(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
    ) {
        possible.extend(
            problem
                .start_clauses
                .iter()
                .copied()
                .map(|start_clause| StartRule { start_clause })
                .map(Rule::Start),
        );
    }

    fn possible_nonstart_rules(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        goal: &Goal,
    ) {
        let literal = goal.current_literal(clause_storage);
        if literal.atom.is_predicate() {
            self.possible_predicate_rules(
                possible,
                problem,
                term_graph,
                clause_storage,
                literal.polarity,
                literal.atom.get_predicate_symbol(term_graph),
            );
        } else if !literal.polarity {
            possible.push(Rule::EqualityReduction);
        }
    }

    fn possible_predicate_rules(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        polarity: bool,
        symbol: Id<Symbol>,
    ) {
        self.possible_reduction_rules(
            possible,
            term_graph,
            clause_storage,
            polarity,
            symbol,
        );
        self.possible_lemma_rules(possible, term_graph, polarity, symbol);
        self.possible_extension_rules(possible, problem, polarity, symbol);
    }

    fn possible_reduction_rules(
        &self,
        possible: &mut Vec<Rule>,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        polarity: bool,
        f: Id<Symbol>,
    ) {
        for goal in self.stack.into_iter().rev().skip(1) {
            let path_literal =
                self.stack[goal].current_literal(clause_storage);
            if path_literal.polarity == polarity
                || !path_literal.atom.is_predicate()
            {
                continue;
            }
            let g = path_literal.atom.get_predicate_symbol(term_graph);
            if f == g {
                possible.push(Rule::Reduction(ReductionRule { goal }));
            }
        }
    }

    fn possible_lemma_rules(
        &self,
        possible: &mut Vec<Rule>,
        term_graph: &TermGraph,
        polarity: bool,
        f: Id<Symbol>,
    ) {
        for lemma in self.available_lemmata() {
            let literal = self.lemmata[lemma].literal;
            if literal.polarity == polarity || !literal.atom.is_predicate() {
                continue;
            }
            let g = literal.atom.get_predicate_symbol(term_graph);
            if f == g {
                possible.push(Rule::Lemma(LemmaRule { lemma }));
            }
        }
    }

    fn possible_extension_rules(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        polarity: bool,
        f: Id<Symbol>,
    ) {
        let opposite = !polarity as usize;
        let positions = &problem.predicate_occurrences[opposite][f];
        possible.extend(
            positions
                .iter()
                .copied()
                .map(|position| ExtensionRule { position })
                .map(Rule::Extension),
        );
    }

    fn path_literals<'a, 'clauses, 'iterator>(
        &'a self,
        clause_storage: &'clauses ClauseStorage,
    ) -> impl Iterator<Item = Literal> + 'iterator
    where
        'a: 'iterator,
        'clauses: 'iterator,
    {
        self.stack
            .into_iter()
            .rev()
            .skip(1)
            .map(move |id| self.stack[id].current_literal(clause_storage))
    }

    fn available_lemmata(&self) -> impl Iterator<Item = Id<Lemma>> + '_ {
        self.stack
            .into_iter()
            .flat_map(move |id| self.stack[id].lemmata.items().copied())
    }
}
