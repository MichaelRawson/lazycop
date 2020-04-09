use crate::core::goal::Goal;
use crate::io::record::Record;
use crate::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct Lemma {
    valid_for: Id<Goal>,
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

    pub(crate) fn copy_from(&mut self, other: &GoalStack) {
        self.clear();
        self.stack.copy_from(&other.stack);
        self.lemmata.copy_from(&other.lemmata);
    }

    pub(crate) fn open_branches(&self) -> u32 {
        self.stack
            .into_iter()
            .map(|id| self.stack[id].num_open_branches())
            .sum::<u32>()
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
                record.start_inference("start");
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
                record.start_inference("reduction");
                let literal = self.stack[reduction.goal]
                    .clause
                    .current_literal(clause_storage)
                    .expect("reduction with empty clause");
                let mut valid_for = reduction.goal;
                valid_for.increment();
                let goal =
                    self.stack.last_mut().expect("reduction on empty tableau");
                goal.reduction(
                    record,
                    &problem.symbol_table,
                    term_graph,
                    clause_storage,
                    constraint_list,
                    literal,
                    valid_for,
                );
                self.close_branches(
                    record,
                    &problem.symbol_table,
                    term_graph,
                    clause_storage,
                );
            }
            Rule::Lemma(lemma) => {
                record.start_inference("lemma");
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
                    lemma.valid_for,
                );
                self.close_branches(
                    record,
                    &problem.symbol_table,
                    term_graph,
                    clause_storage,
                );
            }
            Rule::EqualityReduction => {
                record.start_inference("equality reduction");
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
                self.close_branches(
                    record,
                    &problem.symbol_table,
                    term_graph,
                    clause_storage,
                );
            }
            Rule::Extension(extension) => {
                record.start_inference("extension");
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
                self.add_regularity_constraints(
                    constraint_list,
                    term_graph,
                    clause_storage,
                    &new_goal,
                );
                self.stack.push(new_goal);
            }
        }
        record.end_inference();
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

    fn close_branches<R: Record>(
        &mut self,
        record: &mut R,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
    ) {
        while let Some(solved) = self.stack.last() {
            if !solved.clause.is_empty() {
                return;
            }
            let valid_for = solved.valid_for;
            self.stack.pop();
            if let Some(parent) = self.stack.last_mut() {
                let (valid_for, literal) =
                    parent.mark_literal_solved(clause_storage, valid_for);
                record.lemma(symbol_table, term_graph, literal);
                let lemma = Lemma { valid_for, literal };
                let id = self.lemmata.push(lemma);
                if valid_for < self.stack.len() {
                    self.stack[valid_for].lemmata =
                        self.stack[valid_for].lemmata.push(id);
                }
            }
        }
    }

    fn add_regularity_constraints(
        &self,
        constraint_list: &mut ConstraintList,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        goal: &Goal,
    ) {
        for literal in goal.clause.literals(clause_storage) {
            for path_literal in self.path_literals(clause_storage) {
                if literal.polarity != path_literal.polarity {
                    continue;
                }
                Atom::compute_disequation(
                    &literal.atom,
                    &path_literal.atom,
                    constraint_list,
                    term_graph,
                );
            }
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
        let literal = goal
            .clause
            .current_literal(clause_storage)
            .expect("empty goal");
        match literal.atom {
            Atom::Predicate(term) => self.possible_predicate_rules(
                possible,
                problem,
                term_graph,
                clause_storage,
                literal.polarity,
                term,
            ),
            Atom::Equality(_, _) => {
                if !literal.polarity {
                    possible.push(Rule::EqualityReduction);
                }
            }
        }
    }

    fn possible_predicate_rules(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        polarity: bool,
        term: Id<Term>,
    ) {
        self.possible_reduction_rules(
            possible,
            term_graph,
            clause_storage,
            polarity,
            term,
        );
        self.possible_lemma_rules(possible, term_graph, polarity, term);
        self.possible_extension_rules(
            possible, problem, term_graph, polarity, term,
        );
    }

    fn possible_reduction_rules(
        &self,
        possible: &mut Vec<Rule>,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        polarity: bool,
        term: Id<Term>,
    ) {
        let mut parent_goals = self.stack.into_iter().rev();
        parent_goals.next();
        for goal in parent_goals {
            let path_literal = self.stack[goal]
                .clause
                .current_literal(clause_storage)
                .expect("empty goal in stack");
            if path_literal.polarity == polarity {
                continue;
            }
            if let Atom::Predicate(other) = path_literal.atom {
                if let (TermView::Function(f, _), TermView::Function(g, _)) =
                    (term_graph.view(term), term_graph.view(other))
                {
                    if f == g {
                        possible.push(Rule::Reduction(ReductionRule { goal }));
                    }
                }
            }
        }
    }

    fn possible_lemma_rules(
        &self,
        possible: &mut Vec<Rule>,
        term_graph: &TermGraph,
        polarity: bool,
        term: Id<Term>,
    ) {
        for lemma in self.available_lemmata() {
            let literal = self.lemmata[lemma].literal;
            if literal.polarity != polarity {
                continue;
            }
            if let Atom::Predicate(other) = literal.atom {
                if let (TermView::Function(f, _), TermView::Function(g, _)) =
                    (term_graph.view(term), term_graph.view(other))
                {
                    if f == g {
                        possible.push(Rule::Lemma(LemmaRule { lemma }));
                    }
                }
            }
        }
    }

    fn possible_extension_rules(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        term_graph: &TermGraph,
        polarity: bool,
        term: Id<Term>,
    ) {
        if let TermView::Function(p, _) = term_graph.view(term) {
            if let Some(positions) =
                problem.predicate_occurrences[!polarity as usize].get(p)
            {
                possible.extend(
                    positions
                        .iter()
                        .copied()
                        .map(|position| ExtensionRule { position })
                        .map(Rule::Extension),
                );
            }
        }
    }

    fn path_literals<'a, 'storage, 'iterator>(
        &'a self,
        clause_storage: &'storage ClauseStorage,
    ) -> impl Iterator<Item = Literal> + 'iterator
    where
        'storage: 'iterator,
        'a: 'iterator,
    {
        self.stack.into_iter().rev().skip(1).map(move |id| {
            self.stack[id]
                .clause
                .current_literal(clause_storage)
                .expect("empty clause in stack")
        })
    }

    fn available_lemmata(&self) -> impl Iterator<Item = Id<Lemma>> + '_ {
        self.stack
            .into_iter()
            .flat_map(move |id| self.stack[id].lemmata.items().copied())
    }
}
