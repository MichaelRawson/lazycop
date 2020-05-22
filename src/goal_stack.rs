use crate::goal::Goal;
use crate::io::record::Record;
use crate::prelude::*;

#[derive(Clone, Copy)]
struct Lemmatum(Id<Literal>);

#[derive(Default)]
pub(crate) struct GoalStack {
    stack: Block<Goal>,
    save_stack: Block<Goal>,
}

impl GoalStack {
    pub(crate) fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.stack.clear();
    }

    pub(crate) fn mark(&mut self) {
        self.save_stack.clear();
        self.save_stack.extend_from_slice(&self.stack.as_ref());
    }

    pub(crate) fn undo_to_mark(&mut self) {
        self.stack.clear();
        self.stack.extend_from_slice(&self.save_stack.as_ref());
    }

    pub(crate) fn num_open_branches(&self) -> u32 {
        self.stack
            .as_ref()
            .iter()
            .map(|goal| goal.len())
            .sum::<u32>()
            + 1
    }

    pub(crate) fn apply_rule<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        term_graph: &mut TermGraph,
        clause_storage: &mut ClauseStorage,
        solver: &mut Solver,
        rule: Rule,
    ) {
        match rule {
            Rule::Start(start) => {
                let start = Goal::start(
                    record,
                    problem,
                    term_graph,
                    clause_storage,
                    solver,
                    start.start_clause,
                );
                self.stack.push(start);
            }
            Rule::Reduction(reduction) => {
                let matching = self.stack[reduction.parent].current_literal();
                let goal = self.last_goal_mut();
                goal.reduction(
                    record,
                    &problem.symbol_table,
                    term_graph,
                    clause_storage,
                    solver,
                    matching,
                );
                let valid_from = reduction.parent + Offset::new(1);
                for parent_id in Range::new(valid_from, self.stack.len()) {
                    self.stack[parent_id].set_validity(valid_from);
                }
            }
            Rule::Extension(extension) => {
                let goal = self.last_goal_mut();
                let new_goal = goal.extension(
                    record,
                    problem,
                    term_graph,
                    clause_storage,
                    solver,
                    extension.clause,
                    extension.literal,
                );
                self.stack.push(new_goal);
                self.add_regularity_constraints(
                    solver,
                    term_graph,
                    clause_storage,
                );
            }
            Rule::Lemma(lemma) => {
                let goal = self.last_goal_mut();
                goal.lemma(
                    record,
                    &problem.symbol_table,
                    term_graph,
                    clause_storage,
                    solver,
                    lemma.literal,
                );
                for parent_id in Range::new(lemma.valid_from, self.stack.len())
                {
                    self.stack[parent_id].set_validity(lemma.valid_from);
                }
            }
            Rule::LazyExtension(extension) => {
                let goal = self.last_goal_mut();
                let new_goal = goal.lazy_extension(
                    record,
                    problem,
                    term_graph,
                    clause_storage,
                    solver,
                    extension.clause,
                    extension.literal,
                );
                self.stack.push(new_goal);
                self.add_regularity_constraints(
                    solver,
                    term_graph,
                    clause_storage,
                );
            }
            Rule::Reflexivity => {
                let goal = self.last_goal_mut();
                goal.reflexivity(
                    record,
                    &problem.symbol_table,
                    term_graph,
                    clause_storage,
                    solver,
                );
            }
        }
        self.close_branches();
    }

    fn close_branches(&mut self) {
        let current = self.last_goal_mut();
        if !current.is_empty() {
            return;
        }
        self.stack.pop();
        while let Some(goal) = self.stack.as_mut().last_mut() {
            let (valid_from, literal_id) = goal.close_literal();
            let goal_empty = goal.is_empty();
            self.stack[valid_from].add_lemmatum(literal_id);
            if goal_empty {
                self.stack.pop();
            } else {
                return;
            }
        }
    }

    fn add_regularity_constraints(
        &self,
        solver: &mut Solver,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
    ) {
        self.add_path_regularity_constraints(
            solver,
            term_graph,
            clause_storage,
        );
        self.add_lemmata_regularity_constraints(
            solver,
            term_graph,
            clause_storage,
        );
    }

    fn add_path_regularity_constraints(
        &self,
        solver: &mut Solver,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
    ) {
        let goal = self.last_goal();
        for path in self.path_literals().map(|id| &clause_storage[id]) {
            for open in goal.open_literals().map(|id| &clause_storage[id]) {
                if path.polarity == open.polarity {
                    path.atom.add_disequation_constraints(
                        solver, term_graph, &open.atom,
                    );
                }
            }
        }
    }

    fn add_lemmata_regularity_constraints(
        &self,
        solver: &mut Solver,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
    ) {
        let goal = self.last_goal();
        for lemma in
            self.available_lemmata().map(|(_, id)| &clause_storage[id])
        {
            for open in goal.open_literals().map(|id| &clause_storage[id]) {
                if lemma.polarity != open.polarity {
                    lemma.atom.add_disequation_constraints(
                        solver, term_graph, &open.atom,
                    );
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
        if let Some(goal) = self.stack.as_ref().last() {
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
        let literal = clause_storage[goal.current_literal()];
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
            possible.push(Rule::Reflexivity);
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
        self.possible_lemma_rules(
            clause_storage,
            possible,
            term_graph,
            polarity,
            symbol,
        );
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
        for parent in self.stack.range().rev().skip(1) {
            let literal_id = self.stack[parent].current_literal();
            let literal = clause_storage[literal_id];
            if literal.polarity == polarity || !literal.atom.is_predicate() {
                continue;
            }
            let g = literal.atom.get_predicate_symbol(term_graph);
            if f == g {
                possible.push(Rule::Reduction(ReductionRule { parent }));
            }
        }
    }

    fn possible_lemma_rules(
        &self,
        clause_storage: &ClauseStorage,
        possible: &mut Vec<Rule>,
        term_graph: &TermGraph,
        polarity: bool,
        f: Id<Symbol>,
    ) {
        for (valid_from, literal_id) in self.available_lemmata() {
            let literal = clause_storage[literal_id];
            if literal.polarity != polarity || !literal.atom.is_predicate() {
                continue;
            }
            let g = literal.atom.get_predicate_symbol(term_graph);
            if f == g {
                let literal = literal_id;
                possible.push(Rule::Lemma(LemmaRule {
                    literal,
                    valid_from,
                }));
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
        let rule_type = if problem.equality {
            Rule::LazyExtension
        } else {
            Rule::Extension
        };
        possible.extend(
            positions
                .iter()
                .copied()
                .map(|(clause, literal)| ExtensionRule { clause, literal })
                .map(rule_type),
        );
    }

    fn available_lemmata(
        &self,
    ) -> impl Iterator<Item = (Id<Goal>, Id<Literal>)> + '_ {
        self.stack.range().flat_map(move |goal_id| {
            self.stack[goal_id]
                .available_lemmata()
                .map(move |literal_id| (goal_id, literal_id))
        })
    }

    fn path_literals(&self) -> impl Iterator<Item = Id<Literal>> + '_ {
        self.stack
            .as_ref()
            .iter()
            .rev()
            .skip(1)
            .map(|goal| goal.current_literal())
    }

    fn last_goal(&self) -> &Goal {
        self.stack
            .as_ref()
            .last()
            .expect("last goal of empty tableau")
    }

    fn last_goal_mut(&mut self) -> &mut Goal {
        self.stack
            .as_mut()
            .last_mut()
            .expect("last goal of empty tableau")
    }
}
