use crate::core::goal::Goal;
use crate::io::record::Record;
use crate::util::id_map::IdMap;
use crate::prelude::*;

#[derive(Clone, Copy)]
struct Lemmatum(Id<Literal>);

#[derive(Default)]
pub(crate) struct GoalStack {
    stack: Arena<Goal>,
    save_stack: Arena<Goal>,
    lemmata: IdMap<Goal, Arena<Lemmatum>>,
    save_lemmata: IdMap<Goal, Arena<Lemmatum>>
}

impl GoalStack {
    pub(crate) fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.stack.clear();
        self.lemmata.reset();
    }

    pub(crate) fn mark(&mut self) {
        self.save_stack.clear();
        self.save_stack.extend_from(&self.stack);
        self.save_lemmata.reset();
        self.save_lemmata.ensure_capacity(self.stack.limit());
        for goal_id in self.stack.into_iter() {
            self.save_lemmata[goal_id].extend_from(&self.lemmata[goal_id]);
        }
    }

    pub(crate) fn undo_to_mark(&mut self) {
        self.stack.clear();
        self.stack.extend_from(&self.save_stack);
        self.lemmata.reset();
        self.lemmata.ensure_capacity(self.stack.limit());
        for goal_id in self.stack.into_iter() {
            self.lemmata[goal_id].extend_from(&self.save_lemmata[goal_id]);
        }
    }

    pub(crate) fn open_branches(&self) -> u32 {
        self.stack
            .into_iter()
            .map(|id| self.stack[id].len())
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
                    start.start_clause,
                );
                self.push(start);
            }
            Rule::Reduction(reduction) => {
                let matching = self.stack[reduction.parent].current_literal();
                let goal =
                    self.stack.last_mut().expect("reduction on empty tableau");
                goal.reduction(
                    record,
                    &problem.symbol_table,
                    term_graph,
                    clause_storage,
                    solver,
                    matching,
                );
                let valid_from = reduction.parent.increment();
                for parent_id in IdRange::new(valid_from, self.stack.limit()) {
                    self.stack[parent_id].set_validity(valid_from);
                }
            }
            Rule::Extension(extension) => {
                let goal =
                    self.stack.last_mut().expect("extension on empty tableau");
                let new_goal = goal.extension(
                    record,
                    problem,
                    term_graph,
                    clause_storage,
                    solver,
                    extension.clause,
                    extension.literal,
                );
                self.push(new_goal);
                self.add_regularity_constraints(solver, clause_storage);
            }
            Rule::Lemma(lemma) => {
                let goal = self
                    .stack
                    .last_mut()
                    .expect("lemma applied on empty tableau");
                goal.lemma(
                    record,
                    &problem.symbol_table,
                    term_graph,
                    clause_storage,
                    solver,
                    lemma.literal,
                );
                for parent_id in
                    IdRange::new(lemma.valid_from, self.stack.limit())
                {
                    self.stack[parent_id].set_validity(lemma.valid_from);
                }
            }
            Rule::LazyExtension(extension) => {
                let goal =
                    self.stack.last_mut().expect("extension on empty tableau");
                let new_goal = goal.lazy_extension(
                    record,
                    problem,
                    term_graph,
                    clause_storage,
                    solver,
                    extension.clause,
                    extension.literal,
                );
                self.push(new_goal);
                self.add_regularity_constraints(solver, clause_storage);
            }
            Rule::Reflexivity => {
                let goal = self
                    .stack
                    .last_mut()
                    .expect("reflexivity on empty tableau");
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
        self.lemmata.ensure_capacity(self.stack.limit());
        let current = self.stack.last().expect("empty stack");
        if !current.is_empty() {
            return;
        }
        self.pop();
        while let Some(goal) = self.stack.last_mut() {
            let (valid_from, literal) = goal.close_literal();
            self.lemmata[valid_from].push(Lemmatum(literal));
            if goal.is_empty() {
                self.pop();
            }
            else {
                return;
            }
        }
    }

    fn add_regularity_constraints(
        &mut self,
        solver: &mut Solver,
        clause_storage: &ClauseStorage,
    ) {
        let goal = self
            .stack
            .last()
            .expect("adding constraints to empty stack");
        let path_literals = self.path_literals().map(|id| clause_storage[id]);
        let lemmata = self
            .available_lemmata()
            .map(|(_, id)| clause_storage[id].inverted());
        let regularity_literals = path_literals.chain(lemmata);
        for regularity_literal in regularity_literals {
            for literal in goal.open_literals() {
                let literal = clause_storage[literal];
                if regularity_literal.polarity == literal.polarity
                    && regularity_literal.atom.is_predicate()
                    && literal.atom.is_predicate()
                {
                    solver.assert_not_equal(
                        regularity_literal.atom.get_predicate(),
                        literal.atom.get_predicate(),
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
        for parent in self.stack.into_iter().rev().skip(1) {
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
        self.stack.into_iter().flat_map(move |goal_id| {
            let lemmata = &self.lemmata[goal_id];
            lemmata.into_iter().map(move |id| (goal_id, lemmata[id].0))
        })
    }

    fn path_literals(&self) -> impl Iterator<Item = Id<Literal>> + '_ {
        self.stack
            .into_iter()
            .rev()
            .skip(1)
            .map(move |id| self.stack[id].current_literal())
    }

    fn push(&mut self, goal: Goal) {
        self.stack.push(goal);
        self.lemmata.ensure_capacity(self.stack.limit());
    }

    fn pop(&mut self) {
        self.stack.pop();
        self.lemmata[self.stack.limit()].clear();
    }
}
