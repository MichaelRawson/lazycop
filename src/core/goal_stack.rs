use crate::core::goal::Goal;
use crate::io::record::Record;
use crate::prelude::*;

#[derive(Default)]
pub(crate) struct GoalStack {
    stack: Arena<Goal>,
}

impl GoalStack {
    pub(crate) fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.stack.clear();
    }

    pub(crate) fn reset_to(&mut self, other: &GoalStack) {
        self.clear();
        self.stack.extend_from(&other.stack);
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
                self.stack.push(start);
            }
            Rule::Reduction(reduction) => {
                let literal = clause_storage[reduction.literal];
                let goal =
                    self.stack.last_mut().expect("reduction on empty tableau");
                goal.reduction(
                    record,
                    &problem.symbol_table,
                    term_graph,
                    clause_storage,
                    solver,
                    literal,
                );
            }
            Rule::Lemma(lemma) => {
                let literal = clause_storage[lemma.literal];
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
                    literal,
                );
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
                    solver,
                );
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
                    extension.position,
                );
                self.stack.push(new_goal);
                self.add_regularity_constraints(solver, clause_storage);
            }
        }
        self.close_branches();
    }

    fn close_branches(&mut self) {
        let current = self.stack.last().expect("empty stack");
        if !current.is_empty() {
            return;
        }
        self.stack.pop();
        while let Some(goal) = self.stack.last_mut() {
            goal.close_literal();
            if !goal.is_empty() {
                return;
            }
            self.stack.pop();
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
        let lemmata = self.available_lemmata().map(|id| {
            let mut lemma = clause_storage[id];
            lemma.polarity = !lemma.polarity;
            lemma
        });
        let regularity_literals = path_literals.chain(lemmata);
        for regularity_literal in regularity_literals {
            for literal in goal.open_literals() {
                let literal = clause_storage[literal];
                if regularity_literal.polarity != literal.polarity {
                    continue;
                }
                if !Atom::same_type(&regularity_literal.atom, &literal.atom) {
                    continue;
                }
                solver.unequal(regularity_literal.atom, literal.atom);
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
        for literal_id in self.path_literals() {
            let literal = clause_storage[literal_id];
            if literal.polarity == polarity || !literal.atom.is_predicate() {
                continue;
            }
            let g = literal.atom.get_predicate_symbol(term_graph);
            if f == g {
                let literal = literal_id;
                possible.push(Rule::Reduction(ReductionRule { literal }));
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
        for literal_id in self.available_lemmata() {
            let literal = clause_storage[literal_id];
            if literal.polarity != polarity || !literal.atom.is_predicate() {
                continue;
            }
            let g = literal.atom.get_predicate_symbol(term_graph);
            if f == g {
                let literal = literal_id;
                possible.push(Rule::Lemma(LemmaRule { literal }));
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

    fn available_lemmata(&self) -> impl Iterator<Item = Id<Literal>> + '_ {
        self.stack
            .into_iter()
            .flat_map(move |id| self.stack[id].closed_literals())
    }

    fn path_literals(&self) -> impl Iterator<Item = Id<Literal>> + '_ {
        self.stack
            .into_iter()
            .rev()
            .skip(1)
            .map(move |id| self.stack[id].current_literal())
    }
}
