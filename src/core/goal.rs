use crate::core::goal_stack::Lemma;
use crate::io::record::Record;
use crate::prelude::*;
use crate::util::rc_stack::RcStack;

#[derive(Clone)]
pub(crate) struct Goal {
    pub(crate) clause: Clause,
    pub(crate) lemmata: RcStack<Id<Lemma>>,
    pub(crate) valid_from: Id<Goal>,
}

impl Goal {
    pub(crate) fn num_open_branches(&self) -> u32 {
        self.clause.len()
    }

    pub(crate) fn solve_literal(
        &mut self,
        clause_storage: &ClauseStorage,
        valid_from: Id<Goal>,
    ) -> (Id<Goal>, Literal) {
        let valid_from = std::cmp::max(self.valid_from, valid_from);
        self.valid_from = Id::default();
        let literal = self
            .clause
            .pop_literal(clause_storage)
            .expect("literal marked solved on empty goal");
        (valid_from, literal)
    }

    pub(crate) fn start<R: Record>(
        record: &mut R,
        problem: &Problem,
        term_graph: &mut TermGraph,
        clause_storage: &mut ClauseStorage,
        start_clause: Id<ProblemClause>,
    ) -> Self {
        let offset = term_graph.current_offset();
        let problem_clause = &problem[start_clause];
        let clause = clause_storage.clause(
            problem_clause
                .literals
                .into_iter()
                .map(|id| problem_clause.literals[id].offset(offset)),
        );
        term_graph.extend_from(&problem_clause.term_graph);
        record.start(
            &problem.symbol_table,
            &term_graph,
            &clause_storage,
            clause,
        );
        let lemmata = RcStack::default();
        let valid_from = Id::default();
        Self {
            clause,
            lemmata,
            valid_from,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reduction<R: Record>(
        &mut self,
        record: &mut R,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        constraint_list: &mut ConstraintList,
        matching: Literal,
        valid_from: Id<Goal>,
    ) {
        let literal = self.pop_literal(clause_storage);
        self.valid_from = std::cmp::max(self.valid_from, valid_from);
        let p = literal.atom.get_predicate();
        let q = matching.atom.get_predicate();
        constraint_list.add_equality(p, q);
        record.reduction(
            &symbol_table,
            &term_graph,
            &clause_storage,
            self.clause,
            p,
            q,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lemma<R: Record>(
        &mut self,
        record: &mut R,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        constraint_list: &mut ConstraintList,
        lemma: Literal,
        valid_from: Id<Goal>,
    ) {
        let literal = self.pop_literal(clause_storage);
        self.valid_from = std::cmp::max(self.valid_from, valid_from);
        let p = literal.atom.get_predicate();
        let q = lemma.atom.get_predicate();
        constraint_list.add_equality(p, q);
        record.lemma(
            &symbol_table,
            &term_graph,
            &clause_storage,
            self.clause,
            p,
            q,
        );
    }

    pub(crate) fn equality_reduction<R: Record>(
        &mut self,
        record: &mut R,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        constraint_list: &mut ConstraintList,
    ) {
        let literal = self.pop_literal(clause_storage);
        let (left, right) = literal.atom.get_equality();
        constraint_list.add_equality(left, right);
        record.equality_reduction(
            &symbol_table,
            &term_graph,
            &clause_storage,
            self.clause,
            left,
            right,
        );
    }

    pub(crate) fn extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        term_graph: &mut TermGraph,
        clause_storage: &mut ClauseStorage,
        constraint_list: &mut ConstraintList,
        position: Id<Position>,
    ) -> Self {
        let offset = term_graph.current_offset();
        let literal = self.current_literal(clause_storage);
        let position = &problem[position];
        let problem_clause = &problem[position.clause];
        let matching = problem_clause.literals[position.literal];
        let p = literal.atom.get_predicate();
        let q = matching.atom.offset(offset).get_predicate();
        let clause = clause_storage.clause_with(
            problem_clause
                .literals
                .into_iter()
                .filter(|id| *id != position.literal)
                .map(|id| problem_clause.literals[id].offset(offset)),
            Literal::new(false, Atom::Equality(p, q)),
        );
        term_graph.extend_from(&problem_clause.term_graph);
        for original_literal in self.clause.literals(&clause_storage) {
            for new_literal in clause.literals(&clause_storage).skip(1) {
                if original_literal.polarity == new_literal.polarity {
                    continue;
                }
                if !original_literal
                    .atom
                    .possibly_equal(&new_literal.atom, term_graph)
                {
                    continue;
                }
                constraint_list
                    .add_disequality(original_literal.atom, new_literal.atom);
            }
        }

        record.extension(
            &problem.symbol_table,
            &term_graph,
            &clause_storage,
            self.clause.peek_rest(),
            clause,
        );
        let lemmata = RcStack::default();
        let valid_from = Id::default();
        Self {
            clause,
            lemmata,
            valid_from,
        }
    }

    pub(crate) fn current_literal(
        &self,
        clause_storage: &ClauseStorage,
    ) -> Literal {
        self.clause
            .current_literal(clause_storage)
            .expect("empty clause")
    }

    pub(crate) fn pop_literal(
        &mut self,
        clause_storage: &ClauseStorage,
    ) -> Literal {
        self.clause
            .pop_literal(clause_storage)
            .expect("empty clause")
    }
}
