use crate::core::goal_stack::Lemma;
use crate::io::record::Record;
use crate::prelude::*;
use crate::util::rc_stack::RcStack;

#[derive(Clone)]
pub(crate) struct Goal {
    pub(crate) clause: Clause,
    pub(crate) lemmata: RcStack<Id<Lemma>>,
    pub(crate) valid_for: Id<Goal>,
}

impl Goal {
    pub(crate) fn num_open_branches(&self) -> u32 {
        self.clause.len() - 1
    }

    pub(crate) fn mark_literal_solved(
        &mut self,
        clause_storage: &ClauseStorage,
        valid_for: Id<Goal>,
    ) -> (Id<Goal>, Literal) {
        let valid_for = std::cmp::max(self.valid_for, valid_for);
        self.valid_for = Id::default();
        let literal = self
            .clause
            .pop_literal(clause_storage)
            .expect("literal marked solved on empty goal");
        (valid_for, literal)
    }

    pub(crate) fn start<R: Record>(
        record: &mut R,
        problem: &Problem,
        term_graph: &mut TermGraph,
        clause_storage: &mut ClauseStorage,
        start_clause: Id<ProblemClause>,
    ) -> Self {
        record.therefore();
        let (literals, new_term_graph) = problem.clause_data(start_clause);
        let clause =
            clause_storage.copy(term_graph.current_offset(), literals);
        term_graph.copy(new_term_graph);
        record.clause(
            &problem.symbol_table,
            &term_graph,
            &clause_storage,
            clause,
        );
        let lemmata = RcStack::default();
        let valid_for = Id::default();
        Self {
            clause,
            lemmata,
            valid_for,
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
        valid_for: Id<Goal>,
    ) {
        record.clause(
            &symbol_table,
            &term_graph,
            &clause_storage,
            self.clause,
        );
        record.therefore();
        let literal = self
            .clause
            .pop_literal(&clause_storage)
            .expect("reduction on empty clause");
        self.valid_for = std::cmp::max(self.valid_for, valid_for);
        if let (Atom::Predicate(p), Atom::Predicate(q)) =
            (literal.atom, matching.atom)
        {
            constraint_list.add_equality(p, q);
            record.equality_constraint(&symbol_table, &term_graph, p, q);
        } else {
            unreachable!("reduction on non-predicate literal");
        };
        record.clause(
            &symbol_table,
            &term_graph,
            &clause_storage,
            self.clause,
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
        valid_for: Id<Goal>,
    ) {
        record.clause(
            &symbol_table,
            &term_graph,
            &clause_storage,
            self.clause,
        );
        record.lemma(
            &symbol_table,
            &term_graph,
            lemma,
        );
        record.therefore();
        let literal = self
            .clause
            .pop_literal(&clause_storage)
            .expect("lemma applied on empty clause");
        self.valid_for = std::cmp::max(self.valid_for, valid_for);
        if let (Atom::Predicate(p), Atom::Predicate(q)) =
            (literal.atom, lemma.atom)
        {
            constraint_list.add_equality(p, q);
            record.equality_constraint(&symbol_table, &term_graph, p, q);
        } else {
            unreachable!("lemma applied on non-predicate literal");
        };
        record.clause(
            &symbol_table,
            &term_graph,
            &clause_storage,
            self.clause,
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
        record.clause(
            &symbol_table,
            &term_graph,
            &clause_storage,
            self.clause,
        );
        record.therefore();
        let literal = self
            .clause
            .pop_literal(&clause_storage)
            .expect("equality reduction on empty clause");
        let (left, right) = if let Atom::Equality(left, right) = literal.atom {
            (left, right)
        } else {
            unreachable!("equality reduction on non-equality literal");
        };
        constraint_list.add_equality(left, right);
        record.equality_constraint(&symbol_table, &term_graph, left, right);
        record.clause(
            &symbol_table,
            &term_graph,
            &clause_storage,
            self.clause,
        );
    }

    pub(crate) fn extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        term_graph: &mut TermGraph,
        clause_storage: &mut ClauseStorage,
        constraint_list: &mut ConstraintList,
        position: Position,
    ) -> Self {
        record.clause(
            &problem.symbol_table,
            &term_graph,
            &clause_storage,
            self.clause,
        );
        record.therefore();

        let offset = term_graph.current_offset();
        let literal = self
            .clause
            .current_literal(&clause_storage)
            .expect("extension on empty clause");
        let (problem_clause, matching_literal) = position;
        let (literals, new_term_graph) = problem.clause_data(problem_clause);
        let (left, right) =
            match (literal.atom, literals[matching_literal].atom) {
                (Atom::Predicate(p), Atom::Predicate(q)) => (p, q + offset),
                _ => unreachable!("extension on non-matching literal"),
            };
        term_graph.copy(new_term_graph);
        let clause = clause_storage.copy_replace(
            offset,
            literals,
            matching_literal,
            Literal::new(false, Atom::Equality(left, right)),
        );

        for original_literal in self.clause.literals(&clause_storage) {
            for new_literal in clause.literals(&clause_storage) {
                if original_literal.polarity == new_literal.polarity {
                    continue;
                }
                Atom::compute_disequation(
                    &original_literal.atom,
                    &new_literal.atom,
                    constraint_list,
                    term_graph,
                );
            }
        }

        record.clause(
            &problem.symbol_table,
            &term_graph,
            &clause_storage,
            clause,
        );
        let lemmata = RcStack::default();
        let valid_for = Id::default();
        Self {
            clause,
            lemmata,
            valid_for,
        }
    }
}
