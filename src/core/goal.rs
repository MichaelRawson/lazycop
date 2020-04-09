use crate::io::record::Record;
use crate::prelude::*;

#[derive(Clone, Copy)]
pub struct Goal {
    clause: Clause,
}

impl Goal {
    pub fn is_finished(self) -> bool {
        self.clause.is_empty()
    }

    pub fn open_branches(self) -> u32 {
        self.clause.len() - 1
    }

    pub fn current_literal(
        self,
        clause_storage: &ClauseStorage,
    ) -> Option<Literal> {
        self.clause.current_literal(clause_storage)
    }

    pub fn pop_literal(
        &mut self,
        clause_storage: &ClauseStorage,
    ) -> Option<Literal> {
        self.clause.pop_literal(clause_storage)
    }

    pub fn start<R: Record>(
        record: &mut R,
        problem: &Problem,
        term_graph: &mut TermGraph,
        clause_storage: &mut ClauseStorage,
        start_clause: Id<ProblemClause>,
    ) -> Self {
        record.start_inference("start");
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
        record.end_inference();
        Self { clause }
    }

    pub fn reduction<R: Record>(
        &mut self,
        record: &mut R,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        constraint_list: &mut ConstraintList,
        by: Goal,
    ) {
        record.start_inference("reduction");
        record.clause(&symbol_table, &term_graph, &clause_storage, by.clause);
        record.clause(
            &symbol_table,
            &term_graph,
            &clause_storage,
            self.clause,
        );
        record.therefore();
        let matching = by
            .clause
            .current_literal(&clause_storage)
            .expect("reduction by empty clause");
        let literal = self
            .clause
            .pop_literal(&clause_storage)
            .expect("reduction on empty clause");
        let (left, right) = if let (Atom::Predicate(p), Atom::Predicate(q)) =
            (literal.atom, matching.atom)
        {
            (p, q)
        } else {
            unreachable!("reduction on non-predicate literal");
        };
        constraint_list.add_equality(left, right);
        record.equality_constraint(&symbol_table, &term_graph, left, right);
        record.clause(
            &symbol_table,
            &term_graph,
            &clause_storage,
            self.clause,
        );
        record.end_inference();
    }

    pub fn equality_reduction<R: Record>(
        &mut self,
        record: &mut R,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        constraint_list: &mut ConstraintList,
    ) {
        record.start_inference("equality reduction");
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
        record.end_inference();
    }

    pub fn extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        term_graph: &mut TermGraph,
        clause_storage: &mut ClauseStorage,
        constraint_list: &mut ConstraintList,
        position: Position,
    ) -> Self {
        record.start_inference("extension");
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
                if original_literal.polarity != new_literal.polarity {
                    if let (Atom::Predicate(p), Atom::Predicate(q)) =
                        (original_literal.atom, new_literal.atom)
                    {
                        constraint_list.add_disequality(p, q);
                    }
                }
            }
        }

        record.clause(
            &problem.symbol_table,
            &term_graph,
            &clause_storage,
            clause,
        );
        record.end_inference();
        Self { clause }
    }
}
