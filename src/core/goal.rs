use crate::io::record::Record;
use crate::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct Goal {
    clause: Clause,
}

impl Goal {
    pub(crate) fn is_empty(&self) -> bool {
        self.clause.is_empty()
    }

    pub(crate) fn len(&self) -> u32 {
        self.clause.len() - 1
    }

    pub(crate) fn open_literals(&self) -> impl Iterator<Item = Id<Literal>> {
        self.clause.open()
    }

    pub(crate) fn closed_literals(&self) -> impl Iterator<Item = Id<Literal>> {
        self.clause.closed()
    }

    pub(crate) fn current_literal(&self) -> Id<Literal> {
        self.clause.current_literal()
    }

    pub(crate) fn close_literal(&mut self) -> Id<Literal> {
        self.clause.close_literal().expect("empty clause")
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
        let clause = clause_storage.create_clause(
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
            clause.open(),
        );
        Self { clause }
    }

    pub(crate) fn reduction<R: Record>(
        &mut self,
        record: &mut R,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        constraint_list: &mut ConstraintList,
        matching: Literal,
    ) {
        let literal = clause_storage[self.close_literal()];
        let p = literal.atom.get_predicate();
        let q = matching.atom.get_predicate();
        constraint_list.add_equality(p, q);
        record.reduction(
            &symbol_table,
            &term_graph,
            &clause_storage,
            self.clause.open(),
            p,
            q,
        );
    }

    pub(crate) fn lemma<R: Record>(
        &mut self,
        record: &mut R,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        constraint_list: &mut ConstraintList,
        lemma: Literal,
    ) {
        let literal = clause_storage[self.close_literal()];
        let p = literal.atom.get_predicate();
        let q = lemma.atom.get_predicate();
        constraint_list.add_equality(p, q);
        record.lemma(
            &symbol_table,
            &term_graph,
            &clause_storage,
            self.clause.open(),
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
        let literal = clause_storage[self.close_literal()];
        let (left, right) = literal.atom.get_equality();
        constraint_list.add_equality(left, right);
        record.equality_reduction(
            &symbol_table,
            &term_graph,
            &clause_storage,
            self.clause.open(),
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
        let literal = clause_storage[self.current_literal()];
        let position = &problem[position];
        let problem_clause = &problem[position.clause];
        let matching = problem_clause.literals[position.literal];
        let p = literal.atom.get_predicate();
        let q = matching.atom.offset(offset).get_predicate();
        let clause = clause_storage.create_clause_with(
            problem_clause
                .literals
                .into_iter()
                .filter(|id| *id != position.literal)
                .map(|id| problem_clause.literals[id].offset(offset)),
            Literal::new(false, Atom::Equality(p, q)),
        );
        term_graph.extend_from(&problem_clause.term_graph);
        for original_literal_id in self.clause.open() {
            for new_literal_id in clause.open().skip(1) {
                let original_literal = clause_storage[original_literal_id];
                let new_literal = clause_storage[new_literal_id];
                if original_literal.polarity == new_literal.polarity {
                    continue;
                }
                if Atom::possibly_equal(
                    &original_literal.atom,
                    &new_literal.atom,
                    term_graph,
                ) {
                    constraint_list.add_disequality(
                        original_literal.atom,
                        new_literal.atom,
                    );
                }
            }
        }

        record.extension(
            &problem.symbol_table,
            &term_graph,
            &clause_storage,
            self.clause.open().skip(1),
            clause.open(),
        );
        Self { clause }
    }
}
