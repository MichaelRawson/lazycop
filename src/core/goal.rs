use crate::io::record::Record;
use crate::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct Goal {
    clause: Clause,
    valid_from: Id<Goal>,
}

impl Goal {
    pub(crate) fn new(clause: Clause) -> Self {
        let valid_from = Id::default();
        Self {
            clause,
            valid_from,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.clause.is_empty()
    }

    pub(crate) fn len(&self) -> u32 {
        self.clause.len() - 1
    }

    pub(crate) fn open_literals(&self) -> impl Iterator<Item = Id<Literal>> {
        self.clause.open()
    }

    pub(crate) fn current_literal(&self) -> Id<Literal> {
        self.clause.current_literal()
    }

    pub(crate) fn close_literal(&mut self) -> (Id<Goal>, Id<Literal>) {
        let valid_from = std::mem::take(&mut self.valid_from);
        let literal = self.clause.close_literal();
        (valid_from, literal)
    }

    pub(crate) fn set_validity(&mut self, valid_from: Id<Goal>) {
        self.valid_from = std::cmp::max(self.valid_from, valid_from);
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
        Self::new(clause)
    }

    pub(crate) fn reduction<R: Record>(
        &mut self,
        record: &mut R,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        solver: &mut Solver,
        matching: Id<Literal>,
    ) {
        let literal = clause_storage[self.clause.close_literal()];
        let matching = clause_storage[matching];
        let p = literal.atom.get_predicate();
        let q = matching.atom.get_predicate();
        solver.assert_equal(p, q);
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
        solver: &mut Solver,
        lemma: Id<Literal>,
    ) {
        let literal = clause_storage[self.clause.close_literal()];
        let lemma = clause_storage[lemma];
        let p = literal.atom.get_predicate();
        let q = lemma.atom.get_predicate();
        solver.assert_equal(p, q);
        record.lemma(
            &symbol_table,
            &term_graph,
            &clause_storage,
            self.clause.open(),
            p,
            q,
        );
    }

    pub(crate) fn extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        term_graph: &mut TermGraph,
        clause_storage: &mut ClauseStorage,
        solver: &mut Solver,
        clause_id: Id<ProblemClause>,
        literal_id: Id<Literal>,
    ) -> Self {
        let offset = term_graph.current_offset();
        let literal = clause_storage[self.current_literal()];
        let problem_clause = &problem[clause_id];
        let matching = problem_clause.literals[literal_id];
        let p = literal.atom.get_predicate();
        let q = matching.atom.offset(offset).get_predicate();
        let clause = clause_storage.create_clause(
            problem_clause
                .literals
                .into_iter()
                .filter(|id| *id != literal_id)
                .map(|id| problem_clause.literals[id].offset(offset)),
        );
        term_graph.extend_from(&problem_clause.term_graph);
        solver.assert_equal(p, q);
        self.add_extension_constraints(solver, clause_storage, clause);

        record.extension(
            &problem.symbol_table,
            &term_graph,
            &clause_storage,
            self.clause.pending(),
            clause.open(),
            p,
            q,
        );
        Self::new(clause)
    }

    pub(crate) fn lazy_extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        term_graph: &mut TermGraph,
        clause_storage: &mut ClauseStorage,
        solver: &mut Solver,
        clause_id: Id<ProblemClause>,
        literal_id: Id<Literal>,
    ) -> Self {
        let offset = term_graph.current_offset();
        let literal = clause_storage[self.current_literal()];
        let problem_clause = &problem[clause_id];
        let matching = problem_clause.literals[literal_id];
        let p = literal.atom.get_predicate();
        let q = matching.atom.offset(offset).get_predicate();
        let clause = clause_storage.create_clause_with(
            Literal::new(false, Atom::Equality(p, q)),
            problem_clause
                .literals
                .into_iter()
                .filter(|id| *id != literal_id)
                .map(|id| problem_clause.literals[id].offset(offset)),
        );
        term_graph.extend_from(&problem_clause.term_graph);
        self.add_extension_constraints(solver, clause_storage, clause);

        record.lazy_extension(
            &problem.symbol_table,
            &term_graph,
            &clause_storage,
            self.clause.pending(),
            clause.open(),
        );
        Self::new(clause)
    }

    fn add_extension_constraints(
        &self,
        solver: &mut Solver,
        clause_storage: &ClauseStorage,
        clause: Clause,
    ) {
        for original_literal_id in self.clause.open() {
            for new_literal_id in clause.open().skip(1) {
                let original_literal = clause_storage[original_literal_id];
                let new_literal = clause_storage[new_literal_id];
                if original_literal.polarity == new_literal.polarity {
                    continue;
                }
                if original_literal.atom.is_predicate()
                    && new_literal.atom.is_predicate()
                {
                    solver.assert_not_equal(
                        original_literal.atom.get_predicate(),
                        new_literal.atom.get_predicate(),
                    );
                }
            }
        }
    }

    pub(crate) fn reflexivity<R: Record>(
        &mut self,
        record: &mut R,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        solver: &mut Solver,
    ) {
        let literal = clause_storage[self.clause.close_literal()];
        let (left, right) = literal.atom.get_equality();
        solver.assert_equal(left, right);
        record.reflexivity(
            &symbol_table,
            &term_graph,
            &clause_storage,
            self.clause.open(),
            left,
            right,
        );
    }
}
