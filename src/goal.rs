use crate::io::record::Record;
use crate::prelude::*;

#[derive(Clone)]
pub(crate) struct Goal {
    clause: Clause,
    valid_from: Id<Goal>,
    lemmata: Vec<Id<Literal>>,
}

impl Goal {
    pub(crate) fn new(clause: Clause) -> Self {
        let valid_from = Id::default();
        let lemmata = vec![];
        Self {
            clause,
            valid_from,
            lemmata,
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

    pub(crate) fn add_lemmatum(&mut self, literal: Id<Literal>) {
        self.lemmata.push(literal);
    }

    pub(crate) fn available_lemmata(
        &self,
    ) -> impl Iterator<Item = Id<Literal>> + '_ {
        self.lemmata.iter().copied()
    }

    pub(crate) fn start<R: Record>(
        record: &mut R,
        problem: &Problem,
        term_graph: &mut TermGraph,
        clause_storage: &mut ClauseStorage,
        solver: &mut Solver,
        start_clause: Id<ProblemClause>,
    ) -> Self {
        let offset = term_graph.current_offset();
        let problem_clause = &problem[start_clause];
        let clause = clause_storage.create_clause(
            problem_clause
                .literals
                .as_ref()
                .iter()
                .map(|literal| literal.offset(offset)),
        );
        term_graph.extend_from(&problem_clause.term_graph);
        Self::add_tautology_constraints(
            solver,
            term_graph,
            clause_storage,
            clause,
        );
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
        matching_id: Id<Literal>,
    ) -> Self {
        let (p, q, mut clause) = self.extension_common(
            problem,
            term_graph,
            clause_storage,
            solver,
            clause_id,
            matching_id,
        );
        solver.assert_equal(p, q);
        clause.close_literal();

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
        matching_id: Id<Literal>,
    ) -> Self {
        let lazy_index = clause_storage.len();
        let (p, q, clause) = self.extension_common(
            problem,
            term_graph,
            clause_storage,
            solver,
            clause_id,
            matching_id,
        );
        let lazy_unification = Literal::new(false, Atom::Equality(p, q));
        clause_storage[lazy_index] = lazy_unification;

        record.lazy_extension(
            &problem.symbol_table,
            &term_graph,
            &clause_storage,
            self.clause.pending(),
            clause.open(),
        );
        Self::new(clause)
    }

    fn extension_common(
        &mut self,
        problem: &Problem,
        term_graph: &mut TermGraph,
        clause_storage: &mut ClauseStorage,
        solver: &mut Solver,
        clause_id: Id<ProblemClause>,
        matching_id: Id<Literal>,
    ) -> (Id<Term>, Id<Term>, Clause) {
        let offset = term_graph.current_offset();
        let literal = clause_storage[self.current_literal()];
        let problem_clause = &problem[clause_id];
        let matching = problem_clause.literals[matching_id].offset(offset);
        let p = literal.atom.get_predicate();
        let q = matching.atom.get_predicate();
        let clause = clause_storage.create_clause_with(
            matching,
            problem_clause
                .literals
                .range()
                .filter(|id| *id != matching_id)
                .map(|id| problem_clause.literals[id].offset(offset)),
        );
        term_graph.extend_from(&problem_clause.term_graph);
        Self::add_tautology_constraints(
            solver,
            term_graph,
            clause_storage,
            clause,
        );
        self.add_strong_connection_constraints(
            solver,
            term_graph,
            clause_storage,
            clause,
        );

        (p, q, clause)
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

    fn add_strong_connection_constraints(
        &self,
        solver: &mut Solver,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        clause: Clause,
    ) {
        for original in self.clause.open().map(|id| &clause_storage[id]) {
            for new in clause.open().skip(1).map(|id| &clause_storage[id]) {
                if original.polarity != new.polarity {
                    original.atom.add_disequation_constraints(
                        solver, term_graph, &new.atom,
                    );
                }
            }
        }
    }

    fn add_tautology_constraints(
        solver: &mut Solver,
        term_graph: &TermGraph,
        clause_storage: &ClauseStorage,
        clause: Clause,
    ) {
        let literals = clause.open();
        for literal_id in literals {
            let literal = clause_storage[literal_id];
            if literal.polarity {
                literal.atom.add_positive_constraints(solver)
            }

            let mut others = literals;
            others.next();
            for other in others.map(|id| &clause_storage[id]) {
                if literal.polarity != other.polarity {
                    literal.atom.add_disequation_constraints(
                        solver,
                        term_graph,
                        &other.atom,
                    );
                }
            }
        }
    }
}
