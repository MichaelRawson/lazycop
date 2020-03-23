use crate::output::record::Record;
use crate::prelude::*;

#[derive(Clone)]
pub struct Subgoal {
    path: Path,
    clause: Clause,
}

impl Subgoal {
    pub fn is_done(&self) -> bool {
        self.clause.is_empty()
    }

    pub fn num_literals(&self) -> usize {
        self.clause.len()
    }

    pub fn start<R: Record>(
        record: &mut R,
        term_graph: &mut TermGraph,
        problem: &Problem,
        clause_id: Id<Clause>,
    ) -> Self {
        record.start_inference("start");
        let path = Path::default();
        let clause = problem.copy_clause_into(term_graph, clause_id);
        record.axiom(&problem.symbol_table, &term_graph, &clause);
        record.end_inference();
        Self { path, clause }
    }

    pub fn apply_extension<R: Record>(
        &mut self,
        record: &mut R,
        term_graph: &mut TermGraph,
        problem: &Problem,
        coordinate: Id<(Clause, Literal)>,
    ) -> Self {
        record.start_inference("extension");
        record.premise(&problem.symbol_table, term_graph, &self.clause);
        let current_literal = self.clause.pop_literal();
        let (clause_id, literal_id) = problem.index.query_predicates(
            &problem.symbol_table,
            &term_graph,
            !current_literal.polarity,
            current_literal.predicate_term(),
        )[coordinate.index()];

        let mut clause = problem.copy_clause_into(term_graph, clause_id);
        record.axiom(&problem.symbol_table, term_graph, &clause);
        let matching_literal = clause.remove_literal(literal_id);
        self.clause.extend(current_literal.resolve_or_disequations(
            &problem.symbol_table,
            term_graph,
            &matching_literal,
        ));
        let path = Path::based_on(&self.path, current_literal);
        let new_goal = Self { path, clause };
        record.conclusion(
            "extension",
            &[-2, -1],
            &problem.symbol_table,
            &term_graph,
            &new_goal.clause,
        );
        record.conclusion(
            "extension",
            &[-3, -2],
            &problem.symbol_table,
            &term_graph,
            &self.clause,
        );
        record.end_inference();
        new_goal
    }

    pub fn apply_reduction<P: Policy, R: Record>(
        &mut self,
        record: &mut R,
        term_graph: &mut TermGraph,
        problem: &Problem,
        path_id: Id<Literal>,
    ) -> bool {
        record.start_inference("reduction");
        record.premise(&problem.symbol_table, term_graph, &self.clause);
        let matching = &self.path[path_id];
        let literal = self.clause.pop_literal();
        if !literal.resolve::<P>(&problem.symbol_table, term_graph, matching) {
            return false;
        }
        record.conclusion(
            "reduction",
            &[-1],
            &problem.symbol_table,
            term_graph,
            &self.clause,
        );
        record.end_inference();
        true
    }

    pub fn apply_symmetry<P: Policy, R: Record>(
        &mut self,
        record: &mut R,
        term_graph: &mut TermGraph,
        problem: &Problem,
    ) -> bool {
        record.start_inference("symmetry");
        record.premise(&problem.symbol_table, term_graph, &self.clause);
        let literal = self.clause.pop_literal();
        if !literal.equality_unify::<P>(&problem.symbol_table, term_graph) {
            return false;
        }
        record.conclusion(
            "symmetry",
            &[-1],
            &problem.symbol_table,
            &term_graph,
            &self.clause,
        );
        record.end_inference();
        true
    }

    pub fn possible_rules(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        term_graph: &TermGraph,
    ) {
        let literal = self.clause.last_literal();
        self.possible_extensions(possible, problem, term_graph, literal);
        self.possible_reductions(possible, problem, term_graph, literal);
        self.possible_symmetry(possible, problem, term_graph, literal);
    }

    fn possible_extensions<'a>(
        &'a self,
        possible: &mut Vec<Rule>,
        problem: &'a Problem,
        term_graph: &TermGraph,
        literal: &Literal,
    ) {
        if literal.is_predicate() {
            let num_results = problem
                .index
                .query_predicates(
                    &problem.symbol_table,
                    &term_graph,
                    !literal.polarity,
                    literal.predicate_term(),
                )
                .len();
            possible.extend(
                (0..num_results).map(|index| Rule::Extension(index.into())),
            );
        }
    }

    fn possible_reductions(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        term_graph: &TermGraph,
        literal: &Literal,
    ) {
        if literal.is_predicate() {
            for (path_index, path_literal) in self.path.literals().enumerate()
            {
                if literal.might_resolve(
                    &problem.symbol_table,
                    &term_graph,
                    path_literal,
                ) {
                    let path_index = path_index.into();
                    possible.push(Rule::Reduction(path_index));
                }
            }
        }
    }

    fn possible_symmetry(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        term_graph: &TermGraph,
        literal: &Literal,
    ) {
        if literal.might_equality_unify(&problem.symbol_table, &term_graph) {
            possible.push(Rule::Symmetry);
        }
    }
}
