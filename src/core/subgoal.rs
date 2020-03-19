use crate::output::record::Record;
use crate::prelude::*;

#[derive(Clone)]
pub struct Subgoal {
    path: Vec<Literal>,
    clause: Clause,
}

impl Subgoal {
    pub fn derived_goal(&self, clause: Clause) -> Self {
        let path = self.path.clone();
        Self { path, clause }
    }

    pub fn is_done(&self) -> bool {
        self.clause.is_empty()
    }

    pub fn num_literals(&self) -> usize {
        self.clause.len()
    }

    pub fn start<R: Record>(
        record: &mut R,
        term_list: &mut TermList,
        problem: &Problem,
        clause_id: Id<Clause>,
    ) -> Self {
        record.start_inference("start");
        let path = vec![];
        let clause = problem.copy_clause_into(term_list, clause_id);
        record.axiom(&problem.symbol_list, &term_list, &clause);
        record.end_inference();
        Self { path, clause }
    }

    pub fn apply_lazy_predicate_extension<R: Record>(
        &mut self,
        record: &mut R,
        term_list: &mut TermList,
        problem: &Problem,
        clause_id: Id<Clause>,
        literal_id: Id<Literal>,
    ) -> Self {
        record.start_inference("lazy_predicate_extension");
        record.premise(&problem.symbol_list, term_list, &self.clause);
        let current_literal = self.clause.pop_literal();
        let mut extension_clause =
            problem.copy_clause_into(term_list, clause_id);
        record.axiom(&problem.symbol_list, term_list, &extension_clause);
        let matching_literal = extension_clause.remove_literal(literal_id);
        self.clause.extend(current_literal.lazy_disequalities(
            &problem.symbol_list,
            term_list,
            &matching_literal,
        ));
        let mut new_goal = self.derived_goal(extension_clause);
        new_goal.path.push(current_literal);
        record.conclusion(
            "lazy_predicate_extension",
            &[-2, -1],
            &problem.symbol_list,
            &term_list,
            &new_goal.clause,
        );
        record.conclusion(
            "lazy_predicate_extension",
            &[-3, -2],
            &problem.symbol_list,
            &term_list,
            &self.clause,
        );
        record.end_inference();
        new_goal
    }

    pub fn apply_equality_reduction<R: Record>(
        &mut self,
        record: &mut R,
        term_list: &mut TermList,
        problem: &Problem,
    ) -> bool {
        record.start_inference("equality_reduction");
        record.premise(&problem.symbol_list, term_list, &self.clause);
        let literal = self.clause.pop_literal();
        if !literal.equality_reduce(&problem.symbol_list, term_list) {
            return false;
        }
        record.conclusion(
            "equality_reduction",
            &[-1],
            &problem.symbol_list,
            &term_list,
            &self.clause,
        );
        record.end_inference();
        true
    }

    pub fn apply_predicate_reduction<R: Record>(
        &mut self,
        record: &mut R,
        term_list: &mut TermList,
        problem: &Problem,
        path_id: Id<Literal>,
    ) -> bool {
        record.start_inference("predicate_reduction");
        record.premise(&problem.symbol_list, term_list, &self.clause);
        let matching = &self.path[path_id.index()];
        let literal = self.clause.pop_literal();
        if !literal.resolve(&problem.symbol_list, term_list, matching) {
            return false;
        }
        record.conclusion(
            "predicate_reduction",
            &[-1],
            &problem.symbol_list,
            term_list,
            &self.clause,
        );
        record.end_inference();
        true
    }

    pub fn possible_rules(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        term_list: &TermList,
    ) {
        let literal = self.clause.last_literal();
        self.possible_lazy_predicate_extensions(
            possible, problem, term_list, literal,
        );
        self.possible_equality_reduction(
            possible, problem, term_list, literal,
        );
        self.possible_predicate_reductions(
            possible, problem, term_list, literal,
        );
    }

    fn possible_lazy_predicate_extensions<'a>(
        &'a self,
        possible: &mut Vec<Rule>,
        problem: &'a Problem,
        term_list: &TermList,
        literal: &Literal,
    ) {
        if literal.is_predicate() {
            possible.extend(
                problem
                    .index
                    .query_lazy_predicates(
                        &problem.symbol_list,
                        &term_list,
                        !literal.polarity,
                        literal.predicate_term(),
                    )
                    .map(|(clause, literal)| {
                        Rule::LazyPredicateExtension(clause, literal)
                    }),
            );
        }
    }

    fn possible_equality_reduction(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        term_list: &TermList,
        literal: &Literal,
    ) {
        if literal.might_equality_reduce(&problem.symbol_list, &term_list) {
            possible.push(Rule::EqualityReduction);
        }
    }

    fn possible_predicate_reductions(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        term_list: &TermList,
        literal: &Literal,
    ) {
        if literal.is_predicate() {
            for (path_index, path_literal) in self.path.iter().enumerate() {
                if literal.might_resolve(
                    &problem.symbol_list,
                    &term_list,
                    path_literal,
                ) {
                    let path_index = path_index.into();
                    possible.push(Rule::PredicateReduction(path_index));
                }
            }
        }
    }
}
