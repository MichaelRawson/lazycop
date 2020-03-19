use crate::core::unification::unify;
use crate::output::record::Record;
use crate::prelude::*;

#[derive(Clone)]
pub struct Subgoal {
    pub path: Vec<Literal>,
    pub clause: Clause,
}

impl Subgoal {
    pub fn derived_goal(&self, clause: Clause) -> Self {
        let path = self.path.clone();
        Self { path, clause }
    }

    pub fn is_done(&self) -> bool {
        self.clause.is_empty()
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

    pub fn apply_lazy_extension<R: Record>(
        &mut self,
        record: &mut R,
        term_list: &mut TermList,
        problem: &Problem,
        clause_id: Id<Clause>,
        literal_id: Id<Literal>,
    ) -> Self {
        record.start_inference("lazy_extension");
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
            "lazy_extension",
            &[-2, -1],
            &problem.symbol_list,
            &term_list,
            &new_goal.clause,
        );
        record.conclusion(
            "lazy_extension",
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
        assert!(!literal.polarity);
        let (left, right) = match literal.atom {
            Atom::Equality(left, right) => (left, right),
            _ => unreachable!(),
        };

        if !unify(&problem.symbol_list, term_list, left, right) {
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

    pub fn possible_rules(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        term_list: &TermList,
    ) {
        let literal = self.clause.last_literal();
        self.possible_lazy_extensions(possible, problem, term_list, literal);
        self.possible_equality_reduction(
            possible, problem, term_list, literal,
        );
        self.possible_predicate_reductions(
            possible, problem, term_list, literal,
        );
    }

    fn possible_lazy_extensions<'a>(
        &'a self,
        possible: &mut Vec<Rule>,
        problem: &'a Problem,
        term_list: &TermList,
        literal: &Literal,
    ) {
        if let Atom::Predicate(predicate) = literal.atom {
            possible.extend(
                problem
                    .index
                    .query_lazy_predicates(
                        &problem.symbol_list,
                        &term_list,
                        !literal.polarity,
                        predicate,
                    )
                    .map(|(clause, literal)| {
                        Rule::LazyExtension(clause, literal)
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
