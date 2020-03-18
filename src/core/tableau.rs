use crate::core::unification::{might_unify, unify};
use crate::output::record::Record;
use crate::prelude::*;

#[derive(Default)]
pub struct Tableau {
    blocked: bool,
    term_list: TermList,
    subgoals: Vec<Subgoal>,
}

impl Tableau {
    pub fn is_closed(&self) -> bool {
        !self.blocked && self.subgoals.is_empty()
    }

    pub fn num_subgoals(&self) -> u32 {
        self.subgoals.len() as u32
    }

    pub fn clear(&mut self) {
        self.term_list.clear();
        self.subgoals.clear();
    }

    pub fn reconstruct<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        script: &[Rule],
    ) {
        for rule in script {
            match *rule {
                Rule::Start(clause_id) => {
                    self.start(record, problem, clause_id);
                }
                Rule::EqualityReduction => {
                    self.equality_reduction(record, problem);
                }
                Rule::PredicateExtension(clause_id, literal_id) => {
                    self.predicate_extension(
                        record, problem, clause_id, literal_id,
                    );
                }
            }
        }
    }

    pub fn possible_rules(&self, problem: &Problem) -> Vec<Rule> {
        if self.blocked {
            return vec![];
        }

        let mut possible = vec![];
        assert!(!self.is_closed());
        let subgoal = self.subgoals.last().unwrap();
        assert!(!subgoal.is_done());
        let literal = *subgoal.current_literal().unwrap();

        match literal.atom {
            Atom::Predicate(predicate) => {
                self.possible_predicate_extensions(
                    &mut possible,
                    problem,
                    literal.polarity,
                    predicate,
                );
            }
            Atom::Equality(left, right) => {
                if !literal.polarity {
                    self.possible_equality_reduction(
                        &mut possible,
                        problem,
                        left,
                        right,
                    );
                }
            }
        }

        possible
    }

    fn possible_predicate_extensions(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        polarity: bool,
        predicate: Id<Term>,
    ) {
        possible.extend(
            problem
                .index
                .query_predicate(
                    &problem.symbol_list,
                    &self.term_list,
                    !polarity,
                    predicate,
                )
                .map(|(clause, literal)| {
                    Rule::PredicateExtension(clause, literal)
                }),
        );
    }

    fn possible_equality_reduction(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        left: Id<Term>,
        right: Id<Term>,
    ) {
        if might_unify(&problem.symbol_list, &self.term_list, left, right) {
            possible.push(Rule::EqualityReduction);
        }
    }

    fn start<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        clause_id: Id<Clause>,
    ) {
        record.start_inference("start");
        assert!(self.subgoals.is_empty());
        assert!(self.term_list.is_empty());
        let clause = self.copy_clause(record, problem, clause_id);
        self.subgoals.push(Subgoal::start(clause));
        record.end_inference();
    }

    fn equality_reduction<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
    ) {
        record.start_inference("equality_reduction");
        assert!(!self.subgoals.is_empty());
        let mut subgoal = self.subgoals.pop().unwrap();
        record.premise(&problem.symbol_list, &self.term_list, &subgoal.clause);

        assert!(!subgoal.is_done());
        let literal = subgoal.pop_literal().unwrap();
        assert!(!literal.polarity);
        let (left, right) = match literal.atom {
            Atom::Equality(left, right) => (left, right),
            _ => unreachable!(),
        };

        if !unify(&problem.symbol_list, &mut self.term_list, left, right) {
            self.blocked = true;
            return;
        }

        record.conclusion(
            "equality_reduction",
            &[-1],
            &problem.symbol_list,
            &self.term_list,
            &subgoal.clause,
        );

        if !subgoal.is_done() {
            self.subgoals.push(subgoal);
        }
        record.end_inference();
    }

    fn predicate_extension<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        clause_id: Id<Clause>,
        literal_id: Id<Literal>,
    ) {
        record.start_inference("predicate_extension");
        assert!(!self.subgoals.is_empty());
        let mut subgoal = self.subgoals.pop().unwrap();
        record.premise(&problem.symbol_list, &self.term_list, &subgoal.clause);
        let mut extension_clause =
            self.copy_clause(record, problem, clause_id);

        assert!(!subgoal.is_done());
        let literal = subgoal.pop_literal().unwrap();
        let matching = extension_clause.remove_literal(literal_id);
        let mut new_goal = Subgoal::with_path(&subgoal, extension_clause);
        new_goal.push_path(literal);
        assert_ne!(literal.polarity, matching.polarity);

        match (literal.atom, matching.atom) {
            (Atom::Predicate(p), Atom::Predicate(q)) => {
                let p = self.term_list.view(&problem.symbol_list, p);
                let q = self.term_list.view(&problem.symbol_list, q);
                match (p, q) {
                    (
                        TermView::Function(p, pargs),
                        TermView::Function(q, qargs),
                    ) => {
                        assert!(p == q);
                        assert!(pargs.len() == qargs.len());
                        let eqs = pargs
                            .zip(qargs)
                            .map(|(t, s)| Atom::Equality(t, s))
                            .map(|eq| Literal::new(false, eq));
                        subgoal.extend_clause(eqs);
                    }
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        };

        record.conclusion(
            "predicate_extension",
            &[-2, -1],
            &problem.symbol_list,
            &self.term_list,
            &new_goal.clause,
        );
        record.conclusion(
            "predicate_extension",
            &[-3, -2],
            &problem.symbol_list,
            &self.term_list,
            &subgoal.clause,
        );

        if !new_goal.is_done() {
            self.subgoals.push(new_goal);
        }
        if !subgoal.is_done() {
            self.subgoals.push(subgoal);
        }
        record.end_inference();
    }

    fn copy_clause<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        clause_id: Id<Clause>,
    ) -> Clause {
        let offset = self.term_list.current_offset();
        let (mut clause, clause_term_list) = problem.get_clause(clause_id);
        self.term_list.copy_from(clause_term_list);
        clause.offset(offset);
        record.axiom(&problem.symbol_list, &self.term_list, &clause);
        clause
    }
}
