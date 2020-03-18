use crate::output::record::Record;
use crate::prelude::*;

#[derive(Default)]
pub struct Tableau {
    term_list: TermList,
    subgoals: Vec<Subgoal>,
}

impl Tableau {
    pub fn is_closed(&self) -> bool {
        self.subgoals.is_empty()
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
                Rule::ExtendPredicate(clause_id, literal_id) => {
                    self.extend_predicate(
                        record, problem, clause_id, literal_id,
                    );
                }
            }
        }
    }

    pub fn possible_rules(&self, problem: &Problem) -> Vec<Rule> {
        let mut possible = vec![];
        assert!(!self.is_closed());
        let subgoal = self.subgoals.last().unwrap();
        assert!(!subgoal.is_done());
        let literal = *subgoal.current_literal().unwrap();

        match literal.atom {
            Atom::Predicate(predicate) => {
                self.possible_extend_predicate(
                    &mut possible,
                    problem,
                    literal.polarity,
                    predicate,
                );
            }
            Atom::Equality(_left, _right) => {}
        }

        possible
    }

    fn possible_extend_predicate(
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
                    Rule::ExtendPredicate(clause, literal)
                }),
        );
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

    fn extend_predicate<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        clause_id: Id<Clause>,
        literal_id: Id<Literal>,
    ) {
        record.start_inference("extend_predicate");
        assert!(!self.subgoals.is_empty());
        assert!(!self.term_list.is_empty());
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
            "extend_predicate",
            &[-2, -1],
            &problem.symbol_list,
            &self.term_list,
            &new_goal.clause,
        );
        record.conclusion(
            "extend_predicate",
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
