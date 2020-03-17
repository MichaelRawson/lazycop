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
            Atom::Predicate(p) => {
                possible.extend(
                    problem
                        .index
                        .query_predicate(
                            &problem.symbol_list,
                            &self.term_list,
                            !literal.polarity,
                            p,
                        )
                        .map(|(clause, literal)| {
                            Rule::ExtendPredicate(clause, literal)
                        }),
                );
            }
            Atom::Equality(_left, _right) => {}
        }

        possible
    }

    fn start<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        clause_id: Id<Clause>,
    ) {
        assert!(self.subgoals.is_empty());
        assert!(self.term_list.is_empty());
        let (clause, clause_term_list) = problem.get_clause(clause_id);
        self.term_list.copy_from(clause_term_list);
        record.start(&problem.symbol_list, &self.term_list, &clause);
        self.subgoals.push(Subgoal::start(clause));
    }

    fn extend_predicate<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        clause_id: Id<Clause>,
        literal_id: Id<Literal>,
    ) {
        assert!(!self.subgoals.is_empty());
        assert!(!self.term_list.is_empty());
        let mut extension = self.copy_clause(problem, clause_id);
        let subgoal = self.subgoals.last_mut().unwrap();
        let literal = subgoal.pop_literal();
        let matching = extension.remove_literal(literal_id);

        let extension = Subgoal::with_path(subgoal, extension);
        if subgoal.is_done() {
            self.subgoals.pop();
        }

        assert_ne!(literal.polarity, matching.polarity);
        let eq_goal = match (literal.atom, matching.atom) {
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
                            .map(|eq| Literal::new(false, eq))
                            .collect();
                        let clause = Clause::new(eqs);
                        Subgoal::with_path(&extension, clause)
                    }
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        };

        if !extension.is_done() {
            self.subgoals.push(extension);
        }
        if !eq_goal.is_done() {
            self.subgoals.push(eq_goal);
        }
    }

    fn copy_clause(
        &mut self,
        problem: &Problem,
        clause_id: Id<Clause>,
    ) -> Clause {
        let offset = self.term_list.current_offset();
        let (mut clause, clause_term_list) = problem.get_clause(clause_id);
        self.term_list.copy_from(clause_term_list);
        clause.offset(offset);
        clause
    }
}
