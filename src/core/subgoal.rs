use crate::output::record::Record;
use crate::prelude::*;

#[derive(Clone)]
pub struct Subgoal {
    path: Path,
    clause: Clause,
    lemmas: Vec<Literal>,
}

impl Subgoal {
    pub fn num_literals(&self) -> usize {
        self.clause.len()
    }

    pub fn is_done(&self) -> bool {
        self.clause.is_empty()
    }

    pub fn is_regular(
        &self,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
    ) -> bool {
        self.clause.iter().all(|literal| {
            !self.path.contains(symbol_table, term_graph, literal)
        })
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
        let lemmas = vec![];
        record.axiom(&problem.symbol_table, &term_graph, &clause);
        record.end_inference();
        Self {
            path,
            clause,
            lemmas,
        }
    }

    pub fn apply_extension<P: Policy, R: Record>(
        &mut self,
        record: &mut R,
        term_graph: &mut TermGraph,
        problem: &Problem,
        coordinate: Id<(Clause, Literal)>,
    ) -> Option<(Self, Self)> {
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
        let path = Path::based_on(&self.path, current_literal);
        let lemmas = self.lemmas.clone();
        let matching_literal = clause.remove_literal(literal_id);
        let extension_goal = Self {
            path,
            clause,
            lemmas,
        };

        let clause = Clause::new(current_literal.resolve_or_disequations(
            &problem.symbol_table,
            term_graph,
            &matching_literal,
        ));
        let path = self.path.clone();
        let lemmas = self.lemmas.clone();
        let eq_goal = Self {
            path,
            clause,
            lemmas,
        };

        self.lemmas.push(current_literal);
        record.conclusion(
            "extension_original",
            &[-1, -1],
            &problem.symbol_table,
            &term_graph,
            &self.clause,
        );
        record.conclusion(
            "extension_new",
            &[-2, -1],
            &problem.symbol_table,
            &term_graph,
            &extension_goal.clause,
        );
        record.conclusion(
            "extension_disequalities",
            &[-2, -1],
            &problem.symbol_table,
            &term_graph,
            &eq_goal.clause,
        );
        record.end_inference();
        Some((extension_goal, eq_goal))
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
        record.lemma(&problem.symbol_table, term_graph, matching);
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

    pub fn apply_lemma<P: Policy, R: Record>(
        &mut self,
        record: &mut R,
        term_graph: &mut TermGraph,
        problem: &Problem,
        lemma_id: Id<Literal>,
    ) {
        record.start_inference("lemma");
        let lemma = &self.lemmas[lemma_id.index()];
        record.premise(&problem.symbol_table, term_graph, &self.clause);
        record.lemma(&problem.symbol_table, term_graph, lemma);
        self.clause.pop_literal();
        record.conclusion(
            "lemma",
            &[-1],
            &problem.symbol_table,
            &term_graph,
            &self.clause,
        );
        record.end_inference();
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
        self.possible_lemmas(possible, problem, term_graph, literal);
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
                    let path_id = path_index.into();
                    possible.push(Rule::Reduction(path_id));
                }
            }
        }
    }

    fn possible_lemmas(
        &self,
        possible: &mut Vec<Rule>,
        problem: &Problem,
        term_graph: &TermGraph,
        literal: &Literal,
    ) {
        for (lemma_index, lemma) in self.lemmas.iter().enumerate() {
            if literal.equal(&problem.symbol_table, &term_graph, lemma) {
                let lemma_id = lemma_index.into();
                possible.push(Rule::Lemma(lemma_id));
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
