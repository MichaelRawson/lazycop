use crate::clause::Clause;
use crate::constraint::Constraints;
use crate::infer;
use crate::prelude::*;
use crate::record::Record;
use crate::rule::*;

fn copy_lemmata(
    range: Range<Clause>,
    from: &LUT<Clause, Vec<Id<Literal>>>,
    to: &mut LUT<Clause, Vec<Id<Literal>>>,
) {
    if to.len() < from.len() {
        to.resize(from.len());
    }
    for id in range {
        to[id].clone_from(&from[id]);
    }
}

#[derive(Default)]
pub(crate) struct Tableau {
    literals: Literals,
    stack: Block<Clause>,
    valid: LUT<Literal, Id<Clause>>,
    lemmata: LUT<Clause, Vec<Id<Literal>>>,

    save_literals: Id<Literal>,
    save_stack: Block<Clause>,
    save_valid: LUT<Literal, Id<Clause>>,
    save_lemmata: LUT<Clause, Vec<Id<Literal>>>,
}

impl Tableau {
    pub(crate) fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    pub(crate) fn current_clause(&self) -> &Clause {
        some(self.stack.last())
    }

    pub(crate) fn num_open_branches(&self) -> u32 {
        self.stack
            .range()
            .into_iter()
            .map(|id| self.stack[id].open().len() as u32)
            .sum()
    }

    pub(crate) fn reduction_literals(
        &self,
    ) -> impl Iterator<Item = Id<Literal>> + '_ {
        self.path_literals().chain(
            self.stack
                .range()
                .into_iter()
                .flat_map(move |id| self.lemmata[id].iter().copied()),
        )
    }

    pub(crate) fn path_literals(
        &self,
    ) -> impl Iterator<Item = Id<Literal>> + '_ {
        self.stack
            .range()
            .into_iter()
            .rev()
            .skip(1)
            .map(move |id| self.stack[id].current_literal())
    }

    pub(crate) fn clear(&mut self) {
        self.literals.clear();
        self.stack.clear();
        self.valid.resize(Id::default());
    }

    pub(crate) fn save(&mut self) {
        self.save_literals = self.literals.len();
        self.save_stack.copy_from(&self.stack);
        self.save_valid.copy_from(&self.valid);
        copy_lemmata(
            self.stack.range(),
            &self.lemmata,
            &mut self.save_lemmata,
        );
    }

    pub(crate) fn restore(&mut self) {
        self.literals.truncate(self.save_literals);
        self.stack.copy_from(&self.save_stack);
        self.valid.copy_from(&self.save_valid);
        copy_lemmata(
            self.stack.range(),
            &self.save_lemmata,
            &mut self.lemmata,
        );
    }

    pub(crate) fn graph(
        &self,
        graph: &mut Graph,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &crate::binding::Bindings,
    ) {
        let add_clause = |graph: &mut Graph, id| -> Id<Node> {
            let clause = graph.clause();
            for open in self.stack[id].open() {
                let open =
                    self.literals[open].graph(graph, symbols, terms, bindings);
                graph.connect(clause, open)
            }
            for lemma in self.lemmata[id].iter().copied() {
                let lemma = self.literals[lemma]
                    .graph(graph, symbols, terms, bindings);
                graph.connect(lemma, clause);
            }
            clause
        };
        let mut clauses = self.stack.range().into_iter();
        let mut previous = add_clause(graph, some(clauses.next()));
        for next in clauses {
            let next = add_clause(graph, next);
            graph.connect(previous, next);
            previous = next;
        }
    }

    pub(crate) fn possible_rules<E: Extend<Rule>>(
        &self,
        possible: &mut E,
        problem: &Problem,
        terms: &Terms,
    ) {
        infer::rules(possible, self, problem, terms, &self.literals);
    }

    pub(crate) fn apply_rule<R: Record>(
        &mut self,
        record: &mut R,
        problem: &Problem,
        terms: &mut Terms,
        constraints: &mut Constraints,
        rule: &Rule,
    ) {
        match *rule {
            Rule::Start(start) => {
                let start = Clause::start(
                    record,
                    problem,
                    terms,
                    &mut self.literals,
                    constraints,
                    start,
                );
                self.push(start);
                self.close_branches();
            }
            Rule::Reflexivity => {
                some(self.stack.last_mut()).reflexivity(
                    record,
                    &problem.symbols,
                    terms,
                    &self.literals,
                    constraints,
                );
                self.close_branches();
            }
            Rule::Reduction(reduction) => {
                some(self.stack.last_mut()).reduction(
                    record,
                    &problem.symbols,
                    terms,
                    &mut self.literals,
                    constraints,
                    reduction,
                );
                self.reduction_validity(reduction.literal);
                self.close_branches();
            }
            Rule::LRForwardDemodulation(demodulation)
            | Rule::RLForwardDemodulation(demodulation) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                let consequence = some(self.stack.last_mut())
                    .forward_demodulation(
                        record,
                        &problem.symbols,
                        terms,
                        &mut self.literals,
                        constraints,
                        demodulation,
                        rule.is_l2r(),
                    );
                self.push(consequence);
                self.reduction_validity(demodulation.literal);
            }
            Rule::LRBackwardDemodulation(demodulation)
            | Rule::RLBackwardDemodulation(demodulation) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                let consequence = some(self.stack.last_mut())
                    .backward_demodulation(
                        record,
                        &problem.symbols,
                        terms,
                        &mut self.literals,
                        constraints,
                        demodulation,
                        rule.is_l2r(),
                    );
                self.push(consequence);
                self.reduction_validity(demodulation.literal);
            }
            Rule::StrictExtension(extension) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                let extension = some(self.stack.last_mut()).strict_extension(
                    record,
                    problem,
                    terms,
                    &mut self.literals,
                    constraints,
                    extension,
                );
                self.push(extension);
                self.close_branches();
            }
            Rule::LazyExtension(extension) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                let (extension, consequence) = some(self.stack.last_mut())
                    .lazy_extension(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        constraints,
                        extension,
                    );
                self.push(extension);
                self.extension_validity();
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                self.push(consequence);
                self.close_branches();
            }
            Rule::StrictBackwardParamodulation(paramodulation) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                let (extension, consequence) = some(self.stack.last_mut())
                    .strict_backward_paramodulation(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        constraints,
                        paramodulation,
                    );
                self.push(extension);
                self.extension_validity();
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                self.push(consequence);
            }
            Rule::LazyBackwardParamodulation(paramodulation) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                let (extension, consequence) = some(self.stack.last_mut())
                    .lazy_backward_paramodulation(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        constraints,
                        paramodulation,
                    );
                self.push(extension);
                self.extension_validity();
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                self.push(consequence);
            }
            Rule::VariableBackwardParamodulation(paramodulation) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                let (extension, consequence) = some(self.stack.last_mut())
                    .variable_backward_paramodulation(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        constraints,
                        paramodulation,
                    );
                self.push(extension);
                self.extension_validity();
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                self.push(consequence);
            }
            Rule::LRStrictForwardParamodulation(paramodulation)
            | Rule::RLStrictForwardParamodulation(paramodulation) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                let (extension, consequence) = some(self.stack.last_mut())
                    .strict_forward_paramodulation(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        constraints,
                        paramodulation,
                        rule.is_l2r(),
                    );
                self.push(extension);
                self.extension_validity();
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                self.push(consequence);
            }
            Rule::LRLazyForwardParamodulation(paramodulation)
            | Rule::RLLazyForwardParamodulation(paramodulation) => {
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                let (extension, consequence) = some(self.stack.last_mut())
                    .lazy_forward_paramodulation(
                        record,
                        problem,
                        terms,
                        &mut self.literals,
                        constraints,
                        paramodulation,
                        rule.is_l2r(),
                    );
                self.push(extension);
                self.extension_validity();
                self.add_regularity_constraints(
                    constraints,
                    terms,
                    &self.literals,
                );
                self.push(consequence);
            }
        }
    }

    fn push(&mut self, clause: Clause) {
        let id = self.stack.push(clause);
        if self.stack.len() > self.lemmata.len() {
            self.lemmata.resize(self.stack.len());
        }
        self.lemmata[id].clear();
    }

    fn extension_validity(&mut self) {
        let valid_in = self.stack.len() + Offset::new(-1);
        let literal = some(self.stack.last()).current_literal();
        self.valid.resize(self.literals.len());
        self.valid[literal] = valid_in;
    }

    fn reduction_validity(&mut self, reduction: Id<Literal>) {
        self.valid.resize(self.literals.len());
        let valid_in = self
            .stack
            .range()
            .into_iter()
            .find(|id| self.stack[*id].current_literal() == reduction)
            .map(|id| id + Offset::new(1))
            .unwrap_or(self.valid[reduction]);

        for affected in Range::new(valid_in, self.stack.len())
            .into_iter()
            .rev()
            .skip(1)
        {
            let literal = self.stack[affected].current_literal();
            let existing = self.valid[literal];
            self.valid[literal] = std::cmp::max(existing, valid_in);
        }
    }

    fn close_branches(&mut self) {
        let last = some(self.stack.last());
        if !last.is_empty() {
            return;
        }
        self.valid.resize(self.literals.len());
        self.stack.pop();
        while let Some(parent) = self.stack.last_mut() {
            let id = parent.close_literal();
            let valid_in = self.valid[id];
            let mut lemma = self.literals[id];
            lemma.polarity = !lemma.polarity;
            let lemma = self.literals.push(lemma);
            self.valid.resize(self.literals.len());
            self.valid[lemma] = valid_in;
            self.lemmata[valid_in].push(lemma);

            if !parent.is_empty() {
                return;
            }
            self.stack.pop();
        }
    }

    fn add_regularity_constraints(
        &self,
        constraints: &mut Constraints,
        terms: &Terms,
        literals: &Literals,
    ) {
        let current =
            &self.literals[some(self.stack.last()).current_literal()];

        for path in self
            .path_literals()
            .map(|id| &literals[id])
            .filter(|path| path.polarity == current.polarity)
        {
            current.add_disequation_constraints(constraints, terms, &path);
        }

        if current.is_predicate() {
            for reduction in self
                .reduction_literals()
                .map(|id| &literals[id])
                .filter(|reduction| reduction.polarity != current.polarity)
            {
                current.add_disequation_constraints(
                    constraints,
                    terms,
                    &reduction,
                );
            }
        } else if current.is_equality() && !current.polarity {
            current.add_reflexivity_constraints(constraints);
        }
    }
}
