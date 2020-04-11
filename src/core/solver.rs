use crate::io::record::Record;
use crate::prelude::*;
use crate::util::id_map::IdMap;

#[derive(Default)]
pub(crate) struct Solver {
    bindings: IdMap<Variable, Option<Id<Term>>>,
    pairs: Vec<(Id<Term>, Id<Term>)>,
    terms: Vec<Id<Term>>,
}

impl Solver {
    pub(crate) fn solve<R: Record>(
        &mut self,
        record: &mut R,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        equalities: &[(Id<Term>, Id<Term>)],
    ) -> bool {
        self.bindings.ensure_capacity(term_graph.len().transmute());
        self.bindings.wipe();
        self.pairs.clear();
        self.pairs.extend_from_slice(equalities);
        if !self.solve_equalities(term_graph) {
            return false;
        }
        record.unification(symbol_table, term_graph, &self.bindings);
        true
    }

    fn solve_equalities(&mut self, term_graph: &TermGraph) -> bool {
        while let Some((left, right)) = self.pairs.pop() {
            let (left, lview) = self.view(term_graph, left);
            let (right, rview) = self.view(term_graph, right);
            if left == right {
                continue;
            }

            match (lview, rview) {
                (TermView::Variable(x), TermView::Variable(_)) => {
                    self.bindings[x] = Some(right);
                }
                (_, TermView::Variable(_)) => {
                    self.pairs.push((right, left));
                }
                (TermView::Variable(x), _) => {
                    if self.occurs(term_graph, x, right) {
                        return false;
                    }
                    self.bindings[x] = Some(right);
                }
                (TermView::Function(f, ts), TermView::Function(g, ss))
                    if f == g =>
                {
                    self.pairs.extend(ts.zip(ss));
                }
                _ => {
                    return false;
                }
            }
        }
        true
    }

    fn occurs(
        &mut self,
        term_graph: &TermGraph,
        x: Id<Variable>,
        term: Id<Term>,
    ) -> bool {
        self.terms.clear();
        self.terms.push(term);
        while let Some(term) = self.terms.pop() {
            let (_, view) = self.view(term_graph, term);
            match view {
                TermView::Variable(y) if x == y => {
                    return true;
                }
                TermView::Variable(_) => {}
                TermView::Function(_, ts) => {
                    self.terms.extend(ts);
                }
            }
        }
        false
    }

    fn view(
        &self,
        term_graph: &TermGraph,
        mut term: Id<Term>,
    ) -> (Id<Term>, TermView) {
        loop {
            match term_graph.view(term) {
                TermView::Variable(x) => {
                    if let Some(next) = self.bindings[x] {
                        term = next;
                    } else {
                        return (x.transmute(), TermView::Variable(x));
                    }
                }
                view => {
                    return (term, view);
                }
            }
        }
    }
}
