use crate::io::record::Record;
use crate::prelude::*;
use crate::util::id_map::IdMap;

#[derive(Default)]
pub struct Solver {
    bindings: IdMap<Variable, Id<Term>>,
    pairs: Vec<(Id<Term>, Id<Term>)>,
    terms: Vec<Id<Term>>,
}

impl Solver {
    pub fn solve<R: Record>(
        &mut self,
        record: &mut R,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        equalities: &[(Id<Term>, Id<Term>)],
        disequalities: &[(Id<Term>, Id<Term>)],
    ) -> bool {
        self.bindings.clear();
        self.pairs.clear();
        self.pairs.extend_from_slice(equalities);
        record.start_inference("constraint solving");
        for (left, right) in self.pairs.iter().copied() {
            record.equality_constraint(symbol_table, term_graph, left, right);
        }
        record.therefore();
        if !self.solve_equalities(record, symbol_table, term_graph) {
            return false;
        }
        record.end_inference();

        for (left, right) in disequalities.iter().copied() {
            self.pairs.push((left, right));
            if !self.check_disequality(term_graph) {
                return false;
            }
        }
        true
    }

    fn solve_equalities<R: Record>(
        &mut self,
        record: &mut R,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
    ) -> bool {
        while let Some((left, right)) = self.pairs.pop() {
            let (left, lview) = self.view(term_graph, left);
            let (right, rview) = self.view(term_graph, right);
            if left == right {
                continue;
            }

            match (lview, rview) {
                (TermView::Variable(x), TermView::Variable(_)) => {
                    self.bind(record, symbol_table, term_graph, x, right);
                }
                (TermView::Variable(x), _) => {
                    if self.occurs(term_graph, x, right) {
                        return false;
                    }
                    self.bind(record, symbol_table, term_graph, x, right);
                }
                (_, TermView::Variable(x)) => {
                    if self.occurs(term_graph, x, left) {
                        return false;
                    }
                    self.bind(record, symbol_table, term_graph, x, left);
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

    #[inline]
    fn check_disequality(&mut self, term_graph: &TermGraph) -> bool {
        while let Some((left, right)) = self.pairs.pop() {
            let (left, lview) = self.view(term_graph, left);
            let (right, rview) = self.view(term_graph, right);
            if left == right {
                continue;
            }
            match (lview, rview) {
                (TermView::Function(f, ts), TermView::Function(g, ss))
                    if f == g =>
                {
                    self.pairs.extend(ts.zip(ss));
                }
                _ => {
                    return true;
                }
            }
        }
        false
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

    fn bind<R: Record>(
        &mut self,
        record: &mut R,
        symbol_table: &SymbolTable,
        term_graph: &TermGraph,
        variable: Id<Variable>,
        term: Id<Term>,
    ) {
        record.binding(symbol_table, term_graph, variable, term);
        self.bindings.set(variable, term);
    }

    fn view(
        &self,
        term_graph: &TermGraph,
        mut term: Id<Term>,
    ) -> (Id<Term>, TermView) {
        loop {
            match term_graph.view(term) {
                TermView::Variable(x) => {
                    if let Some(next) = self.bindings.get(x) {
                        term = *next;
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
