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
        disequalities: &[(Atom, Atom)],
    ) -> bool {
        self.bindings.ensure_capacity(term_graph.len().transmute());
        self.bindings.wipe();

        if !self.solve_equalities(term_graph, equalities)
            || !self.solve_disequalities(term_graph, disequalities)
        {
            self.pairs.clear();
            self.terms.clear();
            return false;
        }
        record.unification(symbol_table, term_graph, &self.bindings);
        true
    }

    fn solve_equalities(
        &mut self,
        term_graph: &TermGraph,
        equalities: &[(Id<Term>, Id<Term>)],
    ) -> bool {
        self.pairs.extend_from_slice(equalities);
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
                (TermView::Variable(x), _) => {
                    if self.occurs(term_graph, x, right) {
                        return false;
                    }
                    self.bindings[x] = Some(right);
                }
                (_, TermView::Variable(x)) => {
                    if self.occurs(term_graph, x, left) {
                        return false;
                    }
                    self.bindings[x] = Some(left);
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

    fn solve_disequalities(
        &mut self,
        term_graph: &TermGraph,
        disequalities: &[(Atom, Atom)],
    ) -> bool {
        for (left, right) in disequalities {
            match (left, right) {
                (Atom::Predicate(p), Atom::Predicate(q)) => {
                    if !self.solve_disequality(term_graph, *p, *q) {
                        return false;
                    }
                }
                (Atom::Equality(l1, r1), Atom::Equality(l2, r2)) => {
                    if self.solve_disequality(term_graph, *l1, *l2)
                        && self.solve_disequality(term_graph, *r1, *r2)
                    {
                        return false;
                    }
                    if self.solve_disequality(term_graph, *l1, *r2)
                        && self.solve_disequality(term_graph, *l2, *r1)
                    {
                        return false;
                    }
                }
                _ => {}
            }
        }
        true
    }

    fn solve_disequality(
        &mut self,
        term_graph: &TermGraph,
        left: Id<Term>,
        right: Id<Term>,
    ) -> bool {
        self.pairs.push((left, right));
        while let Some((left, right)) = self.pairs.pop() {
            let (left, lview) = self.view(term_graph, left);
            let (right, rview) = self.view(term_graph, right);
            if left == right {
                continue;
            }

            if let (TermView::Function(f, ts), TermView::Function(g, ss)) =
                (lview, rview)
            {
                if f == g {
                    self.pairs.extend(ts.zip(ss));
                    continue;
                }
            }
            self.pairs.clear();
            return true;
        }
        false
    }

    fn occurs(
        &mut self,
        term_graph: &TermGraph,
        x: Id<Variable>,
        term: Id<Term>,
    ) -> bool {
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
            let (id, view) = term_graph.view(term);
            if let TermView::Variable(x) = view {
                if let Some(next) = self.bindings[x] {
                    term = next;
                    continue;
                }
            }
            return (id, view);
        }
    }
}
