use crate::prelude::*;
use crate::util::id_map::IdMap;

#[derive(Default)]
pub(crate) struct Solver {
    pub bindings: IdMap<Variable, Option<Id<Term>>>,
    save_bindings: IdMap<Variable, Option<Id<Term>>>,
    equalities: Vec<(Id<Term>, Id<Term>)>,
    disequalities: Vec<(Atom, Atom)>,
    postponed_disequalities: Vec<(Atom, Atom)>,
    buf: Vec<Id<Term>>,
}

impl Solver {
    pub(crate) fn equal(&mut self, left: Id<Term>, right: Id<Term>) {
        self.equalities.push((left, right));
    }

    pub(crate) fn unequal(&mut self, left: Atom, right: Atom) {
        self.disequalities.push((left, right));
    }

    pub(crate) fn clear(&mut self) {
        self.bindings.wipe();
        self.save_bindings.wipe();
        self.equalities.clear();
        self.disequalities.clear();
        self.postponed_disequalities.clear();
    }

    pub(crate) fn mark(&mut self) {
        self.save_bindings.copy_from(&self.bindings);
    }

    pub(crate) fn undo_to_mark(&mut self) {
        self.bindings.copy_from(&self.save_bindings);
        self.equalities.clear();
        self.disequalities.clear();
    }

    pub(crate) fn simplify(&mut self, term_graph: &TermGraph) {
        self.bindings.ensure_capacity(term_graph.len());
        self.solve_equalities(term_graph);
        self.simplify_disequalities(term_graph);
    }

    pub(crate) fn solve(&mut self, term_graph: &TermGraph) -> bool {
        self.bindings.ensure_capacity(term_graph.len());
        self.solve_equalities(term_graph)
            && self.solve_disequalities(term_graph)
    }

    fn solve_equalities(&mut self, term_graph: &TermGraph) -> bool {
        while let Some((left, right)) = self.equalities.pop() {
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
                    self.equalities.extend(ts.zip(ss));
                }
                _ => {
                    return false;
                }
            }
        }
        true
    }

    fn solve_disequalities(&mut self, term_graph: &TermGraph) -> bool {
        self.disequalities
            .extend_from_slice(&self.postponed_disequalities);
        while let Some((left, right)) = self.disequalities.pop() {
            match (left, right) {
                (Atom::Predicate(p), Atom::Predicate(q)) => {
                    if self.solve_disequality(term_graph, p, q) {
                        continue;
                    }
                }
                (Atom::Equality(l1, r1), Atom::Equality(l2, r2)) => {
                    if self.solve_disequality(term_graph, l1, l2)
                        || self.solve_disequality(term_graph, r1, r2)
                    {
                        continue;
                    }
                }
                _ => unreachable!("bad disequation"),
            }
            return false;
        }
        true
    }

    fn solve_disequality(
        &mut self,
        term_graph: &TermGraph,
        left: Id<Term>,
        right: Id<Term>,
    ) -> bool {
        self.equalities.clear();
        self.equalities.push((left, right));
        while let Some((left, right)) = self.equalities.pop() {
            let (left, lview) = self.view(term_graph, left);
            let (right, rview) = self.view(term_graph, right);
            if left == right {
                continue;
            }

            if let (TermView::Function(f, ts), TermView::Function(g, ss)) =
                (lview, rview)
            {
                if f == g {
                    self.equalities.extend(ts.zip(ss));
                    continue;
                }
            }
            return true;
        }
        false
    }

    fn simplify_disequalities(&mut self, term_graph: &TermGraph) {
        while let Some((left, right)) = self.disequalities.pop() {
            match (left, right) {
                (Atom::Predicate(p), Atom::Predicate(q)) => {
                    if self.is_disequality_trivial(term_graph, p, q) {
                        continue;
                    }
                }
                (Atom::Equality(l1, r1), Atom::Equality(l2, r2)) => {
                    if self.is_disequality_trivial(term_graph, l1, l2)
                        || self.is_disequality_trivial(term_graph, r1, r2)
                    {
                        continue;
                    }
                }
                _ => unreachable!("bad disequation"),
            }
            self.postponed_disequalities.push((left, right));
        }
    }

    fn is_disequality_trivial(
        &mut self,
        term_graph: &TermGraph,
        left: Id<Term>,
        right: Id<Term>,
    ) -> bool {
        self.equalities.push((left, right));
        while let Some((left, right)) = self.equalities.pop() {
            let (left, lview) = self.view(term_graph, left);
            let (right, rview) = self.view(term_graph, right);
            if left == right {
                continue;
            }

            if let (TermView::Function(f, ts), TermView::Function(g, ss)) =
                (lview, rview)
            {
                if f == g {
                    self.equalities.extend(ts.zip(ss))
                } else {
                    self.equalities.clear();
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
        self.buf.clear();
        self.buf.push(term);
        while let Some(term) = self.buf.pop() {
            let (_, view) = self.view(term_graph, term);
            match view {
                TermView::Variable(y) if x == y => {
                    return true;
                }
                TermView::Variable(_) => {}
                TermView::Function(_, ts) => {
                    self.buf.extend(ts);
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
