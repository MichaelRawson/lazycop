use crate::prelude::*;
use crate::util::id_map::IdMap;

#[derive(Clone, Copy)]
struct Equation {
    left: Id<Term>,
    right: Id<Term>,
}

#[derive(Clone, Copy)]
struct Disequation {
    left: Id<Term>,
    right: Id<Term>,
}

#[derive(Clone, Copy)]
struct SymmetricDisequation {
    left: (Id<Term>, Id<Term>),
    right: (Id<Term>, Id<Term>),
}

#[derive(Clone, Copy)]
struct AtomicDisequation {
    variable: Id<Variable>,
    term: Id<Term>,
}

#[derive(Clone, Copy)]
struct SolvedDisequation(Range<AtomicDisequation>);

#[derive(Clone, Copy)]
struct OccursCheckItem(Id<Term>);

#[derive(Default)]
pub(crate) struct Solver {
    pub bindings: IdMap<Variable, Option<Id<Term>>>,
    save_bindings: IdMap<Variable, Option<Id<Term>>>,
    equations: Block<Equation>,
    disequations: Block<Disequation>,
    symmetric_disequations: Block<SymmetricDisequation>,
    solved_disequations: Block<SolvedDisequation>,
    solved_disequations_mark: Id<SolvedDisequation>,
    atomic_disequations: Block<AtomicDisequation>,
    atomic_disequations_mark: Id<AtomicDisequation>,
    occurs_buf: Block<OccursCheckItem>,
}

impl Solver {
    pub(crate) fn assert_equal(&mut self, left: Id<Term>, right: Id<Term>) {
        self.equations.push(Equation { left, right });
    }

    pub(crate) fn assert_not_equal(
        &mut self,
        left: Id<Term>,
        right: Id<Term>,
    ) {
        self.disequations.push(Disequation { left, right });
    }

    pub(crate) fn assert_not_equal_symmetric(
        &mut self,
        left: (Id<Term>, Id<Term>),
        right: (Id<Term>, Id<Term>),
    ) {
        self.symmetric_disequations
            .push(SymmetricDisequation { left, right });
    }

    pub(crate) fn clear(&mut self) {
        self.bindings.reset();
        self.save_bindings.reset();
        self.equations.clear();
        self.disequations.clear();
        self.symmetric_disequations.clear();
        self.solved_disequations.clear();
        self.atomic_disequations.clear();
    }

    pub(crate) fn mark(&mut self) {
        self.save_bindings.copy_from(&self.bindings);
        self.solved_disequations_mark = self.solved_disequations.len();
        self.atomic_disequations_mark = self.atomic_disequations.len();
    }

    pub(crate) fn undo_to_mark(&mut self) {
        self.bindings.copy_from(&self.save_bindings);
        self.equations.clear();
        self.disequations.clear();
        self.symmetric_disequations.clear();
        self.solved_disequations
            .truncate(self.solved_disequations_mark);
        self.atomic_disequations
            .truncate(self.atomic_disequations_mark);
    }

    pub(crate) fn solve(&mut self, term_graph: &TermGraph) {
        self.bindings.ensure_capacity(term_graph.len().transmute());
        self.solve_equations(term_graph);
        self.solve_disequations(term_graph);
        self.solve_symmetric_disequations(term_graph);
    }

    pub(crate) fn check(&mut self, term_graph: &TermGraph) -> bool {
        self.bindings.ensure_capacity(term_graph.len().transmute());
        self.check_equations(term_graph)
            && self.solve_disequations(term_graph)
            && self.solve_symmetric_disequations(term_graph)
            && self.check_solved_disequations(term_graph)
    }

    fn check_equations(&mut self, term_graph: &TermGraph) -> bool {
        while let Some(equation) = self.equations.pop() {
            let (left, lview) = self.view(term_graph, equation.left);
            let (right, rview) = self.view(term_graph, equation.right);
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
                    self.equations.extend(
                        ts.zip(ss)
                            .map(|(left, right)| Equation { left, right }),
                    );
                }
                _ => {
                    return false;
                }
            }
        }
        true
    }

    fn solve_equations(&mut self, term_graph: &TermGraph) {
        while let Some(equation) = self.equations.pop() {
            let (left, lview) = self.view(term_graph, equation.left);
            let (right, rview) = self.view(term_graph, equation.right);
            if left == right {
                continue;
            }

            match (lview, rview) {
                (TermView::Variable(x), _) => {
                    self.bindings[x] = Some(right);
                }
                (_, TermView::Variable(x)) => {
                    self.bindings[x] = Some(left);
                }
                (TermView::Function(_, ts), TermView::Function(_, ss)) => {
                    self.equations.extend(
                        ts.zip(ss)
                            .map(|(left, right)| Equation { left, right }),
                    );
                }
            }
        }
    }

    fn solve_disequations(&mut self, term_graph: &TermGraph) -> bool {
        while let Some(disequation) = self.disequations.pop() {
            let start = self.atomic_disequations.len();
            if self.solve_disequation(term_graph, disequation) {
                let stop = self.atomic_disequations.len();
                if start == stop {
                    return false;
                } else {
                    let range = Range::new(start, stop);
                    self.solved_disequations.push(SolvedDisequation(range));
                }
            }
        }
        true
    }

    fn solve_symmetric_disequations(
        &mut self,
        term_graph: &TermGraph,
    ) -> bool {
        while let Some(symmetric) = self.symmetric_disequations.pop() {
            let diseq1 = Disequation {
                left: symmetric.left.0,
                right: symmetric.right.0,
            };
            let diseq2 = Disequation {
                left: symmetric.left.1,
                right: symmetric.right.1,
            };
            if !self.solve_symmetric_disequation(term_graph, diseq1, diseq2) {
                return false;
            }
            let diseq1 = Disequation {
                left: symmetric.left.0,
                right: symmetric.right.1,
            };
            let diseq2 = Disequation {
                left: symmetric.left.1,
                right: symmetric.right.0,
            };
            if !self.solve_symmetric_disequation(term_graph, diseq1, diseq2) {
                return false;
            }
        }
        true
    }

    fn solve_symmetric_disequation(
        &mut self,
        term_graph: &TermGraph,
        diseq1: Disequation,
        diseq2: Disequation,
    ) -> bool {
        let start = self.atomic_disequations.len();
        if self.solve_disequation(term_graph, diseq1)
            && self.solve_disequation(term_graph, diseq2)
        {
            let stop = self.atomic_disequations.len();
            if start == stop {
                return false;
            } else {
                let range = Range::new(start, stop);
                self.solved_disequations.push(SolvedDisequation(range));
            }
        }
        true
    }

    fn solve_disequation(
        &mut self,
        term_graph: &TermGraph,
        disequation: Disequation,
    ) -> bool {
        let start = self.atomic_disequations.len();
        let left = disequation.left;
        let right = disequation.right;
        self.equations.push(Equation { left, right });
        while let Some(equation) = self.equations.pop() {
            let (left, lview) = self.view(term_graph, equation.left);
            let (right, rview) = self.view(term_graph, equation.right);
            if left == right {
                continue;
            }

            match (lview, rview) {
                (TermView::Variable(variable), _) => {
                    let term = right;
                    let atomic = AtomicDisequation { variable, term };
                    self.atomic_disequations.push(atomic);
                }
                (_, TermView::Variable(variable)) => {
                    let term = left;
                    let atomic = AtomicDisequation { variable, term };
                    self.atomic_disequations.push(atomic);
                }
                (TermView::Function(f, ts), TermView::Function(g, ss))
                    if f == g =>
                {
                    self.equations.extend(
                        ts.zip(ss)
                            .map(|(left, right)| Equation { left, right }),
                    );
                }
                _ => {
                    self.atomic_disequations.truncate(start);
                    self.equations.clear();
                    return false;
                }
            }
        }
        true
    }

    fn check_solved_disequations(&mut self, term_graph: &TermGraph) -> bool {
        self.solved_disequations.range().all(|solved_id| {
            !self.check_solved_disequation(
                term_graph,
                self.solved_disequations[solved_id],
            )
        })
    }

    fn check_solved_disequation(
        &mut self,
        term_graph: &TermGraph,
        mut solved: SolvedDisequation,
    ) -> bool {
        solved.0.all(|atomic_id| {
            self.check_atomic_disequation(
                term_graph,
                self.atomic_disequations[atomic_id],
            )
        })
    }

    fn check_atomic_disequation(
        &mut self,
        term_graph: &TermGraph,
        atomic: AtomicDisequation,
    ) -> bool {
        if let Some(left) = self.bindings[atomic.variable] {
            let right = atomic.term;
            self.equations.push(Equation { left, right });
        } else {
            return false;
        }
        while let Some(equation) = self.equations.pop() {
            let (left, lview) = self.view(term_graph, equation.left);
            let (right, rview) = self.view(term_graph, equation.right);
            if left == right {
                continue;
            }

            match (lview, rview) {
                (TermView::Function(f, ts), TermView::Function(g, ss))
                    if f == g =>
                {
                    self.equations.extend(
                        ts.zip(ss)
                            .map(|(left, right)| Equation { left, right }),
                    );
                }
                _ => {
                    self.equations.clear();
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
        self.occurs_buf.push(OccursCheckItem(term));
        while let Some(OccursCheckItem(term)) = self.occurs_buf.pop() {
            let (_, view) = self.view(term_graph, term);
            match view {
                TermView::Variable(y) if x == y => {
                    self.occurs_buf.clear();
                    return true;
                }
                TermView::Variable(_) => {}
                TermView::Function(_, ts) => {
                    self.occurs_buf.extend(ts.map(OccursCheckItem));
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
