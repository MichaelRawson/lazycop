use crate::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Variable(pub(crate) u32);

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum Term {
    Var(Variable),
    Fun(Id<Symbol>, Vec<Term>),
}

impl Term {
    fn subst(self, lookup: &[(Variable, Term)]) -> Self {
        match self {
            Term::Var(x) => {
                if let Some((_, term)) = lookup.iter().find(|(y, _)| x == *y) {
                    term.clone()
                } else {
                    self
                }
            }
            Term::Fun(f, args) => Term::Fun(
                f,
                args.into_iter().map(|term| term.subst(lookup)).collect(),
            ),
        }
    }

    fn vars<E: Extend<Variable>>(&self, vars: &mut E) {
        match self {
            Term::Var(x) => {
                vars.extend(std::iter::once(*x));
            }
            Term::Fun(_, ts) => {
                for t in ts {
                    t.vars(vars);
                }
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum Atom {
    Pred(Term),
    Eq(Term, Term),
}

impl Atom {
    fn subst(self, lookup: &[(Variable, Term)]) -> Self {
        match self {
            Atom::Pred(term) => Atom::Pred(term.subst(lookup)),
            Atom::Eq(left, right) => {
                Atom::Eq(left.subst(lookup), right.subst(lookup))
            }
        }
    }

    fn vars<E: Extend<Variable>>(&self, vars: &mut E) {
        match self {
            Atom::Pred(term) => term.vars(vars),
            Atom::Eq(left, right) => {
                left.vars(vars);
                right.vars(vars);
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Literal(pub(crate) bool, pub(crate) Atom);

impl Literal {
    fn vars<E: Extend<Variable>>(&self, vars: &mut E) {
        self.1.vars(vars);
    }
}

#[derive(Clone)]
pub(crate) enum Formula {
    Atom(Atom),
    Not(Box<Formula>),
    And(Vec<Formula>),
    Or(Vec<Formula>),
    Equiv(Box<Formula>, Box<Formula>),
    Forall(Variable, Box<Formula>),
    Exists(Variable, Box<Formula>),
}

impl Formula {
    pub(crate) fn negated(self) -> Self {
        Self::Not(Box::new(self))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum SkNNF {
    Lit(Literal),
    And(Vec<SkNNF>),
    Or(Vec<SkNNF>),
}

impl SkNNF {
    fn vars<E: Extend<Variable>>(&self, vars: &mut E) {
        match self {
            Self::Lit(lit) => lit.vars(vars),
            Self::And(fs) | Self::Or(fs) => {
                for f in fs {
                    f.vars(vars);
                }
            }
        }
    }
}

pub(crate) struct CNF(pub(crate) Vec<Literal>);

#[derive(Default)]
struct SkNNFTransform {
    bound: Vec<Variable>,
    skolems: Vec<(Variable, Term)>,
    fresh: usize,
}

impl SkNNFTransform {
    fn formula(
        &mut self,
        symbols: &mut Symbols,
        mut polarity: bool,
        mut formula: Formula,
    ) -> SkNNF {
        while let Formula::Not(negated) = formula {
            polarity = !polarity;
            formula = *negated;
        }

        match (polarity, formula) {
            (_, Formula::Atom(atom)) => {
                SkNNF::Lit(Literal(polarity, atom.subst(&self.skolems)))
            }
            (true, Formula::And(fs)) | (false, Formula::Or(fs)) => SkNNF::And(
                fs.into_iter()
                    .map(|f| self.formula(symbols, polarity, f))
                    .collect(),
            ),
            (true, Formula::Or(fs)) | (false, Formula::And(fs)) => SkNNF::Or(
                fs.into_iter()
                    .map(|f| self.formula(symbols, polarity, f))
                    .collect(),
            ),
            (_, Formula::Equiv(p, q)) => {
                let p = *p;
                let q = *q;
                let notp = self.formula(symbols, !polarity, p.clone());
                let p = self.formula(symbols, polarity, p);
                let notq = self.formula(symbols, false, q.clone());
                let q = self.formula(symbols, true, q);
                SkNNF::And(vec![
                    SkNNF::Or(vec![notp, q]),
                    SkNNF::Or(vec![notq, p]),
                ])
            }
            (true, Formula::Forall(x, f)) | (false, Formula::Exists(x, f)) => {
                self.bound.push(x);
                let f = self.formula(symbols, polarity, *f);
                self.bound.pop();
                f
            }
            (true, Formula::Exists(x, f)) | (false, Formula::Forall(x, f)) => {
                let name = Name::Skolem(self.fresh);
                self.fresh += 1;
                let arity = self.bound.len() as u32;
                let symbol = symbols.push(Symbol {
                    name,
                    arity,
                    #[cfg(feature = "smt")]
                    is_predicate: false,
                });
                let skolem = Term::Fun(
                    symbol,
                    self.bound.iter().copied().map(Term::Var).collect(),
                );

                self.skolems.push((x, skolem));
                let f = self.formula(symbols, polarity, *f);
                self.skolems.pop();
                f
            }
            (_, Formula::Not(_)) => unreachable(),
        }
    }
}

#[derive(Default)]
struct CNFTransform {
    todo: Vec<SkNNF>,
    vars: Vec<Variable>,
    fresh: usize,
}

impl CNFTransform {
    fn define(&mut self, symbols: &mut Symbols, formula: SkNNF) -> Literal {
        if let SkNNF::Lit(Literal(polarity, atom)) = formula {
            return Literal(polarity, atom);
        }
        formula.vars(&mut self.vars);
        self.vars.sort();
        self.vars.dedup();

        let arity = self.vars.len() as u32;
        let name = Name::Definition(self.fresh);
        self.fresh += 1;
        let symbol = symbols.push(Symbol {
            arity,
            name,
            #[cfg(feature = "smt")]
            is_predicate: true,
        });
        let term =
            Term::Fun(symbol, self.vars.drain(..).map(Term::Var).collect());
        let atom = Atom::Pred(term);

        if let SkNNF::And(fs) = formula {
            for f in fs {
                self.todo.push(SkNNF::Or(vec![
                    SkNNF::Lit(Literal(false, atom.clone())),
                    f,
                ]));
            }
        } else {
            self.todo.push(SkNNF::Or(vec![
                SkNNF::Lit(Literal(false, atom.clone())),
                formula,
            ]));
        }
        Literal(true, atom)
    }

    fn next(&mut self, symbols: &mut Symbols) -> Option<CNF> {
        loop {
            let formula = self.todo.pop()?;
            match formula {
                SkNNF::Lit(literal) => return Some(CNF(vec![literal])),
                SkNNF::And(fs) => {
                    self.todo.extend(fs.into_iter());
                }
                SkNNF::Or(mut fs) => {
                    let mut literals = vec![];
                    while let Some(f) = fs.pop() {
                        match f {
                            SkNNF::Lit(literal) => {
                                literals.push(literal);
                            }
                            SkNNF::And(_) => {
                                fs.push(SkNNF::Lit(self.define(symbols, f)));
                            }
                            SkNNF::Or(gs) => {
                                fs.extend(gs);
                            }
                        }
                    }
                    let mut index = 0;
                    while index < literals.len() {
                        if literals[index + 1..].contains(&literals[index]) {
                            literals.remove(index);
                        } else {
                            index += 1;
                        }
                    }
                    return Some(CNF(literals));
                }
            }
        }
    }

    fn formula(&mut self, formula: SkNNF) {
        debug_assert!(self.todo.is_empty());
        self.todo.push(formula);
    }
}

#[derive(Default)]
pub(crate) struct Clausifier {
    sknnf: SkNNFTransform,
    cnf: CNFTransform,
}

impl Clausifier {
    pub(crate) fn formula(&mut self, symbols: &mut Symbols, formula: Formula) {
        let sknnf = self.sknnf.formula(symbols, true, formula);
        self.cnf.formula(sknnf);
    }

    pub(crate) fn next(&mut self, symbols: &mut Symbols) -> Option<CNF> {
        self.cnf.next(symbols)
    }
}
