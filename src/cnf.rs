use crate::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Variable(pub(crate) u32);

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
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

#[derive(Debug)]
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

#[derive(Clone, Debug)]
pub(crate) struct Literal(bool, Atom);

impl Literal {
    fn negated(mut self) -> Self {
        self.0 = !self.0;
        self
    }

    fn vars<E: Extend<Variable>>(&self, vars: &mut E) {
        self.1.vars(vars);
    }
}

#[derive(Clone, Debug)]
enum SkNNF {
    Lit(Literal),
    And(Vec<SkNNF>),
    Or(Vec<SkNNF>),
    Equiv(Box<SkNNF>, Box<SkNNF>),
}

#[derive(Debug)]
pub(crate) struct CNF(pub(crate) Vec<Literal>);

impl SkNNF {
    fn literal(self) -> Literal {
        if let SkNNF::Lit(literal) = self {
            literal
        } else {
            unreachable()
        }
    }

    fn is_literal(&self) -> bool {
        if let SkNNF::Lit(_) = self {
            true
        } else {
            false
        }
    }

    fn vars<E: Extend<Variable>>(&self, vars: &mut E) {
        match self {
            SkNNF::Lit(lit) => lit.vars(vars),
            SkNNF::And(fs) | SkNNF::Or(fs) => {
                for f in fs {
                    f.vars(vars);
                }
            }
            SkNNF::Equiv(left, right) => {
                left.vars(vars);
                right.vars(vars);
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct SkNNFTransform {
    bound: Vec<Variable>,
    skolems: Vec<(Variable, Term)>,
    fresh: u32,
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
            (true, Formula::And(fs)) | (false, Formula::Or(fs)) => {
                let fs = fs
                    .into_iter()
                    .map(|f| self.formula(symbols, polarity, f))
                    .collect();
                SkNNF::And(fs)
            }
            (true, Formula::Or(fs)) | (false, Formula::And(fs)) => {
                let fs = fs
                    .into_iter()
                    .map(|f| self.formula(symbols, polarity, f))
                    .collect();
                SkNNF::Or(fs)
            }
            (_, Formula::Equiv(p, q)) => {
                let p = Box::new(self.formula(symbols, polarity, *p));
                let q = Box::new(self.formula(symbols, true, *q));
                SkNNF::Equiv(p, q)
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
                let symbol = symbols.push(Symbol { name, arity });
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
pub(crate) struct CNFTransform {
    todo: Vec<SkNNF>,
    vars: Vec<Variable>,
    fresh: u32,
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
        let symbol = symbols.push(Symbol { arity, name });
        let term =
            Term::Fun(symbol, self.vars.drain(..).map(Term::Var).collect());
        let atom = Atom::Pred(term);
        self.todo.push(SkNNF::Or(vec![
            SkNNF::Lit(Literal(false, atom.clone())),
            formula,
        ]));
        Literal(true, atom)
    }

    fn flatten(
        &mut self,
        symbols: &mut Symbols,
        mut fs: Vec<SkNNF>,
        f: SkNNF,
    ) {
        match f {
            SkNNF::Lit(_) => unreachable(),
            SkNNF::And(gs) if fs.len() == 1 && fs[0].is_literal() => {
                let literal = some(fs.pop());
                for g in gs {
                    let definition = SkNNF::Lit(self.define(symbols, g));
                    self.todo
                        .push(SkNNF::Or(vec![literal.clone(), definition]));
                }
            }
            SkNNF::And(_) => {
                fs.push(SkNNF::Lit(self.define(symbols, f)));
                self.todo.push(SkNNF::Or(fs));
            }
            SkNNF::Or(gs) => {
                fs.extend(gs);
                self.todo.push(SkNNF::Or(fs));
            }
            SkNNF::Equiv(left, right) => {
                let p = self.define(symbols, *left);
                let q = self.define(symbols, *right);
                let notp = p.clone().negated();
                let notq = q.clone().negated();
                fs.push(SkNNF::And(vec![SkNNF::Lit(p), SkNNF::Lit(q)]));
                fs.push(SkNNF::And(vec![SkNNF::Lit(notp), SkNNF::Lit(notq)]));
                self.todo.push(SkNNF::Or(fs));
            }
        }
    }

    fn next(&mut self, symbols: &mut Symbols) -> Option<CNF> {
        loop {
            let formula = self.todo.pop()?;
            match formula {
                SkNNF::Lit(literal) => return Some(CNF(vec![literal])),
                SkNNF::And(fs) => {
                    self.todo.extend(fs.into_iter());
                }
                SkNNF::Equiv(left, right) => {
                    let left = self.define(symbols, *left);
                    let right = self.define(symbols, *right);
                    self.todo.push(SkNNF::Or(vec![
                        SkNNF::Lit(left.clone().negated()),
                        SkNNF::Lit(right.clone()),
                    ]));
                    self.todo.push(SkNNF::Or(vec![
                        SkNNF::Lit(right.negated()),
                        SkNNF::Lit(left.clone()),
                    ]));
                }
                SkNNF::Or(mut fs) => {
                    if let Some(position) =
                        fs.iter().position(|f| !f.is_literal())
                    {
                        let f = fs.swap_remove(position);
                        self.flatten(symbols, fs, f);
                    } else {
                        return Some(CNF(fs
                            .into_iter()
                            .map(|f| f.literal())
                            .collect()));
                    }
                }
            }
        }
    }

    fn formula<'it, 'me: 'it, 'sym: 'it>(
        &'me mut self,
        symbols: &'sym mut Symbols,
        formula: SkNNF,
    ) -> impl Iterator<Item = CNF> + 'it {
        debug_assert!(self.todo.is_empty());
        self.todo.push(formula);
        std::iter::from_fn(move || self.next(symbols))
    }
}

#[derive(Default)]
pub(crate) struct Clausifier {
    sknnf: SkNNFTransform,
    cnf: CNFTransform,
}

impl Clausifier {
    pub(crate) fn clausify(
        &mut self,
        symbols: &mut Symbols,
        formula: Formula,
    ) {
        let sknnf = self.sknnf.formula(symbols, true, formula);
        for formula in self.cnf.formula(symbols, sknnf) {
            println!("{:?}", formula);
        }
    }
}
