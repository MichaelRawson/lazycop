use crate::clausify;
use crate::io::exit;
use crate::io::szs;
use crate::prelude::*;
use fnv::FnvHashMap;
use memmap::Mmap;
use std::env;
use std::fmt;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tptp::{cnf, common, fof, top, TPTPIterator};

fn report_os_error<D: fmt::Display>(path: D, e: std::io::Error) -> ! {
    println!("% error reading {}: {}", path, e);
    szs::os_error();
    exit::failure()
}

fn report_syntax_error() -> ! {
    println!("% unsupported syntax or syntax error");
    szs::syntax_error();
    exit::failure()
}

fn report_inappropriate<T: fmt::Display>(t: T) -> ! {
    println!("% unsupported input feature: {}", t);
    szs::inappropriate();
    exit::failure()
}

struct TPTPFile<'a> {
    _map: Mmap,
    parser: TPTPIterator<'a, ()>,
    path: Arc<PathBuf>,
}

fn open_relative(old: &Path, path: &Path) -> Option<File> {
    let mut directory: PathBuf = old.parent()?.into();
    directory.push(path);
    File::open(directory).ok()
}

fn open_root(path: &Path) -> io::Result<File> {
    let mut directory = env::var("TPTP_DIR")
        .map(PathBuf::from)
        .or_else(|_| env::current_dir())
        .unwrap_or_else(|e| report_os_error("TPTP_DIR", e));
    directory.push(path);
    File::open(directory)
}

fn open_include(old: &Path, path: &Path) -> io::Result<File> {
    open_relative(old, path)
        .map(Ok)
        .unwrap_or_else(|| open_root(path))
}

impl<'a> TPTPFile<'a> {
    fn new(old: Option<&Path>, path: &Path) -> Self {
        let file = if let Some(old) = old {
            open_include(old, path)
        } else {
            open_root(path)
        };
        let _map = file
            .and_then(|file| unsafe { Mmap::map(&file) })
            .unwrap_or_else(|e| report_os_error(path.display(), e));

        let bytes = _map.as_ref();
        let bytes: &'a [u8] = unsafe { std::mem::transmute(bytes) };
        let parser = TPTPIterator::new(bytes);
        let path = Arc::new(path.into());
        Self { _map, parser, path }
    }

    fn next(&mut self) -> Option<top::TPTPInput<'a>> {
        let result = self.parser.next()?;
        let input = result.unwrap_or_else(|_| report_syntax_error());
        Some(input)
    }
}

struct TPTPProblem<'a> {
    stack: Vec<TPTPFile<'a>>,
    empty: Vec<TPTPFile<'a>>,
}

impl<'a> TPTPProblem<'a> {
    fn new(path: &Path) -> Self {
        let stack = vec![TPTPFile::new(None, path)];
        let empty = vec![];
        Self { stack, empty }
    }

    fn next(&mut self) -> Option<top::AnnotatedFormula<'a>> {
        loop {
            let input = loop {
                let top = self.stack.last_mut()?;
                if let Some(next) = top.next() {
                    break next;
                } else {
                    let empty = unwrap(self.stack.pop());
                    self.empty.push(empty);
                }
            };
            match input {
                top::TPTPInput::Annotated(annotated) => {
                    return Some(*annotated)
                }
                top::TPTPInput::Include(include) => {
                    let path = (include.file_name.0).0;
                    let path = Path::new(path);
                    let old = self.current_path();
                    let old = Some(old.as_ref().as_ref());
                    self.stack.push(TPTPFile::new(old, path));
                }
            };
        }
    }

    fn current_path(&self) -> Arc<PathBuf> {
        unwrap(self.stack.last()).path.clone()
    }
}

pub(crate) struct Loader<'a> {
    problem: TPTPProblem<'a>,
    unbound: Vec<(common::Variable<'a>, clausify::Variable)>,
    bound: Vec<(common::Variable<'a>, clausify::Variable)>,
    fresh: u32,
    functors: FnvHashMap<common::Functor<'a>, Id<Symbol>>,
}

impl<'a> Loader<'a> {
    pub(crate) fn new(path: &Path) -> Self {
        let problem = TPTPProblem::new(path);
        let unbound = vec![];
        let bound = vec![];
        let fresh = 0;
        let functors = FnvHashMap::default();
        Self {
            problem,
            unbound,
            bound,
            fresh,
            functors,
        }
    }

    pub(crate) fn next(
        &mut self,
        symbols: &mut Symbols,
    ) -> Option<(bool, Origin, clausify::Formula)> {
        let annotated = self.problem.next()?;
        let path = self.problem.current_path();
        let (is_cnf, conjecture, name, formula) =
            self.annotated_formula(symbols, annotated);
        let name = Arc::new(name);
        let origin = Origin {
            conjecture,
            path,
            name,
        };
        Some((is_cnf, origin, formula))
    }

    fn functor(
        &mut self,
        symbols: &mut Symbols,
        functor: common::Functor<'a>,
        arity: u32,
    ) -> Id<Symbol> {
        if let Some(id) = self.functors.get(&functor) {
            *id
        } else {
            let name = match functor.0 {
                common::AtomicWord::Lower(ref word) => {
                    Name::Regular(format!("{}", word))
                }
                common::AtomicWord::SingleQuoted(ref sq) => {
                    Name::Quoted(sq.0.to_string())
                }
            };
            let symbol = Symbol { name, arity };
            let id = symbols.push(symbol);
            self.functors.insert(functor, id);
            id
        }
    }

    fn fof_plain_term(
        &mut self,
        symbols: &mut Symbols,
        plain: fof::PlainTerm<'a>,
    ) -> clausify::Term {
        match plain {
            fof::PlainTerm::Constant(c) => {
                clausify::Term::Fun(self.functor(symbols, c.0, 0), vec![])
            }
            fof::PlainTerm::Function(f, args) => {
                let args: Vec<_> = args
                    .0
                    .into_iter()
                    .map(|t| self.fof_term(symbols, t))
                    .collect();
                clausify::Term::Fun(
                    self.functor(symbols, f, args.len() as u32),
                    args,
                )
            }
        }
    }

    fn fof_term(
        &mut self,
        symbols: &mut Symbols,
        term: fof::Term<'a>,
    ) -> clausify::Term {
        match term {
            fof::Term::Variable(var) => {
                if let Some((_, bound)) =
                    self.bound.iter().rev().find(|(bound, _)| bound == &var)
                {
                    clausify::Term::Var(*bound)
                } else if let Some((_, unbound)) = self
                    .unbound
                    .iter()
                    .rev()
                    .find(|(unbound, _)| unbound == &var)
                {
                    clausify::Term::Var(*unbound)
                } else {
                    let fresh = clausify::Variable(self.fresh);
                    self.unbound.push((var, fresh));
                    self.fresh += 1;
                    clausify::Term::Var(fresh)
                }
            }
            fof::Term::Function(function) => match *function {
                fof::FunctionTerm::Plain(plain) => {
                    self.fof_plain_term(symbols, plain)
                }
                fof::FunctionTerm::Defined(defined) => {
                    report_inappropriate(defined)
                }
            },
        }
    }

    fn fof_atomic_formula(
        &mut self,
        symbols: &mut Symbols,
        atomic: fof::AtomicFormula<'a>,
    ) -> clausify::Formula {
        match atomic {
            fof::AtomicFormula::Plain(plain) => {
                let pred = self.fof_plain_term(symbols, plain.0);
                clausify::Formula::Atom(clausify::Atom::Pred(pred))
            }
            fof::AtomicFormula::Defined(defined) => match defined {
                fof::DefinedAtomicFormula::Plain(plain) => {
                    match ((((((plain.0).0).0).0).0).0).0 {
                        "true" => clausify::Formula::And(vec![]),
                        "false" => clausify::Formula::Or(vec![]),
                        _ => report_inappropriate(plain),
                    }
                }
                fof::DefinedAtomicFormula::Infix(infix) => {
                    let left = self.fof_term(symbols, *infix.left);
                    let right = self.fof_term(symbols, *infix.right);
                    clausify::Formula::Atom(clausify::Atom::Eq(left, right))
                }
            },
            fof::AtomicFormula::System(system) => report_inappropriate(system),
        }
    }

    fn fof_unitary_formula(
        &mut self,
        symbols: &mut Symbols,
        unitary: fof::UnitaryFormula<'a>,
    ) -> clausify::Formula {
        match unitary {
            fof::UnitaryFormula::Quantified(quantified) => {
                let quantifier = match quantified.quantifier {
                    fof::Quantifier::Forall => clausify::Formula::Forall,
                    fof::Quantifier::Exists => clausify::Formula::Exists,
                };
                let bound = quantified.bound.0;
                let num_bound = bound.len();
                for x in bound {
                    self.bound.push((x, clausify::Variable(self.fresh)));
                    self.fresh += 1;
                }

                let mut formula =
                    self.fof_unit_formula(symbols, *quantified.formula);
                for _ in 0..num_bound {
                    let (_, x) = unwrap(self.bound.pop());
                    formula = quantifier(x, Box::new(formula));
                }

                formula
            }
            fof::UnitaryFormula::Atomic(atomic) => {
                self.fof_atomic_formula(symbols, *atomic)
            }
            fof::UnitaryFormula::Parenthesised(logic) => {
                self.fof_logic_formula(symbols, *logic)
            }
        }
    }

    fn fof_infix_unary(
        &mut self,
        symbols: &mut Symbols,
        infix: fof::InfixUnary<'a>,
    ) -> clausify::Formula {
        let left = self.fof_term(symbols, *infix.left);
        let right = self.fof_term(symbols, *infix.right);
        clausify::Formula::Atom(clausify::Atom::Eq(left, right)).negated()
    }

    fn fof_unary_formula(
        &mut self,
        symbols: &mut Symbols,
        unary: fof::UnaryFormula<'a>,
    ) -> clausify::Formula {
        match unary {
            fof::UnaryFormula::Unary(_, unary) => {
                self.fof_unit_formula(symbols, *unary).negated()
            }
            fof::UnaryFormula::InfixUnary(infix) => {
                self.fof_infix_unary(symbols, infix)
            }
        }
    }

    fn fof_unit_formula(
        &mut self,
        symbols: &mut Symbols,
        unit: fof::UnitFormula<'a>,
    ) -> clausify::Formula {
        match unit {
            fof::UnitFormula::Unitary(unitary) => {
                self.fof_unitary_formula(symbols, unitary)
            }
            fof::UnitFormula::Unary(unary) => {
                self.fof_unary_formula(symbols, unary)
            }
        }
    }

    fn fof_logic_formula(
        &mut self,
        symbols: &mut Symbols,
        logic: fof::LogicFormula<'a>,
    ) -> clausify::Formula {
        match logic {
            fof::LogicFormula::Binary(binary) => match binary {
                fof::BinaryFormula::Nonassoc(nonassoc) => {
                    let left = self.fof_unit_formula(symbols, *nonassoc.left);
                    let right =
                        self.fof_unit_formula(symbols, *nonassoc.right);
                    match nonassoc.op {
                        common::NonassocConnective::LRImplies => {
                            clausify::Formula::Or(vec![left.negated(), right])
                        }
                        common::NonassocConnective::RLImplies => {
                            clausify::Formula::Or(vec![left, right.negated()])
                        }
                        common::NonassocConnective::Equivalent => {
                            clausify::Formula::Equiv(
                                Box::new(left),
                                Box::new(right),
                            )
                        }
                        common::NonassocConnective::NotEquivalent => {
                            clausify::Formula::Equiv(
                                Box::new(left),
                                Box::new(right),
                            )
                            .negated()
                        }
                        common::NonassocConnective::NotOr => {
                            clausify::Formula::Or(vec![left, right]).negated()
                        }
                        common::NonassocConnective::NotAnd => {
                            clausify::Formula::And(vec![left, right]).negated()
                        }
                    }
                }
                fof::BinaryFormula::Assoc(assoc) => match assoc {
                    fof::BinaryAssoc::Or(or) => clausify::Formula::Or(
                        or.0.into_iter()
                            .map(|unit| self.fof_unit_formula(symbols, unit))
                            .collect(),
                    ),
                    fof::BinaryAssoc::And(and) => clausify::Formula::And(
                        and.0
                            .into_iter()
                            .map(|unit| self.fof_unit_formula(symbols, unit))
                            .collect(),
                    ),
                },
            },
            fof::LogicFormula::Unary(unary) => {
                self.fof_unary_formula(symbols, unary)
            }
            fof::LogicFormula::Unitary(unitary) => {
                self.fof_unitary_formula(symbols, unitary)
            }
        }
    }

    fn literal(
        &mut self,
        symbols: &mut Symbols,
        literal: cnf::Literal<'a>,
    ) -> clausify::Formula {
        match literal {
            cnf::Literal::Atomic(atomic) => {
                self.fof_atomic_formula(symbols, atomic)
            }
            cnf::Literal::NegatedAtomic(atomic) => {
                self.fof_atomic_formula(symbols, atomic).negated()
            }
            cnf::Literal::Infix(infix) => self.fof_infix_unary(symbols, infix),
        }
    }

    fn cnf_formula(
        &mut self,
        symbols: &mut Symbols,
        cnf: cnf::Formula<'a>,
    ) -> clausify::Formula {
        let mut literals = match cnf {
            cnf::Formula::Disjunction(disjunction) => disjunction,
            cnf::Formula::Parenthesised(disjunction) => disjunction,
        }
        .0;
        if literals.len() == 1 {
            let literal = unwrap(literals.pop());
            return self.literal(symbols, literal);
        }
        clausify::Formula::Or(
            literals
                .into_iter()
                .map(|literal| self.literal(symbols, literal))
                .collect::<Vec<_>>(),
        )
    }

    fn fof_formula(
        &mut self,
        symbols: &mut Symbols,
        fof: fof::Formula<'a>,
    ) -> clausify::Formula {
        self.fof_logic_formula(symbols, fof.0)
    }

    fn annotated_formula(
        &mut self,
        symbols: &mut Symbols,
        formula: top::AnnotatedFormula<'a>,
    ) -> (bool, bool, String, clausify::Formula) {
        let (is_cnf, name, mut role, mut formula) = match formula {
            top::AnnotatedFormula::Fof(fof) => {
                let fof = fof.0;
                let formula = self.fof_formula(symbols, *fof.formula);
                (false, format!("{}", fof.name), fof.role, formula)
            }
            top::AnnotatedFormula::Cnf(cnf) => {
                let cnf = cnf.0;
                let formula = self.cnf_formula(symbols, *cnf.formula);
                (true, format!("{}", cnf.name), cnf.role, formula)
            }
        };
        debug_assert!(self.bound.is_empty());
        self.unbound.clear();
        self.fresh = 0;

        if role == top::FormulaRole::Conjecture {
            formula = formula.negated();
            role = top::FormulaRole::NegatedConjecture;
        }
        let conjecture = role == top::FormulaRole::NegatedConjecture;
        (is_cnf, conjecture, name, formula)
    }
}
