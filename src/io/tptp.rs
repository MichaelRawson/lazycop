use crate::cnf;
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
use tptp::parsers::TPTPIterator;
use tptp::syntax as st;

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

    fn next(&mut self) -> Option<st::TPTPInput<'a>> {
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

    fn next(&mut self) -> Option<st::AnnotatedFormula<'a>> {
        loop {
            let input = loop {
                let top = self.stack.last_mut()?;
                if let Some(next) = top.next() {
                    break next;
                } else {
                    let empty = some(self.stack.pop());
                    self.empty.push(empty);
                }
            };
            match input {
                st::TPTPInput::Annotated(annotated) => break Some(annotated),
                st::TPTPInput::Include(include) => {
                    let path = include.file_name.as_ref().as_ref();
                    let path = Path::new(path);
                    let old = self.current_path();
                    let old = Some(old.as_ref().as_ref());
                    self.stack.push(TPTPFile::new(old, path));
                }
            };
        }
    }

    fn current_path(&self) -> Arc<PathBuf> {
        some(self.stack.last()).path.clone()
    }
}

pub(crate) struct Loader<'a> {
    problem: TPTPProblem<'a>,
    unbound: Vec<(st::Variable<'a>, cnf::Variable)>,
    bound: Vec<(st::Variable<'a>, cnf::Variable)>,
    fresh: u32,
    functors: FnvHashMap<st::Functor<'a>, Id<Symbol>>,
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
    ) -> Option<(bool, Origin, cnf::Formula)> {
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
        functor: st::Functor<'a>,
        arity: u32,
    ) -> Id<Symbol> {
        if let Some(id) = self.functors.get(&functor) {
            *id
        } else {
            let name = match functor.0 {
                st::AtomicWord::Lower(ref word) => {
                    Name::Regular(format!("{}", word))
                }
                st::AtomicWord::SingleQuoted(ref sq) => {
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
        plain: st::FofPlainTerm<'a>,
    ) -> cnf::Term {
        match plain {
            st::FofPlainTerm::Constant(c) => {
                cnf::Term::Fun(self.functor(symbols, c.0, 0), vec![])
            }
            st::FofPlainTerm::Function(f, args) => {
                let args: Vec<_> = args
                    .0
                    .into_iter()
                    .map(|t| self.fof_term(symbols, t))
                    .collect();
                cnf::Term::Fun(
                    self.functor(symbols, f, args.len() as u32),
                    args,
                )
            }
        }
    }

    fn fof_term(
        &mut self,
        symbols: &mut Symbols,
        term: st::FofTerm<'a>,
    ) -> cnf::Term {
        match term {
            st::FofTerm::Variable(var) => {
                if let Some((_, bound)) =
                    self.bound.iter().rev().find(|(bound, _)| bound == &var)
                {
                    cnf::Term::Var(*bound)
                } else if let Some((_, unbound)) = self
                    .unbound
                    .iter()
                    .rev()
                    .find(|(unbound, _)| unbound == &var)
                {
                    cnf::Term::Var(*unbound)
                } else {
                    let fresh = cnf::Variable(self.fresh);
                    self.unbound.push((var, fresh));
                    self.fresh += 1;
                    cnf::Term::Var(fresh)
                }
            }
            st::FofTerm::Function(function) => match function {
                st::FofFunctionTerm::Plain(plain) => {
                    self.fof_plain_term(symbols, plain)
                }
                st::FofFunctionTerm::Defined(defined) => {
                    report_inappropriate(defined)
                }
            },
        }
    }

    fn fof_atomic_formula(
        &mut self,
        symbols: &mut Symbols,
        atomic: st::FofAtomicFormula<'a>,
    ) -> cnf::Formula {
        match atomic {
            st::FofAtomicFormula::Plain(plain) => {
                let pred = self.fof_plain_term(symbols, plain.0);
                cnf::Formula::Atom(cnf::Atom::Pred(pred))
            }
            st::FofAtomicFormula::Defined(defined) => match defined {
                st::FofDefinedAtomicFormula::Plain(plain) => {
                    match ((((((plain.0).0).0).0).0).0).0 {
                        "true" => cnf::Formula::And(vec![]),
                        "false" => cnf::Formula::Or(vec![]),
                        _ => report_inappropriate(plain),
                    }
                }
                st::FofDefinedAtomicFormula::Infix(infix) => {
                    let left = self.fof_term(symbols, *infix.left);
                    let right = self.fof_term(symbols, *infix.right);
                    cnf::Formula::Atom(cnf::Atom::Eq(left, right))
                }
            },
            st::FofAtomicFormula::System(system) => {
                report_inappropriate(system)
            }
        }
    }

    fn fof_unitary_formula(
        &mut self,
        symbols: &mut Symbols,
        unitary: st::FofUnitaryFormula<'a>,
    ) -> cnf::Formula {
        match unitary {
            st::FofUnitaryFormula::Quantified(quantified) => {
                let quantifier = match quantified.quantifier {
                    st::FofQuantifier::Forall => cnf::Formula::Forall,
                    st::FofQuantifier::Exists => cnf::Formula::Exists,
                };
                let bound = quantified.bound.0;
                let num_bound = bound.len();
                for x in bound {
                    self.bound.push((x, cnf::Variable(self.fresh)));
                    self.fresh += 1;
                }

                let mut formula =
                    self.fof_unit_formula(symbols, *quantified.formula);
                for _ in 0..num_bound {
                    let (_, x) = some(self.bound.pop());
                    formula = quantifier(x, Box::new(formula));
                }

                formula
            }
            st::FofUnitaryFormula::Atomic(atomic) => {
                self.fof_atomic_formula(symbols, atomic)
            }
            st::FofUnitaryFormula::Parenthesised(logic) => {
                self.fof_logic_formula(symbols, *logic)
            }
        }
    }

    fn fof_infix_unary(
        &mut self,
        symbols: &mut Symbols,
        infix: st::FofInfixUnary<'a>,
    ) -> cnf::Formula {
        let left = self.fof_term(symbols, *infix.left);
        let right = self.fof_term(symbols, *infix.right);
        cnf::Formula::Atom(cnf::Atom::Eq(left, right)).negated()
    }

    fn fof_unary_formula(
        &mut self,
        symbols: &mut Symbols,
        unary: st::FofUnaryFormula<'a>,
    ) -> cnf::Formula {
        match unary {
            st::FofUnaryFormula::Unary(_, unary) => {
                self.fof_unit_formula(symbols, *unary).negated()
            }
            st::FofUnaryFormula::InfixUnary(infix) => {
                self.fof_infix_unary(symbols, infix)
            }
        }
    }

    fn fof_unit_formula(
        &mut self,
        symbols: &mut Symbols,
        unit: st::FofUnitFormula<'a>,
    ) -> cnf::Formula {
        match unit {
            st::FofUnitFormula::Unitary(unitary) => {
                self.fof_unitary_formula(symbols, unitary)
            }
            st::FofUnitFormula::Unary(unary) => {
                self.fof_unary_formula(symbols, unary)
            }
        }
    }

    fn fof_logic_formula(
        &mut self,
        symbols: &mut Symbols,
        logic: st::FofLogicFormula<'a>,
    ) -> cnf::Formula {
        match logic {
            st::FofLogicFormula::Binary(binary) => match binary {
                st::FofBinaryFormula::Nonassoc(nonassoc) => {
                    let left = self.fof_unit_formula(symbols, *nonassoc.left);
                    let right =
                        self.fof_unit_formula(symbols, *nonassoc.right);
                    match nonassoc.op {
                        st::NonassocConnective::LRImplies => {
                            cnf::Formula::Or(vec![left.negated(), right])
                        }
                        st::NonassocConnective::RLImplies => {
                            cnf::Formula::Or(vec![left, right.negated()])
                        }
                        st::NonassocConnective::Equivalent => {
                            cnf::Formula::Equiv(
                                Box::new(left),
                                Box::new(right),
                            )
                        }
                        st::NonassocConnective::NotEquivalent => {
                            cnf::Formula::Equiv(
                                Box::new(left),
                                Box::new(right),
                            )
                            .negated()
                        }
                        st::NonassocConnective::NotOr => {
                            cnf::Formula::Or(vec![left, right]).negated()
                        }
                        st::NonassocConnective::NotAnd => {
                            cnf::Formula::And(vec![left, right]).negated()
                        }
                    }
                }
                st::FofBinaryFormula::Assoc(assoc) => match assoc {
                    st::FofBinaryAssoc::Or(or) => cnf::Formula::Or(
                        or.0.into_iter()
                            .map(|unit| self.fof_unit_formula(symbols, unit))
                            .collect(),
                    ),
                    st::FofBinaryAssoc::And(and) => cnf::Formula::And(
                        and.0
                            .into_iter()
                            .map(|unit| self.fof_unit_formula(symbols, unit))
                            .collect(),
                    ),
                },
            },
            st::FofLogicFormula::Unary(unary) => {
                self.fof_unary_formula(symbols, unary)
            }
            st::FofLogicFormula::Unitary(unitary) => {
                self.fof_unitary_formula(symbols, unitary)
            }
        }
    }

    fn literal(
        &mut self,
        symbols: &mut Symbols,
        literal: st::Literal<'a>,
    ) -> cnf::Formula {
        match literal {
            st::Literal::Atomic(atomic) => {
                self.fof_atomic_formula(symbols, atomic)
            }
            st::Literal::NegatedAtomic(atomic) => {
                self.fof_atomic_formula(symbols, atomic).negated()
            }
            st::Literal::Infix(infix) => self.fof_infix_unary(symbols, infix),
        }
    }

    fn cnf_formula(
        &mut self,
        symbols: &mut Symbols,
        cnf: st::CnfFormula<'a>,
    ) -> cnf::Formula {
        let mut literals = match cnf {
            st::CnfFormula::Disjunction(disjunction) => disjunction,
            st::CnfFormula::Parenthesised(disjunction) => disjunction,
        }
        .0;
        if literals.len() == 1 {
            let literal = some(literals.pop());
            return self.literal(symbols, literal);
        }
        cnf::Formula::Or(
            literals
                .into_iter()
                .map(|literal| self.literal(symbols, literal))
                .collect::<Vec<_>>(),
        )
    }

    fn annotated_formula(
        &mut self,
        symbols: &mut Symbols,
        formula: st::AnnotatedFormula<'a>,
    ) -> (bool, bool, String, cnf::Formula) {
        let (is_cnf, name, mut role, mut formula) = match formula {
            st::AnnotatedFormula::Fof(fof) => {
                let formula =
                    self.fof_logic_formula(symbols, (*fof.formula).0);
                (false, format!("{}", fof.name), fof.role, formula)
            }
            st::AnnotatedFormula::Cnf(cnf) => {
                let formula = self.cnf_formula(symbols, *cnf.formula);
                (true, format!("{}", cnf.name), cnf.role, formula)
            }
        };
        debug_assert!(self.bound.is_empty());
        self.unbound.clear();
        self.fresh = 0;

        if role == st::FormulaRole::Conjecture {
            formula = formula.negated();
            role = st::FormulaRole::NegatedConjecture;
        }
        let conjecture = role == st::FormulaRole::NegatedConjecture;
        (is_cnf, conjecture, name, formula)
    }
}
