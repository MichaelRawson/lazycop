use crate::prelude::*;
use z3::ast::Ast;

pub(crate) fn context() -> z3::Context {
    z3::Context::new(&z3::Config::default())
}

pub(crate) struct Solver<'ctx> {
    context: &'ctx z3::Context,
    solver: z3::Solver<'ctx>,
    var: z3::ast::Dynamic<'ctx>,
    signature: Block<z3::FuncDecl<'ctx>>,
}

impl<'ctx> Solver<'ctx> {
    pub(crate) fn new(context: &'ctx z3::Context, symbols: &Symbols) -> Self {
        let mut params = z3::Params::new(context);
        params.set_bool("model", false);
        params.set_bool("unsat_core", true);
        params.set_bool("combined_solver.ignore_solver1", true);
        params.set_bool("smt.minimize_unsat_cores", true);
        params.set_symbol("smt.logic", "QF_UF");
        let solver = z3::Solver::new(context);
        solver.set_params(&params);

        let o = z3::Sort::bool(context);
        let i = z3::Sort::uninterpreted(&context, "i".into());
        let x = z3::FuncDecl::new(context, "x", &[], &i);
        let var = x.apply(&[]);
        let mut domain = vec![];
        let mut signature = Block::default();
        for id in symbols.range() {
            let symbol = &symbols[id];
            let range = if symbol.is_predicate { &o } else { &i };
            domain.resize(symbol.arity as usize, &i);
            let sort = z3::FuncDecl::new(context, id.index(), &domain, range);
            signature.push(sort);
        }

        Self {
            context,
            solver,
            var,
            signature,
        }
    }

    pub(crate) fn assert(&self, id: u32, assertion: z3::ast::Bool<'ctx>) {
        let label = z3::ast::Bool::new_const(self.context, id);
        self.solver.assert_and_track(&assertion, &label);
    }

    pub(crate) fn check(&self) -> bool {
        self.solver.check() == z3::SatResult::Unsat
    }

    pub(crate) fn ground<I: Iterator<Item = Clause>>(
        &self,
        symbols: &Symbols,
        terms: &Terms,
        literals: &Literals,
        bindings: &Bindings,
        clauses: I,
    ) -> z3::ast::Bool<'ctx> {
        let clauses = clauses
            .map(|clause| {
                let clause = clause
                    .original()
                    .into_iter()
                    .map(|literal| {
                        let literal = &literals[literal];
                        let polarity = literal.polarity;
                        let mut atom = match literal.atom {
                            Atom::Predicate(p) => {
                                let p = self
                                    .ground_term(symbols, terms, bindings, p);
                                unwrap(p.as_bool())
                            }
                            Atom::Equality(s, t) => {
                                let s = self.ground_term(
                                    symbols,
                                    terms,
                                    bindings,
                                    bindings.resolve(terms, s),
                                );
                                let t = self.ground_term(
                                    symbols,
                                    terms,
                                    bindings,
                                    bindings.resolve(terms, t),
                                );
                                s._eq(&t)
                            }
                        };
                        if !polarity {
                            atom = atom.not();
                        }
                        atom
                    })
                    .collect::<Vec<_>>();
                let refs = clause.iter().collect::<Vec<_>>();
                z3::ast::Bool::or(self.context, &refs)
            })
            .collect::<Vec<_>>();
        let refs = clauses.iter().collect::<Vec<_>>();
        z3::ast::Bool::and(self.context, &refs)
    }

    fn ground_term(
        &self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
        term: Id<Term>,
    ) -> z3::ast::Dynamic<'ctx> {
        match terms.view(symbols, term) {
            TermView::Variable(_) => self.var.clone(),
            TermView::Function(f, ts) => {
                let ts = ts
                    .into_iter()
                    .map(|t| {
                        self.ground_term(
                            symbols,
                            terms,
                            bindings,
                            bindings.resolve(terms, terms.resolve(t)),
                        )
                    })
                    .collect::<Vec<_>>();
                let refs = ts.iter().collect::<Vec<_>>();
                self.signature[f.transmute()].apply(&refs)
            }
        }
    }
}
