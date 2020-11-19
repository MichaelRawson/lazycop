use crate::prelude::*;
use std::ffi::CStr;
use z3_sys::*;

pub(crate) struct Assertion;

pub(crate) struct Solver {
    context: Z3_context,
    solver: Z3_solver,
    bool_sort: Z3_sort,
    obj_sort: Z3_sort,
    signature: Block<Z3_func_decl>,
    scratch: Vec<Z3_ast>,
    fresh: Vec<(Id<Variable>, Z3_ast)>,
}

impl Solver {
    pub(crate) fn new(symbols: &Symbols) -> Self {
        unsafe {
            let context = Z3_mk_context(Z3_mk_config());
            let params = Z3_mk_params(context);
            Z3_params_inc_ref(context, params);
            let set_param = |param, value| {
                Z3_params_set_bool(
                    context,
                    params,
                    Z3_mk_string_symbol(
                        context,
                        CStr::from_bytes_with_nul_unchecked(param).as_ptr(),
                    ),
                    value,
                );
            };
            set_param(b"model\0", false);
            set_param(b"unsat_core\0", true);
            set_param(b"unsat_core\0", true);
            let solver = Z3_mk_solver_for_logic(
                context,
                Z3_mk_string_symbol(
                    context,
                    CStr::from_bytes_with_nul_unchecked(b"QF_UF\0").as_ptr(),
                ),
            );
            Z3_solver_inc_ref(context, solver);
            Z3_solver_set_params(context, solver, params);

            let bool_sort = Z3_mk_bool_sort(context);
            let obj_sort = Z3_mk_uninterpreted_sort(
                context,
                Z3_mk_int_symbol(context, 0),
            );
            let mut domain = vec![];
            let mut signature = Block::default();
            for id in symbols.range() {
                let symbol = &symbols[id];
                let range = if symbol.is_predicate {
                    bool_sort
                } else {
                    obj_sort
                };
                domain.resize(symbol.arity as usize, obj_sort);
                let symbol = Z3_mk_int_symbol(context, id.index() as i32);
                let sort = Z3_mk_func_decl(
                    context,
                    symbol,
                    domain.len() as u32,
                    domain.as_ptr(),
                    range,
                );
                signature.push(sort);
            }

            let scratch = vec![];
            let fresh = vec![];
            Self {
                context,
                solver,
                bool_sort,
                obj_sort,
                signature,
                scratch,
                fresh,
            }
        }
    }

    pub(crate) fn assert(&self, id: Id<Assertion>, assertion: Z3_ast) {
        unsafe {
            let symbol = Z3_mk_int_symbol(self.context, id.index() as i32);
            let label = Z3_mk_const(self.context, symbol, self.bool_sort);
            Z3_solver_assert_and_track(
                self.context,
                self.solver,
                assertion,
                label,
            );
        }
    }

    pub(crate) fn check(&self) -> bool {
        unsafe { Z3_solver_check(self.context, self.solver) == Z3_L_FALSE }
    }

    pub(crate) fn unsat_core(&self) -> Vec<Id<Assertion>> {
        let mut indices = vec![];
        let core =
            unsafe { Z3_solver_get_unsat_core(self.context, self.solver) };
        let length = unsafe { Z3_ast_vector_size(self.context, core) };
        for core_index in 0..length {
            let ast =
                unsafe { Z3_ast_vector_get(self.context, core, core_index) };
            let app = unsafe { Z3_to_app(self.context, ast) };
            let decl = unsafe { Z3_get_app_decl(self.context, app) };
            let symbol = unsafe { Z3_get_decl_name(self.context, decl) };
            let index = unsafe { Z3_get_symbol_int(self.context, symbol) };
            indices.push(Id::default() + Offset::new(index));
        }
        indices
    }

    pub(crate) fn ground<I: Iterator<Item = Clause>>(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        literals: &Literals,
        bindings: &Bindings,
        clauses: I,
    ) -> Z3_ast {
        for clause in clauses {
            let clause =
                self.ground_clause(symbols, terms, literals, bindings, clause);
            self.scratch.push(clause);
        }
        let grounding = unsafe {
            Z3_mk_and(
                self.context,
                self.scratch.len() as u32,
                self.scratch.as_ptr(),
            )
        };

        self.scratch.clear();
        self.fresh.clear();
        grounding
    }

    fn ground_clause(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        literals: &Literals,
        bindings: &Bindings,
        clause: Clause,
    ) -> Z3_ast {
        let mark = self.scratch.len();
        for literal in clause.original() {
            let literal = self.ground_literal(
                symbols,
                terms,
                bindings,
                &literals[literal],
            );
            self.scratch.push(literal);
        }
        let literals = &self.scratch[mark..];
        let clause = unsafe {
            Z3_mk_or(self.context, literals.len() as u32, literals.as_ptr())
        };
        self.scratch.truncate(mark);
        clause
    }

    fn ground_literal(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
        literal: &Literal,
    ) -> Z3_ast {
        let polarity = literal.polarity;
        let mut atom = match literal.atom {
            Atom::Predicate(p) => {
                self.ground_term(symbols, terms, bindings, p)
            }
            Atom::Equality(s, t) => {
                let s = self.ground_term(
                    symbols,
                    terms,
                    bindings,
                    bindings.resolve(s),
                );
                let t = self.ground_term(
                    symbols,
                    terms,
                    bindings,
                    bindings.resolve(t),
                );
                unsafe { Z3_mk_eq(self.context, s, t) }
            }
        };
        if !polarity {
            atom = unsafe { Z3_mk_not(self.context, atom) };
        }
        atom
    }

    fn ground_term(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
        term: Id<Term>,
    ) -> Z3_ast {
        match terms.view(symbols, term) {
            TermView::Variable(x) => {
                if let Some((_, fresh)) =
                    self.fresh.iter().find(|(y, _)| x == *y)
                {
                    *fresh
                } else {
                    let fresh = unsafe {
                        Z3_mk_fresh_const(
                            self.context,
                            std::ptr::null(),
                            self.obj_sort,
                        )
                    };
                    self.fresh.push((x, fresh));
                    fresh
                }
            }
            TermView::Function(f, ts) => {
                let mark = self.scratch.len();
                for t in ts {
                    let t = self.ground_term(
                        symbols,
                        terms,
                        bindings,
                        bindings.resolve(terms.resolve(t)),
                    );
                    self.scratch.push(t);
                }
                let args = &self.scratch[mark..];
                let app = unsafe {
                    Z3_mk_app(
                        self.context,
                        self.signature[f.transmute()],
                        args.len() as u32,
                        args.as_ptr(),
                    )
                };
                self.scratch.truncate(mark);
                app
            }
        }
    }
}
