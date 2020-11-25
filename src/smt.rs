use crate::prelude::*;
use std::ffi::CStr;
use z3_sys::*;

pub(crate) struct Assertion(Z3_ast);
unsafe impl Send for Assertion {}

pub(crate) struct Context(Z3_context);
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Default for Context {
    fn default() -> Self {
        Self(unsafe { Z3_mk_context(Z3_mk_config()) })
    }
}

impl Context {
    pub(crate) fn translate(
        &mut self,
        other: &Self,
        assertion: Assertion,
    ) -> Assertion {
        Assertion(unsafe { Z3_translate(other.0, assertion.0, self.0) })
    }
}

pub(crate) struct Grounder {
    context: Z3_context,
    //cache: LUT<Term, Z3_ast>
    var: Z3_ast,
    signature: Block<Z3_func_decl>,
    scratch: Vec<Z3_ast>,
}

pub(crate) struct Solver {
    context: Z3_context,
    solver: Z3_solver,
    bool_sort: Z3_sort,
}

impl Solver {
    pub(crate) fn new(context: &Context) -> Self {
        let context = context.0;
        let solver = unsafe {
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
            set_param(b"unsat_core\0", true);
            set_param(b"core.minimize\0", true);
            let solver = Z3_mk_solver_for_logic(
                context,
                Z3_mk_string_symbol(
                    context,
                    CStr::from_bytes_with_nul_unchecked(b"QF_UF\0").as_ptr(),
                ),
            );
            Z3_solver_inc_ref(context, solver);
            Z3_solver_set_params(context, solver, params);
            Z3_params_dec_ref(context, params);
            solver
        };
        let bool_sort = unsafe { Z3_mk_bool_sort(context) };
        Self {
            context,
            solver,
            bool_sort,
        }
    }

    pub(crate) fn assert(&mut self, id: Id<Assertion>, assertion: Assertion) {
        unsafe {
            let symbol = Z3_mk_int_symbol(self.context, id.index() as i32);
            let label = Z3_mk_const(self.context, symbol, self.bool_sort);
            Z3_solver_assert_and_track(
                self.context,
                self.solver,
                assertion.0,
                label,
            );
        }
    }

    pub(crate) fn check(&mut self) -> bool {
        let result = unsafe { Z3_solver_check(self.context, self.solver) };
        result == Z3_L_FALSE
    }

    pub(crate) fn unsat_core(&self) -> Vec<Id<Assertion>> {
        let mut indices = vec![];
        let core =
            unsafe { Z3_solver_get_unsat_core(self.context, self.solver) };
        let length = unsafe { Z3_ast_vector_size(self.context, core) };
        for core_index in 0..length {
            let index = unsafe {
                let ast = Z3_ast_vector_get(self.context, core, core_index);
                let app = Z3_to_app(self.context, ast);
                let decl = Z3_get_app_decl(self.context, app);
                let symbol = Z3_get_decl_name(self.context, decl);
                Z3_get_symbol_int(self.context, symbol)
            };
            indices.push(Id::default() + Offset::new(index));
        }
        indices
    }
}

impl Grounder {
    pub(crate) fn new(context: &Context, symbols: &Symbols) -> Self {
        let context = context.0;
        let bool_sort = unsafe { Z3_mk_bool_sort(context) };
        let obj_sort = unsafe {
            Z3_mk_uninterpreted_sort(
                context,
                Z3_mk_string_symbol(
                    context,
                    CStr::from_bytes_with_nul_unchecked(b"obj\0").as_ptr(),
                ),
            )
        };
        let var =
            unsafe { Z3_mk_fresh_const(context, std::ptr::null(), obj_sort) };
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
            let sort = unsafe {
                Z3_mk_fresh_func_decl(
                    context,
                    std::ptr::null(),
                    domain.len() as u32,
                    domain.as_ptr(),
                    range,
                )
            };
            signature.push(sort);
        }

        let scratch = vec![];
        Self {
            context,
            var,
            signature,
            scratch,
        }
    }

    pub(crate) fn ground<I: Iterator<Item = Clause>>(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
        literals: &Literals,
        clauses: I,
    ) -> Assertion {
        for clause in clauses {
            let clause =
                self.ground_clause(symbols, terms, bindings, literals, clause);
            self.scratch.push(clause);
        }
        let ast = unsafe {
            Z3_mk_and(
                self.context,
                self.scratch.len() as u32,
                self.scratch.as_ptr(),
            )
        };
        self.scratch.clear();
        Assertion(ast)
    }

    fn ground_clause(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
        literals: &Literals,
        clause: Clause,
    ) -> Z3_ast {
        let mark = self.scratch.len();
        for literal in clause.original().into_iter() {
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
        let mut atom =
            self.ground_atom(symbols, terms, bindings, &literal.atom);
        if !literal.polarity {
            atom = unsafe { Z3_mk_not(self.context, atom) };
        }
        atom
    }

    fn ground_atom(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
        atom: &Atom,
    ) -> Z3_ast {
        match atom {
            Atom::Predicate(p) => {
                self.ground_term(symbols, terms, bindings, *p)
            }
            Atom::Equality(s, t) => {
                let s = self.ground_term(
                    symbols,
                    terms,
                    bindings,
                    bindings.resolve(*s),
                );
                let t = self.ground_term(
                    symbols,
                    terms,
                    bindings,
                    bindings.resolve(*t),
                );
                unsafe { Z3_mk_eq(self.context, s, t) }
            }
        }
    }

    fn ground_term(
        &mut self,
        symbols: &Symbols,
        terms: &Terms,
        bindings: &Bindings,
        term: Id<Term>,
    ) -> Z3_ast {
        match terms.view(symbols, term) {
            TermView::Variable(_) => self.var,
            TermView::Function(f, ts) => {
                let mark = self.scratch.len();
                for t in ts {
                    let term = self.ground_term(
                        symbols,
                        terms,
                        bindings,
                        bindings.resolve(terms.resolve(t)),
                    );
                    self.scratch.push(term);
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
