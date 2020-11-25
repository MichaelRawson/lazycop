use std::os::raw::{c_char, c_int, c_uint};

pub(crate) const Z3_L_FALSE: i32 = -1;

#[repr(C)]
#[derive(Copy, Clone)]
struct blob {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct Z3_symbol(*mut blob);

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct Z3_ast(*mut blob);

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct Z3_ast_vector(*mut blob);

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct Z3_func_decl(*mut blob);

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct Z3_app(*mut blob);

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct Z3_sort(*mut blob);

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct Z3_params(*mut blob);

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct Z3_solver(*mut blob);

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct Z3_config(*mut blob);

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct Z3_context(*mut blob);

#[link(name = "z3")]
extern "C" {
    pub(crate) fn Z3_mk_int_symbol(c: Z3_context, i: c_int) -> Z3_symbol;
    pub(crate) fn Z3_mk_string_symbol(
        c: Z3_context,
        s: *const c_char,
    ) -> Z3_symbol;
    pub(crate) fn Z3_get_symbol_int(c: Z3_context, s: Z3_symbol) -> c_int;
    pub(crate) fn Z3_mk_bool_sort(c: Z3_context) -> Z3_sort;
    pub(crate) fn Z3_mk_uninterpreted_sort(
        c: Z3_context,
        s: Z3_symbol,
    ) -> Z3_sort;
    pub(crate) fn Z3_mk_fresh_const(
        c: Z3_context,
        prefix: *const c_char,
        sort: Z3_sort,
    ) -> Z3_ast;
    pub(crate) fn Z3_mk_fresh_func_decl(
        c: Z3_context,
        prefix: *const c_char,
        domain_size: c_uint,
        domain: *const Z3_sort,
        range: Z3_sort,
    ) -> Z3_func_decl;
    pub(crate) fn Z3_get_decl_name(
        c: Z3_context,
        decl: Z3_func_decl,
    ) -> Z3_symbol;
    pub(crate) fn Z3_mk_const(
        c: Z3_context,
        s: Z3_symbol,
        sort: Z3_sort,
    ) -> Z3_ast;
    pub(crate) fn Z3_mk_app(
        c: Z3_context,
        f: Z3_func_decl,
        num_args: c_uint,
        args: *const Z3_ast,
    ) -> Z3_ast;
    pub(crate) fn Z3_get_app_decl(c: Z3_context, app: Z3_app) -> Z3_func_decl;
    pub(crate) fn Z3_to_app(c: Z3_context, app: Z3_ast) -> Z3_app;
    pub(crate) fn Z3_mk_eq(
        c: Z3_context,
        left: Z3_ast,
        right: Z3_ast,
    ) -> Z3_ast;
    pub(crate) fn Z3_mk_not(c: Z3_context, f: Z3_ast) -> Z3_ast;
    pub(crate) fn Z3_mk_and(
        c: Z3_context,
        num_args: c_uint,
        args: *const Z3_ast,
    ) -> Z3_ast;
    pub(crate) fn Z3_mk_or(
        c: Z3_context,
        num_args: c_uint,
        args: *const Z3_ast,
    ) -> Z3_ast;
    pub(crate) fn Z3_translate(
        source: Z3_context,
        a: Z3_ast,
        target: Z3_context,
    ) -> Z3_ast;
    pub(crate) fn Z3_ast_vector_size(
        c: Z3_context,
        v: Z3_ast_vector,
    ) -> c_uint;
    pub(crate) fn Z3_ast_vector_get(
        c: Z3_context,
        v: Z3_ast_vector,
        index: c_uint,
    ) -> Z3_ast;
    pub(crate) fn Z3_mk_params(c: Z3_context) -> Z3_params;
    pub(crate) fn Z3_params_set_bool(c: Z3_context, p: Z3_params, param: Z3_symbol, value: bool);
    pub(crate) fn Z3_params_inc_ref(c: Z3_context, p: Z3_params);
    pub(crate) fn Z3_params_dec_ref(c: Z3_context, p: Z3_params);
    pub(crate) fn Z3_mk_solver_for_logic(
        c: Z3_context,
        logic: Z3_symbol,
    ) -> Z3_solver;
    pub(crate) fn Z3_solver_set_params(
        c: Z3_context,
        s: Z3_solver,
        p: Z3_params,
    );
    pub(crate) fn Z3_solver_inc_ref(c: Z3_context, s: Z3_solver);
    pub(crate) fn Z3_solver_check(c: Z3_context, s: Z3_solver) -> i32;
    pub(crate) fn Z3_solver_assert_and_track(
        c: Z3_context,
        s: Z3_solver,
        a: Z3_ast,
        p: Z3_ast,
    );
    pub(crate) fn Z3_solver_get_unsat_core(
        c: Z3_context,
        s: Z3_solver,
    ) -> Z3_ast_vector;
    pub(crate) fn Z3_mk_config() -> Z3_config;
    pub(crate) fn Z3_mk_context(config: Z3_config) -> Z3_context;
}
