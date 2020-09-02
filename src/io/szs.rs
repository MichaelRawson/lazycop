pub(crate) fn os_error() {
    println!("% SZS status OSError");
}

pub(crate) fn syntax_error() {
    println!("% SZS status SyntaxError");
}

pub(crate) fn inappropriate() {
    println!("% SZS status Inappropriate");
}

pub(crate) fn resource_out(name: &str) {
    println!("% SZS status ResourceOut for {}", name);
}

pub(crate) fn theorem(name: &str) {
    println!("% SZS status Theorem for {}", name);
}

pub(crate) fn unsatisfiable(name: &str) {
    println!("% SZS status Unsatisfiable for {}", name);
}

pub(crate) fn counter_satisfiable(name: &str) {
    println!("% SZS status CounterSatisfiable for {}", name);
}

pub(crate) fn satisfiable(name: &str) {
    println!("% SZS status Satisfiable for {}", name);
}

pub(crate) fn begin_cnf_refutation(name: &str) {
    println!("% SZS output begin CNFRefutation for {}", name);
}

pub(crate) fn end_cnf_refutation(name: &str) {
    println!("% SZS output end CNFRefutation for {}", name);
}
