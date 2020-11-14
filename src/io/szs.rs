pub(crate) fn os_error() {
    println!("% SZS status OSError");
}

pub(crate) fn syntax_error() {
    println!("% SZS status SyntaxError");
}

pub(crate) fn inappropriate() {
    println!("% SZS status Inappropriate");
}

pub(crate) fn time_out(name: &str) {
    println!("% SZS status TimeOut for {}", name);
}

pub(crate) fn theorem(name: &str) {
    println!("% SZS status Theorem for {}", name);
}

pub(crate) fn unsatisfiable(name: &str) {
    println!("% SZS status Unsatisfiable for {}", name);
}

pub(crate) fn gave_up(name: &str) {
    println!("% SZS status Unknown for {}", name);
    println!("% help: conjecture did not follow from axioms");
    println!("% help: axioms could still be contradictory");
}

pub(crate) fn satisfiable(name: &str) {
    println!("% SZS status Satisfiable for {}", name);
}

pub(crate) fn counter_satisfiable(name: &str) {
    println!("% SZS status CounterSatisfiable for {}", name);
}

pub(crate) fn begin_proof(name: &str) {
    println!("% SZS output begin Proof for {}", name);
}

pub(crate) fn end_proof(name: &str) {
    println!("% SZS output end Proof for {}", name);
}
