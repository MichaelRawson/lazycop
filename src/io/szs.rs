pub(crate) fn os_error() {
    println!("% SZS status OSError");
}

pub(crate) fn input_error() {
    println!("% SZS status InputError");
}

pub(crate) fn inappropriate() {
    println!("% SZS status Inappropriate");
}

pub(crate) fn gave_up(name: &str) {
    println!("% SZS status GaveUp for {}", name);
}

pub(crate) fn unsatisfiable(name: &str) {
    println!("% SZS status Unsatisfiable for {}", name);
}

pub(crate) fn begin_cnf_refutation(name: &str) {
    println!("% SZS output begin CNFRefutation for {}", name);
}

pub(crate) fn end_cnf_refutation(name: &str) {
    println!("% SZS output end CNFRefutation for {}", name);
}
