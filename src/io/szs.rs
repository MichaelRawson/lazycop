pub(crate) fn os_error() {
    println!("% SZS status OSError");
}

pub(crate) fn input_error() {
    println!("% SZS status InputError");
}

pub(crate) fn inappropriate() {
    println!("% SZS status Inappropriate");
}

pub(crate) fn incomplete() {
    println!("% SZS status Incomplete");
}

pub(crate) fn unsatisfiable() {
    println!("% SZS status Unsatisfiable");
}

pub(crate) fn begin_refutation() {
    println!("% SZS begin CNFRefutation");
}

pub(crate) fn end_refutation() {
    println!("% SZS end CNFRefutation");
}
