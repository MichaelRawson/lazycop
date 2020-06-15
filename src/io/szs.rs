pub(crate) fn os_error() {
    println!("% SZS status OSError");
}

pub(crate) fn input_error() {
    println!("% SZS status InputError");
}

pub(crate) fn inappropriate() {
    println!("% SZS status Inappropriate");
}

pub(crate) fn unknown() {
    println!("% SZS status Unknown");
}

pub(crate) fn unsatisfiable() {
    println!("% SZS status Unsatisfiable");
}

pub(crate) fn begin_incomplete_proof() {
    println!("% SZS output begin IncompleteProof");
}

pub(crate) fn end_incomplete_proof() {
    println!("% SZS output end IncompleteProof");
}
