use std::process::exit;

pub(crate) fn success() -> ! {
    exit(0)
}

pub(crate) fn failure() -> ! {
    exit(1)
}
