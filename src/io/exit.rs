use std::process::exit;

pub fn success() -> ! {
    exit(0)
}

pub fn failure() -> ! {
    exit(1)
}
