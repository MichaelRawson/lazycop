#[inline(always)]
pub(crate) fn unreachable() -> ! {
    debug_assert!(false, "unreachable");
    unsafe { std::hint::unreachable_unchecked() }
}

#[inline(always)]
pub(crate) fn some<T>(some: Option<T>) -> T {
    if let Some(value) = some {
        value
    } else {
        unreachable()
    }
}
