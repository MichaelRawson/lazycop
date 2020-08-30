#[inline(always)]
#[track_caller]
pub(crate) fn unreachable() -> ! {
    debug_assert!(false, "unreachable");
    unsafe { std::hint::unreachable_unchecked() }
}

#[inline(always)]
#[track_caller]
pub(crate) fn some<T>(some: Option<T>) -> T {
    if let Some(value) = some {
        value
    } else {
        unreachable()
    }
}
