#[inline(always)]
pub(crate) fn unreachable() -> ! {
    debug_assert!(false, "unreachable");
    unsafe { std::hint::unreachable_unchecked() }
}
