#[inline(always)]
pub(crate) fn unreachable() -> ! {
    unsafe { std::hint::unreachable_unchecked() }
    //unreachable!()
}
