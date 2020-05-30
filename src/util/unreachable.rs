use std::num::NonZeroU32;

#[inline(always)]
pub(crate) fn non_zero(nz: u32) -> NonZeroU32 {
    debug_assert!(nz != 0);
    unsafe { NonZeroU32::new_unchecked(nz) }
}

#[inline(always)]
pub(crate) fn unreachable() -> ! {
    debug_assert!(false, "unreachable");
    unsafe { std::hint::unreachable_unchecked() }
}
