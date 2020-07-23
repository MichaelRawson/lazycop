use std::num::NonZeroU32;

pub(crate) fn some<T>(some: Option<T>) -> T {
    if let Some(value) = some {
        value
    } else {
        unreachable()
    }
}

pub(crate) fn non_zero(nz: u32) -> NonZeroU32 {
    debug_assert!(nz != 0, "non-zero value should not be 0");
    unsafe { NonZeroU32::new_unchecked(nz) }
}

pub(crate) fn unreachable() -> ! {
    debug_assert!(false, "unreachable");
    unsafe { std::hint::unreachable_unchecked() }
}
