//! bit twiddling operations
#![allow(unused)]

pub fn bit_clr(x: u32, bit: u32) -> u32 {
    assert!(bit < 32);
    x & !(1 << bit)
}

pub fn bit_set(x: u32, bit: u32) -> u32 {
    assert!(bit < 32);
    x | (1 << bit)
}

pub fn bit_not(x: u32, bit: u32) -> u32 {
    assert!(bit < 32);
    x ^ (1 << bit)
}

pub fn bit_is_on(x: u32, bit: u32) -> bool {
    assert!(bit < 32);
    (x >> bit) & 1 != 0
}

pub fn bit_is_off(x: u32, bit: u32) -> bool {
    !bit_is_on(x, bit)
}

pub fn bits_mask(nbits: u32) -> u32 {
    if nbits == 32 {
        return !0;
    }
    assert!(nbits < 32);
    (1 << nbits) - 1
}

pub fn bits_get(x: u32, lb: u32, ub: u32) -> u32 {
    assert!(lb <= ub);
    assert!(ub < 32);
    (x >> lb) & bits_mask(ub - lb + 1)
}

pub fn bits_clr(x: u32, lb: u32, ub: u32) -> u32 {
    assert!(lb <= ub);
    assert!(ub < 32);

    let mask = bits_mask(ub - lb + 1);
    x & !(mask << lb)
}

pub fn bits_set(x: u32, lb: u32, ub: u32, v: u32) -> u32 {
    assert!(lb <= ub);
    assert!(ub < 32);

    let n = ub - lb + 1;
    assert!(n <= 32);
    assert!((bits_mask(n) & v) == v);

    bits_clr(x, lb, ub) | (v << lb)
}

pub fn bits_eq(x: u32, lb: u32, ub: u32, val: u32) -> bool {
    assert!(lb <= ub);
    assert!(ub < 32);
    bits_get(x, lb, ub) == val
}

pub fn bit_count(x: u32) -> u32 {
    let mut cnt = 0;
    for i in 0..32 {
        if bit_is_on(x, i) {
            cnt += 1;
        }
    }
    cnt
}

pub fn bits_union(x: u32, y: u32) -> u32 {
    x | y
}

pub fn bits_intersect(x: u32, y: u32) -> u32 {
    x & y
}

pub fn bits_not(x: u32) -> u32 {
    !x
}

pub fn bits_diff(a: u32, b: u32) -> u32 {
    bits_intersect(a, bits_not(b))
}
