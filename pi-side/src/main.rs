#![no_std]
#![no_main]
#![feature(sync_unsafe_cell)]

// os stuff
mod boot;
mod comm;
mod threading;
#[macro_use]
mod syscalls;

mod utils;

// lisp stuff
mod language;

extern crate alloc;

// use alloc::format;
// use alloc::string::String;
// use comm::uart::PiUart;
// use shared::Framer;


fn main() {
    // Pile every interaction into one form.
    //   * nested `let`s with locals that *should* fold through (TODO:
    //     LoadLocal is still Bottom so this won't actually propagate
    //     until we teach SCCP about immutable locals — the test shape
    //     is correct, the IR will just be conservatively pessimistic).
    //   * shadowing — inner `x` gets a fresh LocalId and reads OUTER x;
    //     verifies we don't accidentally fold across scopes.
    //   * one `if` with a literal-derived cond — entire else-arm should
    //     become a dead block; its compute chain DCEs.
    //   * one `if` with a capture-derived cond (`sky`) — both arms live,
    //     pure intermediates whose value goes nowhere should still DCE.
    //   * leading begin expressions whose values are discarded — full
    //     cascade DCE of their compute trees.
    let mut img = language::Image::new();
    let expr = language::parse("
(begin
  (set sky 7)
  (ir4
    (let (x (mul 2 3)                  ; outer x = 6
          y (add 10 5))                ; y = 15
      (let (
            ; through-let: today still LoadLocal Bottom; future work
            z (add x y)                ; would fold to 21 someday
            ; shadowing: inner x is a NEW LocalId; uses OUTER x → Bottom
            x (sub x 1))               ; inner x ≠ outer x
        (begin
          ; discarded entirely — full cascade DCE
          (mul (add 3 3) (sub 100 50))
          ((lambda (i) (+ i 3)) x)
          ; constant cond → else-arm dead block + cascade DCE
          (if (gt (add 1 1) 0)
            (add 100 200)              ; folds to 300, then DCE (begin discards)
            (div 1 0))                  ; unreachable dead block
          ; capture cond → both arms live; phi feeds ret
          (if (lt sky 100)
            (add x z)                  ; locals, Bottom — survives via phi → ret
            (mul x z)))))))             ; same; this if's result is the begin value
").unwrap();
    let result = language::evaluate(expr.into(), &mut img).unwrap();
    println!("{}\n", result);
}
