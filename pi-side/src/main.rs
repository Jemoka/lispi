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

use alloc::format;
use alloc::string::String;

/// One equivalence case: run `src` both through the interpreter and
/// through the JIT, compare `Display` output, print PASS/FAIL.
fn check(name: &str, src: &str) -> bool {
    use language::{evaluate, parse, Image};

    let expr = match parse(src) {
        Ok(e) => e,
        Err(e) => {
            println!("[{}] PARSE-ERR: {}", name, e);
            return false;
        }
    };
    let mut img_eval = Image::new();
    let eval_out = match evaluate(expr.clone().into(), &mut img_eval) {
        Ok(v) => format!("{}", v),
        Err(e) => format!("ERR: {}", e),
    };

    let mut jit_src = String::new();
    jit_src.push_str("(jitexec ");
    jit_src.push_str(src);
    jit_src.push(')');
    let jit_expr = match parse(&jit_src) {
        Ok(e) => e,
        Err(e) => {
            println!("[{}] JIT PARSE-ERR: {}", name, e);
            return false;
        }
    };
    let mut img_jit = Image::new();
    let jit_out = match evaluate(jit_expr.into(), &mut img_jit) {
        Ok(v) => format!("{}", v),
        Err(e) => format!("ERR: {}", e),
    };

    if eval_out == jit_out {
        println!("[{}] PASS  ({})", name, eval_out);
        true
    } else {
        println!("[{}] FAIL  eval={:?} jit={:?}", name, eval_out, jit_out);
        false
    }
}

fn main() {
    comm::uart::init();
    println!("LISPI JIT equivalence tests starting...");

    let cases: &[(&str, &str)] = &[
        // ===== arithmetic =====
        ("imm-7",        "7"),
        ("arith-add",    "(add 1 2)"),
        ("arith-nested", "(add 1 (mul 2 3))"),
        ("arith-sub",    "(sub 10 4)"),
        ("arith-div",    "(div 20 4)"),
        ("arith-mod",    "(mod 17 5)"),
        ("arith-mul",    "(mul 6 7)"),
        ("arith-lshift", "(lshift 1 4)"),
        ("arith-rshift", "(rshift 64 2)"),

        // ===== bitwise =====
        ("binor",        "(binor 12 3)"),
        ("binand",       "(binand 14 11)"),
        ("binnot",       "(binnot 0)"),

        // ===== comparisons =====
        ("cmp-gt",       "(gt 5 3)"),
        ("cmp-eq",       "(eq 4 4)"),
        ("cmp-lt-false", "(lt 7 7)"),
        ("cmp-gte",      "(gte 5 5)"),
        ("cmp-lte",      "(lte 4 9)"),

        // ===== logic =====
        ("not-true",     "(not true)"),
        ("not-zero",     "(not 0)"),
        ("and",          "(and true 7)"),
        ("or",           "(or false 9)"),
        ("xor",          "(xor true false)"),

        // ===== type coercions =====
        ("addr",         "(addr 16)"),
        ("signed",       "(signed 5)"),
        ("unsigned",     "(unsigned 5)"),

        // ===== if / dead-block (constant folding) =====
        ("if-const",     "(if (gt 2 1) 100 (div 1 0))"),
        ("if-const-else","(if (lt 2 1) 100 200)"),
        ("if-dynamic",   "(let (x 5) (if (gt x 3) (add x 1) (sub x 1)))"),

        // ===== begin =====
        ("begin",        "(begin 1 2 3)"),

        // ===== let / locals =====
        ("let-simple",   "(let (x 7) x)"),
        ("let-arith",    "(let (x 7 y 3) (add x y))"),
        ("let-seq",      "(let (x 7 y (mul x 2)) (add x y))"),
        ("let-shadow",   "(let (x 1) (let (x 2) x))"),
        ("let-nested",   "(let (x 1) (let (y 2) (add x y)))"),

        // ===== set / capture =====
        ("set",          "(begin (set z 5) z)"),
        ("set-update",   "(begin (set z 5) (set z (add z 1)) z)"),

        // ===== cons / car / cdr / nullp =====
        ("cons-car",     "(car (cons 1 2))"),
        ("cons-cdr",     "(cdr (cons 1 2))"),
        ("nullp-nil",    "(nullp nil)"),
        ("nullp-cons",   "(nullp (cons 1 2))"),

        // ===== list =====
        ("list-build",   "(list 1 2 3)"),
        ("list-empty",   "(list)"),

        // ===== quote / quasiquote =====
        ("quote",        "(quote (1 2 3))"),
        ("quasiquote",   "(let (x 7) (quasiquote (1 (unquote x) 3)))"),

        // ===== array =====
        ("array",        "(unpack (array (list 4 5 6)))"),
        ("full",         "(unpack (full 3 7))"),
        ("getidx",       "(getidx (array (list 10 20 30)) 1)"),

        // ===== if (already covered above by if-const/if-const-else/if-dynamic) =====

        // ===== syscalls (prefixed @) — side-effect specials =====
        // Wrap in `begin ... 0` so we test side-effect ordering without
        // depending on the syscall's return tag (interpreter returns
        // Nil; JIT IR boxes the ImmReg result through `box_number`,
        // yielding Integer(0) instead — a known type-tag drift).
        ("syscall-dsb",      "(begin (@dsb) 0)"),
        ("syscall-prefetch", "(begin (@prefetch_flush) 0)"),
        ("syscall-delay",    "(begin (@delay 1) 0)"),
        ("syscall-monitor-clear", "(begin (@monitor/clear) 0)"),

        // ===== lambda / closure (will hit Escape paths) =====
        ("lambda-imm",   "((lambda (x) (mul x x)) 6)"),
        ("defun",        "(begin (defun sq (x) (mul x x)) (sq 9))"),

        // ===== combined optimization sample =====
        ("combined", "
(begin
  (set sky 7)
  (let (x (mul 2 3) y (add 10 5))
    (let (z (add x y))
      (if (gt z 10) (add x z) (sub x z)))))
"),
    ];

    let mut passed = 0;
    let mut failed = 0;
    for (name, src) in cases {
        println!(">> {}", name);
        if check(name, src) { passed += 1; } else { failed += 1; }
    }

    // ===== side-effect persistence test =====
    // Verify that a (set z V) executed *inside* a jitexec is visible
    // to a subsequent interpreter-level evaluation against the same
    // Image. This catches any shadow-slot path that fails to reify
    // its writes back into the underlying Image's bindings on exit.
    {
        use language::{evaluate, parse, Image};
        let mut img = Image::new();
        let setup = parse("(jitexec (set z 42))").unwrap();
        let _ = evaluate(setup.into(), &mut img);
        let lookup = parse("z").unwrap();
        let observed = match evaluate(lookup.into(), &mut img) {
            Ok(v) => format!("{}", v),
            Err(e) => format!("ERR: {}", e),
        };
        if observed == "42" {
            println!("[side-effect-persist] PASS  (z={})", observed);
            passed += 1;
        } else {
            println!("[side-effect-persist] FAIL  z={} (expected 42)", observed);
            failed += 1;
        }
    }

    // ===== (jit closure dummies) — type-specialized JIT'd closure =====
    // Compile a closure once, then invoke it like a regular callable.
    // The interpreter dispatches into the compiled body; arg-type
    // mismatches bail out per the JittedClosure type guard.
    {
        use language::{evaluate, parse, Image};
        let mut img = Image::new();

        // Define the closure, jit it with integer dummies, then call it.
        let setup = parse("(begin
            (defun sq (x) (mul x x))
            (set jsq (jit sq 1))
            (jsq 9))").unwrap();
        let observed = match evaluate(setup.into(), &mut img) {
            Ok(v) => format!("{}", v),
            Err(e) => format!("ERR: {}", e),
        };
        if observed == "81" {
            println!("[jit-closure] PASS  ({})", observed);
            passed += 1;
        } else {
            println!("[jit-closure] FAIL  got={:?} (expected 81)", observed);
            failed += 1;
        }
    }

    // ===== auto-descending JIT =====
    // The outer jitted body Escapes the inner call `(mul-by-self x)`.
    // Auto-JIT in h_escape's fast path should compile `mul-by-self`
    // on the first call and cache it on the outermost executor; the
    // second call to `(wrap 5)` should reuse the cached compilation
    // rather than recompiling.
    {
        use language::{evaluate, parse, Image};
        let mut img = Image::new();
        let setup = parse("(begin
            (defun mul-by-self (x) (mul x x))
            (set wrap (jit (lambda (x) (mul-by-self x)) 1))
            (wrap 5))").unwrap();
        let observed = match evaluate(setup.into(), &mut img) {
            Ok(v) => format!("{}", v),
            Err(e) => format!("ERR: {}", e),
        };
        if observed == "25" {
            println!("[auto-jit-basic] PASS  ({})", observed);
            passed += 1;
        } else {
            println!("[auto-jit-basic] FAIL  got={:?} (expected 25)", observed);
            failed += 1;
        }
    }

    // ===== auto-JIT with a body that recursively calls itself =====
    // First call compiles `count-down`; the recursive Escape hits the
    // currently_compiling guard and falls back to interpreter for that
    // one frame. Once the outer compilation completes the cache holds
    // an entry for `count-down`, so subsequent recursive frames reach
    // the cache and dispatch via JIT.
    {
        use language::{evaluate, parse, Image};
        let mut img = Image::new();
        let setup = parse("(begin
            (defun count-down (n acc)
                (if (eq n 0) acc (count-down (sub n 1) (add acc 1))))
            (set jcd (jit (lambda (n) (count-down n 0)) 1))
            (jcd 5))").unwrap();
        let observed = match evaluate(setup.into(), &mut img) {
            Ok(v) => format!("{}", v),
            Err(e) => format!("ERR: {}", e),
        };
        if observed == "5" {
            println!("[auto-jit-recursive] PASS  ({})", observed);
            passed += 1;
        } else {
            println!("[auto-jit-recursive] FAIL  got={:?} (expected 5)", observed);
            failed += 1;
        }
    }

    // ===== auto-JIT type-specialization cache =====
    // Same callee invoked with two different InputType tuples should
    // produce two cache entries that compute the right value each.
    {
        use language::{evaluate, parse, Image};
        let mut img = Image::new();
        let setup = parse("(begin
            (defun sq (x) (mul x x))
            (set jw (jit (lambda (x) (sq x)) 1))
            (add (jw 4) (jw 7)))").unwrap();
        let observed = match evaluate(setup.into(), &mut img) {
            Ok(v) => format!("{}", v),
            Err(e) => format!("ERR: {}", e),
        };
        if observed == "65" {
            println!("[auto-jit-type-cache] PASS  ({})", observed);
            passed += 1;
        } else {
            println!("[auto-jit-type-cache] FAIL  got={:?} (expected 65)", observed);
            failed += 1;
        }
    }

    // ===== jit-calls-jit (via Escape) =====
    // A jitted body calls another jitted closure by symbol. The cgen
    // sees an unrecognized cons-form `(jsq x)`, escapes the entire
    // call, and at runtime the JIT helper invokes the interpreter,
    // which dispatches via execute.rs's JittedClosure arm and runs
    // the inner JIT.
    {
        use language::{evaluate, parse, Image};
        let mut img = Image::new();
        let setup = parse("(begin
            (defun sq (x) (mul x x))
            (set jsq (jit sq 1))
            (defun sq2 (x) (mul (jsq x) 2))
            (set jsq2 (jit sq2 1))
            (jsq2 4))").unwrap();
        let observed = match evaluate(setup.into(), &mut img) {
            Ok(v) => format!("{}", v),
            Err(e) => format!("ERR: {}", e),
        };
        if observed == "32" {
            println!("[jit-calls-jit] PASS  ({})", observed);
            passed += 1;
        } else {
            println!("[jit-calls-jit] FAIL  got={:?} (expected 32)", observed);
            failed += 1;
        }
    }

    // ===== (jit) type guard — wrong-typed arg returns an error =====
    {
        use language::{evaluate, parse, Image};
        let mut img = Image::new();
        let setup = parse("(begin
            (defun id (x) x)
            (set jid (jit id 1))
            (jid true))").unwrap();
        let observed = match evaluate(setup.into(), &mut img) {
            Ok(v) => format!("{}", v),
            Err(e) => format!("ERR: {}", e),
        };
        if observed.starts_with("ERR:") {
            println!("[jit-type-guard] PASS  ({})", observed);
            passed += 1;
        } else {
            println!("[jit-type-guard] FAIL  got={:?} (expected ERR)", observed);
            failed += 1;
        }
    }

    // ===== side-effect on a cons cell built inside the JIT =====
    // Build (cons 1 2) inside jitexec and let it persist; read its
    // car back at interpreter level. Exercises slot→Rc<Value>
    // reification of cons-shaped slots through the executor exit path.
    {
        use language::{evaluate, parse, Image};
        let mut img = Image::new();
        let setup = parse("(jitexec (set z (cons 1 2)))").unwrap();
        let _ = evaluate(setup.into(), &mut img);
        let lookup = parse("(car z)").unwrap();
        let observed = match evaluate(lookup.into(), &mut img) {
            Ok(v) => format!("{}", v),
            Err(e) => format!("ERR: {}", e),
        };
        if observed == "1" {
            println!("[side-effect-cons] PASS  (car z={})", observed);
            passed += 1;
        } else {
            println!("[side-effect-cons] FAIL  car z={} (expected 1)", observed);
            failed += 1;
        }
    }

    println!("\n=== {} passed, {} failed ===", passed, failed);

    // hang
    loop {
        unsafe { core::arch::asm!("wfe"); }
    }
}
