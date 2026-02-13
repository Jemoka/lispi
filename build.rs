fn main() {
    println!("cargo:rustc-link-lib=static=staff-uart");
    println!("cargo:rustc-link-search=.");
    println!("cargo:rerun-if-changed=libstaff-uart.a");
}
