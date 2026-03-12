fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rustc-link-arg=-T{}/memory.ld", manifest_dir);
    println!("cargo:rerun-if-changed=memory.ld");
}
