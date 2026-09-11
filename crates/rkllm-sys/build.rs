fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RKLLM_LIB_DIR");

    // Without the `link` feature nothing is resolved at build time: the caller
    // loads librkllmrt.so at run time through the `libloading` bindings.
    if std::env::var_os("CARGO_FEATURE_LINK").is_none() {
        return;
    }

    if let Some(dir) = std::env::var_os("RKLLM_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", dir.to_string_lossy());
    }

    println!("cargo:rustc-link-lib=dylib=rkllmrt");
}
