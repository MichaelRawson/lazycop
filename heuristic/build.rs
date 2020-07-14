use std::env;
use std::process::Command;

fn main() {
    let cwd = env::current_dir().expect("no current directory");
    let fmt_cwd = cwd.display();

    println!("cargo:rerun-if-changed=model.pt");
    println!("cargo:rerun-if-changed=weights.py");
    println!("cargo:rerun-if-changed=weights.h");
    println!("cargo:rerun-if-changed=model.cu");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublasLt");
    println!("cargo:rustc-link-lib=static=model");
    println!("cargo:rustc-link-search={}/", fmt_cwd);

    Command::new("make").output().expect("failed to run 'make'");
}
