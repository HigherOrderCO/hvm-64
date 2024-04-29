use std::{env, process::Command};

fn main() {
  let rustc_version = Command::new(env::var("RUSTC").unwrap()).arg("--version").output().unwrap().stdout;
  let rustc_version = String::from_utf8(rustc_version).unwrap();

  println!("cargo::rustc-env=RUSTC_VERSION={rustc_version}");
}
