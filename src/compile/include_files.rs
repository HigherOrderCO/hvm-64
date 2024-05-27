use std::{fs, io, path::Path, sync::Arc};

use hvm64_host::Host;

use parking_lot::Mutex;

macro_rules! include_files {
  ($([$($prefix:ident)*])? crate $name:ident {$($sub:tt)*} $($rest:tt)*) => {
    include_files!([$($($prefix)*)?] $name/ { Cargo.toml src/ { $($sub)* } });
    include_files!([$($($prefix)*)?] $($rest)*);
  };

  ($([$($prefix:ident)*])? $mod:ident/ {$($sub:tt)*} $($rest:tt)*) => {
    fs::create_dir_all(concat!(".hvm/", $($(stringify!($prefix), "/",)*)? stringify!($mod)))?;
    include_files!([$($($prefix)* $mod)?] $($sub)*);
    include_files!([$($($prefix)*)?] $($rest)*);
  };

  ($([$($prefix:ident)*])? $mod:ident {$($sub:tt)*} $($rest:tt)*) => {
    include_files!([$($($prefix)*)?] $mod/ {$($sub)*} $($rest)*);
    include_files!([$($($prefix)*)?] $mod $($rest)*);
  };

  ($([$($prefix:ident)*])? $file:ident.$ext:ident $($rest:tt)*) => {
    fs::write(
      concat!(".hvm/", $($(stringify!($prefix), "/",)*)* stringify!($file), ".", stringify!($ext)),
      include_str!(concat!("../../", $($(stringify!($prefix), "/",)*)* stringify!($file), ".", stringify!($ext))),
    )?;
    include_files!([$($($prefix)*)?] $($rest)*);
  };

  ($([$($prefix:ident)*])? $file:ident $($rest:tt)*) => {
    include_files!([$($($prefix)*)?] $file.rs $($rest)*);
  };

  ($([$($prefix:ident)*])?) => {};
}

/// Copies the `hvm-64` source to a temporary `.hvm` directory.
/// Only a subset of `Cargo.toml` is included.
pub fn create_temp_hvm(host: Arc<Mutex<Host>>) -> Result<(), io::Error> {
  let lib = super::compile_host(&host.lock());
  let outdir = ".hvm";
  if Path::new(&outdir).exists() {
    fs::remove_dir_all(outdir)?;
  }

  fs::create_dir_all(".hvm/gen/src/")?;
  fs::write(
    ".hvm/Cargo.toml",
    r#"
[workspace]
resolver = "2"

members = ["util", "runtime", "gen"]

[workspace.lints]
  "#,
  )?;
  fs::write(
    ".hvm/gen/Cargo.toml",
    r#"
[package]
name = "hvm64-gen"
version = "0.0.0"
edition = "2021"

[lib]
crate-type = ["dylib"]

[dependencies]
hvm64-runtime = { path = "../runtime", default-features = false }
"#,
  )?;
  fs::write(".hvm/gen/src/lib.rs", lib)?;

  include_files! {
    prelude
    crate util {
      lib
      array_vec
      bi_enum
      create_var
      deref
      maybe_grow
      ops {
        num
        word
      }
      parse_abbrev_number
      pretty_num
    }
    crate runtime {
      runtime
      addr
      allocator
      def
      dyn_net
      instruction
      interact
      linker
      net
      node
      parallel
      port
      trace
      wire
    }
  }

  Ok(())
}
