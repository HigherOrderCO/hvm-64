#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use std::env;
use std::fs;

use hvmc::ast;
use hvmc::fns;
use hvmc::jit;
use hvmc::run;
use hvmc::u60;

use std::collections::HashSet;

struct Args {
  func: String,
  argm: String,
  opts: HashSet<String>,
}

fn get_args() -> Args {
  let args: Vec<String> = env::args().collect();
  let func = args.get(1).unwrap_or(&"help".to_string()).to_string();
  let argm = args.get(2).unwrap_or(&"".to_string()).to_string();
  let opts = args.iter().skip(3).map(|s| s.to_string()).collect::<HashSet<_>>();
  return Args { func, argm, opts };
}

// Runs 'main' without showing the CLI options
fn run_without_cli(args: Args) {
  let lazy    = args.opts.contains("-L");
  let seq     = lazy || args.opts.contains("-1");
  let file    = args.argm;
  let book    = run::Book::new();
  let mut net = run::Net::new(1 << 28, false);
  let begin   = std::time::Instant::now();
  if lazy { todo!() }
  if seq {
    net.normal(&book);
  } else {
    net.parallel_normal(&book);
  }
  println!("{}", net.show());
  print_stats(&net, begin);
}

fn run_with_cli(args: Args) -> Result<(), Box<dyn std::error::Error>> {
  let lazy = args.opts.contains("-L");
  let seq  = lazy || args.opts.contains("-1");
  match args.func.as_str() {
    "run" => {
      if args.argm.len() > 0 {
        let file    = args.argm;
        let book    = load_book(&file);
        let mut net = run::Net::new(1 << 28, lazy);
        let begin   = std::time::Instant::now();
        if seq {
          net.normal(&book);
        } else {
          net.parallel_normal(&book);
        }
        //println!("{}", net.show());
        println!("{}", net.show());
        if args.opts.contains("-s") {
          print_stats(&net, begin);
        }
      } else {
        println!("Usage: hvmc run <file.hvmc> [-s]");
        std::process::exit(1);
      }
    }
    "compile" => {
      if args.argm.len() > 0 {
        let file  = args.argm;
        let book  = load_book(&file);
        let net   = run::Net::new(1 << 28, lazy);
        let begin = std::time::Instant::now();
        compile_book_to_rust_crate(&file, &book)?;
        compile_rust_crate_to_executable(&file)?;
      } else {
        println!("Usage: hvmc compile <file.hvmc>");
        std::process::exit(1);
      }
    }
    "gen-cuda-book" => {
      if args.argm.len() > 0 {
        let file  = args.argm;
        let book  = load_book(&file);
        let net   = run::Net::new(1 << 28, lazy);
        let begin = std::time::Instant::now();
        println!("{}", gen_cuda_book(&book));
      } else {
        println!("Usage: hvmc gen-cuda-book <file.hvmc>");
        std::process::exit(1);
      }
    }
    _ => {
      println!("Usage: hvmc <cmd> <file.hvmc> [-s]");
      println!("Commands:");
      println!("  run           - Run the given file");
      println!("  compile       - Compile the given file to an executable");
      println!("  gen-cuda-book - Generate a CUDA book from the given file");
      println!("Options:");
      println!("  [-s] Show stats, including rewrite count");
      println!("  [-1] Single-core mode (no parallelism)");
    }
  }
  Ok(())
}

#[cfg(not(feature = "hvm_cli_options"))]
fn main() {
  run_without_cli(get_args())
}

#[cfg(feature = "hvm_cli_options")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
  run_with_cli(get_args())
}

fn print_stats(net: &run::Net, begin: std::time::Instant) {
  let rewrites = net.get_rewrites();
  println!("RWTS   : {}", rewrites.total());
  println!("- ANNI : {}", rewrites.anni);
  println!("- COMM : {}", rewrites.comm);
  println!("- ERAS : {}", rewrites.eras);
  println!("- DREF : {}", rewrites.dref);
  println!("- OPER : {}", rewrites.oper);
  println!("TIME   : {:.3} s", (begin.elapsed().as_millis() as f64) / 1000.0);
  println!("RPS    : {:.3} m", (rewrites.total() as f64) / (begin.elapsed().as_millis() as f64) / 1000.0);
}

// Load file
fn load_book(file: &str) -> run::Book {
  let Ok(file) = fs::read_to_string(file) else {
    eprintln!("Input file not found");
    std::process::exit(1);
  };
  return ast::book_to_runtime(&ast::do_parse_book(&file));
}

pub fn compile_book_to_rust_crate(f_name: &str, book: &run::Book) -> Result<(), std::io::Error> {
  let fns_rs = jit::compile_book(book);
  let outdir = ".hvm";
  if std::path::Path::new(&outdir).exists() {
    fs::remove_dir_all(&outdir)?;
  }
  let cargo_toml = include_str!("../Cargo.toml");
  let cargo_toml = cargo_toml.split("##--COMPILER-CUTOFF--##").next().unwrap();
  let cargo_toml = cargo_toml.replace("\"hvm_cli_options\"", "");
  fs::create_dir_all(&format!("{}/src", outdir))?;
  fs::write(".hvm/Cargo.toml", cargo_toml)?;
  fs::write(".hvm/src/ast.rs", include_str!("../src/ast.rs"))?;
  fs::write(".hvm/src/jit.rs", include_str!("../src/jit.rs"))?;
  fs::write(".hvm/src/lib.rs", include_str!("../src/lib.rs"))?;
  fs::write(".hvm/src/main.rs", include_str!("../src/main.rs"))?;
  fs::write(".hvm/src/run.rs", include_str!("../src/run.rs"))?;
  fs::write(".hvm/src/u60.rs", include_str!("../src/u60.rs"))?;
  fs::write(".hvm/src/fns.rs", fns_rs)?;
  return Ok(());
}

pub fn compile_rust_crate_to_executable(f_name: &str) -> Result<(), std::io::Error> {
  let output = std::process::Command::new("cargo").current_dir("./.hvm").arg("build").arg("--release").output()?;
  let target = format!("./{}", f_name.replace(".hvmc", ""));
  if std::path::Path::new(&target).exists() {
    fs::remove_file(&target)?;
  }
  fs::copy("./.hvm/target/release/hvmc", target)?;
  return Ok(());
}

// TODO: move to hvm-cuda repo
pub fn gen_cuda_book(book: &run::Book) -> String {
  use std::collections::BTreeMap;

  // Sort the book.defs by key
  let mut defs = BTreeMap::new();
  for (fid, def) in book.defs.iter() {
    if def.node.len() > 0 {
      defs.insert(fid, def.clone());
    }
  }

  // Initializes code
  let mut code = String::new();

  // Generate function ids
  for (i, id) in defs.keys().enumerate() {
    code.push_str(&format!("const u32 F_{} = 0x{:x};\n", crate::ast::val_to_name(**id), id));
  }
  code.push_str("\n");

  // Create book
  code.push_str("u32 BOOK_DATA[] = {\n");

  // Generate book data
  for (i, (id, net)) in defs.iter().enumerate() {
    let node_len = net.node.len();
    let rdex_len = net.rdex.len();

    code.push_str(&format!("  // @{}\n", crate::ast::val_to_name(**id)));

    // Collect all pointers from root, nodes and rdex into a single buffer
    code.push_str(&format!("  // .nlen\n"));
    code.push_str(&format!("  0x{:08X},\n", node_len));
    code.push_str(&format!("  // .rlen\n"));
    code.push_str(&format!("  0x{:08X},\n", rdex_len));

    // .node
    code.push_str("  // .node\n");
    for (i, node) in net.node.iter().enumerate() {
      code.push_str(&format!("  0x{:08X},", node.1.0));
      code.push_str(&format!(" 0x{:08X},", node.2.0));
      if (i + 1) % 4 == 0 {
        code.push_str("\n");
      }
    }
    if node_len % 4 != 0 {
      code.push_str("\n");
    }

    // .rdex
    code.push_str("  // .rdex\n");
    for (i, (a, b)) in net.rdex.iter().enumerate() {
      code.push_str(&format!("  0x{:08X},", a.0));
      code.push_str(&format!(" 0x{:08X},", b.0));
      if (i + 1) % 4 == 0 {
        code.push_str("\n");
      }
    }
    if rdex_len % 4 != 0 {
      code.push_str("\n");
    }
  }

  code.push_str("};\n\n");

  code.push_str("u32 JUMP_DATA[] = {\n");

  let mut index = 0;
  for (i, fid) in defs.keys().enumerate() {
    code.push_str(&format!("  0x{:08X}, 0x{:08X}, // @{}\n", fid, index, crate::ast::val_to_name(**fid)));
    index += 2 + 2 * defs[fid].node.len() as u32 + 2 * defs[fid].rdex.len() as u32;
  }

  code.push_str("};");

  return code;
}
