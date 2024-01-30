#![cfg_attr(feature = "trace", feature(const_type_name))]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use hvmc::{
  jit::compile_book,
  run::Def,
  trace::{Tracer, _read_traces, _reset_traces},
  *,
};

use std::{collections::HashSet, env, fs, sync::atomic, time::Instant};

fn main() {
  if cfg!(feature = "trace") {
    trace::set_hook();
  }
  let s = Instant::now();
  // for i in 0 .. 100000 {
  //   // unsafe { _reset_traces() };
  //   println!("{} {:?}", i, s.elapsed());
  //   cli_main();
  //   // _read_traces(100);
  //   // return;
  // }
  if cfg!(feature = "hvm_cli_options") {
    cli_main()
  } else {
    bare_main()
  }
  if cfg!(feature = "trace") {
    hvmc::trace::_read_traces(usize::MAX);
  }
}

fn bare_main() {
  // let args: Vec<String> = env::args().collect();
  // let opts = args.iter().skip(3).map(|s| s.as_str()).collect::<HashSet<_>>();
  // let book = run::Book::new();
  // let data = run::Heap::init(1 << 28);
  // let mut net = run::Net::new(&data);
  // net.boot(ast::name_to_val("main"));
  // let start_time = std::time::Instant::now();
  // if opts.contains("-1") {
  //   net.normal();
  // } else {
  //   net.parallel_normal();
  // }
  // println!("{}", ast::show_runtime_net(&net));
  // print_stats(&net, start_time);
}

fn cli_main() {
  let data = run::Net::init_heap(1 << 32);
  let args: Vec<String> = env::args().collect();
  let help = "help".to_string();
  let opts = args.iter().skip(3).map(|s| s.as_str()).collect::<HashSet<_>>();
  let action = args.get(1).unwrap_or(&help);
  let f_name = args.get(2);
  match action.as_str() {
    "run" => {
      if let Some(file_name) = f_name {
        let (_, host, mut net) = load(&data, file_name);
        let start_time = std::time::Instant::now();
        if opts.contains("-1") {
          net.normal();
        } else {
          net.parallel_normal();
        }
        let elapsed = start_time.elapsed();
        println!("{}", ast::show_net(&host.readback(&net)));
        if opts.contains("-s") {
          print_stats(&net, elapsed);
        }
      } else {
        println!("Usage: hvmc run <file.hvmc> [-s]");
        std::process::exit(1);
      }
    }
    "compiled" => {
      let host: ast::Host = Default::default();
      let host = hvmc::gen::host();
      let mut net = run::Net::new(&data);
      net.boot(&host.defs["main"]);
      let start_time = std::time::Instant::now();
      if opts.contains("-1") {
        net.normal();
      } else {
        net.parallel_normal();
      }
      let elapsed = start_time.elapsed();
      println!("{}", ast::show_net(&host.readback(&net)));
      if opts.contains("-s") {
        print_stats(&net, elapsed);
      }
    }
    "compile" => {
      if let Some(file_name) = f_name {
        let (book, host, _) = load(&data, file_name);
        println!("{}", compile_book(&host).unwrap());
        // compile_book_to_rust_crate(file_name, &book)?;
        // compile_rust_crate_to_executable(file_name)?;
      } else {
        println!("Usage: hvmc compile <file.hvmc>");
        std::process::exit(1);
      }
    }
    _ => {
      println!("Usage: hvmc <cmd> <file.hvmc> [-s]");
      println!("Commands:");
      println!("  run           - Run the given file");
      println!("  compile       - Compile the given file to an executable");
      println!("Options:");
      println!("  [-s] Show stats, including rewrite count");
      println!("  [-1] Single-core mode (no parallelism)");
    }
  }
}

fn print_stats(net: &run::Net, elapsed: std::time::Duration) {
  println!("RWTS   : {}", net.rwts.total());
  println!("- ANNI : {}", net.rwts.anni);
  println!("- COMM : {}", net.rwts.comm);
  println!("- ERAS : {}", net.rwts.eras);
  println!("- DREF : {}", net.rwts.dref);
  println!("- OPER : {}", net.rwts.oper);
  println!("QUIK   : {}", net.quik.total());
  println!("- ANNI : {}", net.quik.anni);
  println!("- COMM : {}", net.quik.comm);
  println!("- ERAS : {}", net.quik.eras);
  println!("- DREF : {}", net.quik.dref);
  println!("- OPER : {}", net.quik.oper);
  println!("TIME   : {:.3} s", (elapsed.as_millis() as f64) / 1000.0);
  println!("RPS    : {:.3} m", ((net.rwts.total() + net.quik.total()) as f64) / (elapsed.as_millis() as f64) / 1000.0);
}

// Load file and generate net
fn load<'a>(data: &'a run::Area, file: &str) -> (ast::Book, ast::Host, run::Net<'a>) {
  let Ok(file) = fs::read_to_string(file) else {
    eprintln!("Input file not found");
    std::process::exit(1);
  };
  let book = ast::parse_book(&file);
  let host = ast::Host::new(&book);
  let mut net = run::Net::new(data);
  net.boot(&host.defs["main"]);
  (book, host, net)
}

// pub fn compile_book_to_rust_crate(f_name: &str, book: &run::Book) -> Result<(), std::io::Error> {
//   let fns_rs = jit::compile_book(book);
//   let outdir = ".hvm";
//   if std::path::Path::new(&outdir).exists() {
//     fs::remove_dir_all(&outdir)?;
//   }
//   let cargo_toml = include_str!("../Cargo.toml");
//   let cargo_toml = cargo_toml.split("##--COMPILER-CUTOFF--##").next().unwrap();
//   let cargo_toml = cargo_toml.replace("\"hvm_cli_options\"", "");
//   fs::create_dir_all(&format!("{}/src", outdir))?;
//   fs::write(".hvm/Cargo.toml", cargo_toml)?;
//   fs::write(".hvm/src/ast.rs", include_str!("../src/ast.rs"))?;
//   fs::write(".hvm/src/jit.rs", include_str!("../src/jit.rs"))?;
//   fs::write(".hvm/src/lib.rs", include_str!("../src/lib.rs"))?;
//   fs::write(".hvm/src/main.rs", include_str!("../src/main.rs"))?;
//   fs::write(".hvm/src/run.rs", include_str!("../src/run.rs"))?;
//   fs::write(".hvm/src/u60.rs", include_str!("../src/u60.rs"))?;
//   fs::write(".hvm/src/fns.rs", fns_rs)?;
//   return Ok(());
// }

pub fn compile_rust_crate_to_executable(f_name: &str) -> Result<(), std::io::Error> {
  let output = std::process::Command::new("cargo").current_dir("./.hvm").arg("build").arg("--release").output()?;
  let target = format!("./{}", f_name.replace(".hvmc", ""));
  if std::path::Path::new(&target).exists() {
    fs::remove_file(&target)?;
  }
  fs::copy("./.hvm/target/release/hvmc", target)?;
  Ok(())
}
