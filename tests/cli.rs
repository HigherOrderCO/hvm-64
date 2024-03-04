//! Test the `hvmc` binary, including its CLI interface.

use std::{
  error::Error,
  io::Read,
  process::{Command, ExitStatus, Stdio},
};

use hvmc::{
  ast::{Net, Tree},
  host::Host,
};
use insta::assert_display_snapshot;

fn get_arithmetic_program_path() -> String {
  return env!("CARGO_MANIFEST_DIR").to_owned() + "/examples/arithmetic.hvmc";
}

fn execute_hvmc(args: &[&str]) -> Result<(ExitStatus, String), Box<dyn Error>> {
  // Spawn the command
  let mut child =
    Command::new(env!("CARGO_BIN_EXE_hvmc")).args(args).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;

  // Capture the output of the command
  let mut stdout = child.stdout.take().ok_or("Couldn't capture stdout!")?;
  let mut stderr = child.stderr.take().ok_or("Couldn't capture stderr!")?;

  // Wait for the command to finish and get the exit status
  let status = child.wait()?;

  // Read the output
  let mut output = String::new();
  stdout.read_to_string(&mut output)?;
  stderr.read_to_string(&mut output)?;

  // Print the output of the command
  Ok((status, output))
}

#[test]
fn test_cli_reduce() {
  // Test normal-form expressions
  assert_display_snapshot!(
    execute_hvmc(&["reduce", "-m", "100M", "--", "#1"]).unwrap().1,
    @"#1"
  );
  // Test non-normal form expressions
  assert_display_snapshot!(
    execute_hvmc(&["reduce", "-m", "100M", "--", "a & #3 ~ <* #4 a>"]).unwrap().1,
    @"#12"
  );
  // Test multiple expressions
  assert_display_snapshot!(
    execute_hvmc(&["reduce", "-m", "100M", "--", "a & #3 ~ <* #4 a>", "a & #64 ~ </ #2 a>"]).unwrap().1,
    @"#12\n#32"
  );

  // Test loading file and reducing expression
  let arithmetic_program = get_arithmetic_program_path();

  assert_display_snapshot!(
    execute_hvmc(&[
      "reduce", "-m", "100M",
      &arithmetic_program,
      "--", "a & @mul ~ (#3 (#4 a))"
    ]).unwrap().1,
    @"#12"
  );

  assert_display_snapshot!(
    execute_hvmc(&[
      "reduce", "-m", "100M",
      &arithmetic_program,
      "--", "a & @mul ~ (#3 (#4 a))", "a & @div ~ (#64 (#2 a))"
    ]).unwrap().1,
    @"#12\n#32"
  )
}

#[test]
fn test_cli_run_with_args() {
  let arithmetic_program = get_arithmetic_program_path();

  // Test simple program running
  assert_display_snapshot!(
    execute_hvmc(&[
      "run", "-m", "100M",
      &arithmetic_program,
    ]).unwrap().1,
    @"({3 </ a b> <% c d>} ({5 a c} [b d]))"
  );

  // Test partial argument passing
  assert_display_snapshot!(
    execute_hvmc(&[
      "run", "-m", "100M",
      &arithmetic_program,
      "#64"
    ]).unwrap().1,
    @"({5 <64/ a> <64% b>} [a b])"
  );

  // Test passing all arguments.
  assert_display_snapshot!(
    execute_hvmc(&[
      "run", "-m", "100M",
      &arithmetic_program,
      "#64",
      "#3"
    ]).unwrap().1,
    @"[#21 #1]"
  );
}

#[test]
fn test_cli_errors() {
  // Test passing all arguments.
  assert_display_snapshot!(
    execute_hvmc(&[
      "run", "this-file-does-not-exist.hvmc"
    ]).unwrap().1,
    @r###"
 Input file "this-file-does-not-exist.hvmc" not found
 "###
  );
  assert_display_snapshot!(
    execute_hvmc(&[
      "reduce", "this-file-does-not-exist.hvmc"
    ]).unwrap().1,
    @r###"
 Input file "this-file-does-not-exist.hvmc" not found
 "###
  );
}

#[test]
fn test_apply_tree() {
  use hvmc::run;
  fn eval_with_args(fun: &str, args: &[&str]) -> Net {
    let area = run::Heap::new_words(1 << 10);

    let mut fun: Net = fun.parse().unwrap();
    for arg in args {
      let arg: Tree = arg.parse().unwrap();
      fun.apply_tree(arg)
    }
    // TODO: When feature/sc-472/argument-passing, use encode_net instead.
    let host = Host::default();

    let mut rnet = run::Net::<run::Strict>::new(&area);
    let root_port = run::Trg::port(run::Port::new_var(rnet.root.addr()));
    host.encode_net(&mut rnet, root_port, &fun);
    rnet.normal();
    let got_result = host.readback(&rnet);
    got_result
  }
  assert_display_snapshot!(
    eval_with_args("(a a)", &vec!["(a a)"]),
    @"(a a)"
  );
  assert_display_snapshot!(
    eval_with_args("b & (a b) ~ a", &vec!["(a a)"]),
    @"a"
  );
  assert_display_snapshot!(
    eval_with_args("(z0 z0)", &vec!["(z1 z1)"]),
    @"(a a)"
  );
  assert_display_snapshot!(
    eval_with_args("(* #1)", &vec!["(a a)"]),
    @"#1"
  );
  assert_display_snapshot!(
    eval_with_args("(<+ a b> (a b))", &vec!["#1", "#2"]),
    @"#3"
  );
  assert_display_snapshot!(
    eval_with_args("(<* a b> (a b))", &vec!["#2", "#3"]),
    @"#6"
  );
  assert_display_snapshot!(
    eval_with_args("(<* a b> (a b))", &vec!["#2"]),
    @"(<2* a> a)"
  );
}
