use crate::{
  host::{DefRef, Host},
  run::{Instruction, Port, Tag},
};
use std::{
  fmt::{self, Write},
  hash::{DefaultHasher, Hasher},
};

/// Compiles a `Host` to Rust, returning a file to replace `gen.rs`.
pub fn compile_host(host: &Host) -> String {
  _compile_host(host).unwrap()
}

fn _compile_host(host: &Host) -> Result<String, fmt::Error> {
  let mut code = String::default();

  writeln!(code, "#![allow(non_upper_case_globals, unused_imports)]")?;
  writeln!(code, "use crate::{{host::{{Host, DefRef}}, run::*, ops::Op::*}};")?;
  writeln!(code, "")?;

  writeln!(code, "pub fn host() -> Host {{")?;
  writeln!(code, "  let mut host = Host::default();")?;
  for raw_name in host.defs.keys() {
    let name = sanitize_name(raw_name);
    writeln!(code, r##"  host.insert_def(r#"{raw_name}"#, DefRef::Static(&DEF_{name}));"##)?;
  }
  writeln!(code, "  host")?;
  writeln!(code, "}}\n")?;

  for (raw_name, def) in &host.defs {
    let name = sanitize_name(raw_name);
    write!(code, "pub const DEF_{name}: &Def = const {{ &Def::new(LabSet::from_bits(&[")?;
    for (i, word) in def.labs.bits.iter().enumerate() {
      if i != 0 {
        write!(code, ", ")?;
      }
      write!(code, "0x{:x}", word)?;
    }
    writeln!(code, "]), call_{name}) }}.upcast();")?;
  }

  writeln!(code)?;

  for (raw_name, def) in &host.defs {
    compile_def(&mut code, host, raw_name, match &def {
      DefRef::Owned(n) => &n.data.instr,
      _ => unreachable!(),
    })?;
  }

  Ok(code)
}

fn compile_def(code: &mut String, host: &Host, raw_name: &str, instr: &[Instruction]) -> fmt::Result {
  let name = sanitize_name(raw_name);
  writeln!(code, "pub fn call_{name}(net: &mut Net, to: Port) {{")?;
  writeln!(code, "  let t0 = Trg::port(to);")?;
  for instr in instr {
    write!(code, "  ")?;
    match instr {
      Instruction::Const { trg, port } => {
        writeln!(code, "let {trg} = Trg::port({});", print_port(host, port))
      }
      Instruction::Link { a, b } => {
        writeln!(code, "net.link_trg({a}, {b});")
      }
      Instruction::LinkConst { trg, port } => {
        writeln!(code, "net.link_trg({trg}, Trg::port({}));", print_port(host, port))
      }
      Instruction::Ctr { lab, trg, lft, rgt } => {
        writeln!(code, "let ({lft}, {rgt}) = net.do_ctr({lab}, {trg});")
      }
      Instruction::Op2 { op, trg, lft, rgt } => {
        writeln!(code, "let ({lft}, {rgt}) = net.do_op2({op:?}, {trg});")
      }
      Instruction::Op1 { op, num, trg, rgt } => {
        writeln!(code, "let {rgt} = net.do_op1({op:?}, {num}, {trg});")
      }
      Instruction::Mat { trg, lft, rgt } => {
        writeln!(code, "let ({lft}, {rgt}) = net.do_mat({trg});")
      }
      Instruction::Wires { av, aw, bv, bw } => {
        writeln!(code, "let ({av}, {aw}, {bv}, {bw}) = net.do_wires();")
      }
    }?;
  }
  writeln!(code, "}}")?;
  code.write_char('\n')?;

  Ok(())
}

fn print_port(host: &Host, port: &Port) -> String {
  if port == &Port::ERA {
    "Port::ERA".to_owned()
  } else if port.tag() == Tag::Ref {
    let name = sanitize_name(&host.back[&port.addr()]);
    format!("Port::new_ref(&DEF_{name})")
  } else if port.tag() == Tag::Num {
    format!("Port::new_num({})", port.num())
  } else {
    unreachable!()
  }
}

/// Adapts `name` to be a valid suffix for a rust identifier, if necessary.
fn sanitize_name(name: &str) -> String {
  if !name.contains('.') {
    name.to_owned()
  } else {
    let mut hasher = DefaultHasher::new();
    hasher.write(name.as_bytes());
    let hash = hasher.finish();
    let mut sanitized = name.replace('.', "_");
    sanitized.push_str("__");
    write!(sanitized, "__{:016x}", hash).unwrap();
    sanitized
  }
}
