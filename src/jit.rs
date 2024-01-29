// Despite the file name, this is not actually a JIT (yet).

use crate::{
  ast,
  run::{DefType, Instruction, Port, Tag},
};
use std::{
  fmt::{self, Write},
  hash::{DefaultHasher, Hasher},
};

pub fn compile_book(host: &ast::Host) -> Result<String, fmt::Error> {
  let mut code = Code::default();

  writeln!(code, "#![allow(non_upper_case_globals)]")?;
  writeln!(code, "#[allow(unused_imports)]")?;
  writeln!(code, "use crate::{{ast::{{Host, DefRef}}, run::{{*, Tag::*}}, ops::Op::*, jit::*}};\n")?;

  writeln!(code, "pub fn host() -> Host {{")?;
  code.indent(|code| {
    writeln!(code, "let mut host = Host::default();")?;
    for raw_name in host.defs.keys() {
      let name = sanitize_name(raw_name);
      writeln!(code, r##"host.defs.insert(r#"{raw_name}"#.to_owned(), DefRef::Static(&DEF_{name}));"##)?;
      writeln!(code, r##"host.back.insert(Port::new_ref(&DEF_{name}).loc(), r#"{raw_name}"#.to_owned());"##)?;
    }
    writeln!(code, "host")
  })?;
  writeln!(code, "}}\n")?;

  for (raw_name, def) in &host.defs {
    let name = sanitize_name(raw_name);
    let lab = def.lab;
    writeln!(code, "pub static DEF_{name}: Def = Def {{ lab: {lab}, inner: DefType::Native(call_{name}) }};")?;
  }

  writeln!(code)?;

  for (raw_name, def) in &host.defs {
    compile_def(&mut code, host, raw_name, match &def.inner {
      DefType::Net(n) => &n.instr,
      DefType::Native(_) => unreachable!(),
    })?;
  }

  Ok(code.code)
}

fn compile_def(code: &mut Code, host: &ast::Host, raw_name: &str, instr: &[Instruction]) -> fmt::Result {
  let name = sanitize_name(raw_name);
  writeln!(code, "pub fn call_{name}(net: &mut Net, to: Port) {{")?;
  code.indent(|code| {
    code.write_str("let t0 = Trg::Port(to);\n")?;
    for instr in instr {
      match instr {
        Instruction::Const { trg, port } => {
          writeln!(code, "let {trg} = Trg::Port({});", print_port(host, port))
        }
        Instruction::Link { a, b } => writeln!(code, "net.link_trg({a}, {b});"),
        Instruction::Set { trg, port } => {
          writeln!(code, "net.link_trg({trg}, Trg::Port({}));", print_port(host, port))
        }
        Instruction::Ctr { lab, trg, lft, rgt } => writeln!(code, "let ({lft}, {rgt}) = net.do_ctr({lab}, {trg});"),
        Instruction::Op2 { op, trg, lft, rgt } => writeln!(code, "let ({lft}, {rgt}) = net.do_op2({op:?}, {trg});"),
        Instruction::Op1 { op, num, trg, rgt } => writeln!(code, "let {rgt} = net.do_op1({op:?}, {num}, {trg});"),
        Instruction::Mat { trg, lft, rgt } => writeln!(code, "let ({lft}, {rgt}) = net.do_mat({trg});"),
        Instruction::Wires { av, aw, bv, bw } => writeln!(code, "let ({av}, {aw}, {bv}, {bw}) = net.do_wires();"),
      }?;
    }
    Ok(())
  })?;
  writeln!(code, "}}")?;
  code.write_char('\n')?;

  return Ok(());
}

fn print_port(host: &ast::Host, port: &Port) -> String {
  if port == &Port::ERA {
    "Port::ERA".to_owned()
  } else if port.tag() == Tag::Ref {
    let name = sanitize_name(&host.back[&port.loc()]);
    format!("Port::new_ref(&DEF_{name})")
  } else if port.tag() == Tag::Num {
    format!("Port::new_num({})", port.num())
  } else {
    unreachable!()
  }
}

#[derive(Default)]
struct Code {
  code: String,
  indent: usize,
  on_newline: bool,
}

impl Code {
  fn indent<T>(&mut self, cb: impl FnOnce(&mut Code) -> T) -> T {
    self.indent += 1;
    let val = cb(self);
    self.indent -= 1;
    val
  }
}

impl Write for Code {
  fn write_str(&mut self, s: &str) -> fmt::Result {
    for s in s.split_inclusive('\n') {
      if self.on_newline {
        for _ in 0 .. self.indent {
          self.code.write_str("  ")?;
        }
      }

      self.on_newline = s.ends_with('\n');
      self.code.write_str(s)?;
    }

    Ok(())
  }
}

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
