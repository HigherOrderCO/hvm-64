#![cfg(feature = "std")]

use alloc::collections::{BTreeMap, BTreeSet};

use crate::prelude::*;

use crate::{
  host::Host,
  run::{Def, Instruction, InterpretedDef, LabSet, Port, Tag},
  stdlib::HostedDef,
};
use core::{fmt::Write, hash::Hasher};
use std::hash::DefaultHasher;

struct DefInfo<'a> {
  rust_name: String,
  def: &'a Def<HostedDef<InterpretedDef>>,
  refs: BTreeSet<&'a str>,
}

/// Compiles a [`Host`] to Rust, returning a file to replace `gen.rs`.
pub fn compile_host(host: &Host) -> String {
  _compile_host(host).unwrap()
}

fn _compile_host(host: &Host) -> Result<String, fmt::Error> {
  let mut code = String::default();

  let mut def_infos: BTreeMap<&str, DefInfo<'_>> = BTreeMap::new();
  for (hvmc_name, def) in &host.defs {
    if let Some(def) = def.downcast_ref::<HostedDef<InterpretedDef>>() {
      def_infos.insert(hvmc_name, DefInfo {
        rust_name: sanitize_name(hvmc_name),
        refs: refs(host, &def.data.0.instr),
        def,
      });
    }
  }

  writeln!(code, "#![allow(warnings)]")?;
  writeln!(
    code,
    "use crate::{{host::Host, stdlib::{{AsHostedDef, HostedDef}}, run::*, ops::{{TypedOp, Ty::*, Op::*}}}};"
  )?;
  writeln!(code)?;

  writeln!(code, "pub fn insert_into_host(host: &mut Host) {{")?;

  // insert empty defs
  for (hvmc_name, DefInfo { rust_name, def, .. }) in &def_infos {
    let labs = compile_lab_set(&def.labs)?;
    writeln!(
      code,
      r##"  host.insert_def(r#"{hvmc_name}"#, unsafe {{ HostedDef::<Def_{rust_name}>::new({labs}) }});"##,
    )?;
  }
  writeln!(code)?;

  // hoist all unique refs present in the right hand side of some def
  for hvmc_name in def_infos.values().flat_map(|info| &info.refs).collect::<BTreeSet<_>>() {
    let rust_name = &def_infos[hvmc_name].rust_name;

    writeln!(code, r##"  let def_{rust_name} = Port::new_ref(&host.defs[r#"{hvmc_name}"#]);"##)?;
  }
  writeln!(code)?;

  // initialize defs that have refs
  for (hvmc_name, DefInfo { rust_name, refs, .. }) in &def_infos {
    if refs.is_empty() {
      continue;
    }

    let fields = refs
      .iter()
      .map(|r| format!("def_{rust_name}: def_{rust_name}.clone()", rust_name = sanitize_name(r)))
      .collect::<Vec<_>>()
      .join(", ");

    writeln!(
      code,
      r##"  host.get_mut::<HostedDef<Def_{rust_name}>>(r#"{hvmc_name}"#).data.0 = Def_{rust_name} {{ {fields} }};"##
    )?;
  }

  writeln!(code, "}}")?;
  writeln!(code)?;

  for DefInfo { rust_name, def, .. } in def_infos.values() {
    compile_struct(&mut code, host, rust_name, def)?;
  }

  Ok(code)
}

fn refs<'a>(host: &'a Host, instructions: &'a [Instruction]) -> BTreeSet<&'a str> {
  let mut refs = BTreeSet::new();

  for instr in instructions {
    if let Instruction::Const { port, .. } | Instruction::LinkConst { port, .. } = instr {
      if port.tag() == Tag::Ref && !port.is_era() {
        refs.insert(host.back[&port.addr()].as_str());
      }
    }
  }

  refs
}

fn compile_struct(
  code: &mut String,
  host: &Host,
  rust_name: &str,
  def: &Def<HostedDef<InterpretedDef>>,
) -> fmt::Result {
  let refs = refs(host, &def.data.0.instr)
    .iter()
    .map(|r| format!("def_{}: Port", sanitize_name(r)))
    .collect::<Vec<_>>()
    .join(",\n  ");

  writeln!(code, "#[derive(Default)]")?;
  writeln!(code, "struct Def_{rust_name} {{")?;
  writeln!(code, "  {refs}")?;
  writeln!(code, "}}")?;
  writeln!(code)?;

  writeln!(code, "impl AsHostedDef for Def_{rust_name} {{")?;
  writeln!(code, "  fn call<M: Mode>(slf: &Def<Self>, net: &mut Net<M>, port: Port) {{")?;
  writeln!(code, "    let t0 = Trg::port(port);")?;

  for instr in &def.data.0.instr {
    write!(code, "    ")?;
    match instr {
      Instruction::Const { trg, port } => {
        writeln!(code, "let {trg} = Trg::port({});", compile_port(host, port))
      }
      Instruction::Link { a, b } => {
        writeln!(code, "net.link_trg({a}, {b});")
      }
      Instruction::LinkConst { trg, port } => {
        writeln!(code, "net.link_trg({trg}, Trg::port({}));", compile_port(host, port))
      }
      Instruction::Ctr { lab, trg, lft, rgt } => {
        writeln!(code, "let ({lft}, {rgt}) = net.do_ctr({lab}, {trg});")
      }
      Instruction::Op { op, trg, rhs, out } => {
        writeln!(code, "let ({rhs}, {out}) = net.do_op({op:?}, {trg});")
      }
      Instruction::OpNum { op, trg, rhs, out } => {
        writeln!(code, "let {out} = net.do_op_num({op:?}, {trg}, {});", compile_port(host, rhs))
      }
      Instruction::Mat { trg, lft, rgt } => {
        writeln!(code, "let ({lft}, {rgt}) = net.do_mat({trg});")
      }
      Instruction::Wires { av, aw, bv, bw } => {
        writeln!(code, "let ({av}, {aw}, {bv}, {bw}) = net.do_wires();")
      }
    }?;
  }
  writeln!(code, "  }}")?;
  writeln!(code, "}}")?;

  Ok(())
}

fn compile_port(host: &Host, port: &Port) -> String {
  if port == &Port::ERA {
    "Port::ERA".to_owned()
  } else if port.tag() == Tag::Ref {
    let name = sanitize_name(&host.back[&port.addr()]);
    format!("slf.data.def_{name}.clone()")
  } else if port.tag() == Tag::Int {
    format!("Port::new_int({})", port.int())
  } else if port.tag() == Tag::F32 {
    let float = port.float();

    if float.is_nan() {
      "Port::new_float(f32::NAN)".to_string()
    } else if float.is_infinite() && float > 0.0 {
      "Port::new_float(f32::INFINITY)".to_string()
    } else if float.is_infinite() {
      "Port::new_float(f32::NEG_INFINITY)".to_string()
    } else {
      format!("Port::new_float({float:?})")
    }
  } else {
    unreachable!()
  }
}

/// Adapts `name` to be a valid suffix for a rust identifier, if necessary.
fn sanitize_name(name: &str) -> String {
  if !name.contains('.') && !name.contains('$') {
    name.to_owned()
  } else {
    // Append a hash to the name to avoid clashes between `foo.bar` and `foo_bar`.
    let mut hasher = DefaultHasher::new();
    hasher.write(name.as_bytes());
    let hash = hasher.finish();
    let mut sanitized = name.replace(|c| c == '.' || c == '$', "_");
    sanitized.push_str("__");
    write!(sanitized, "__{:016x}", hash).unwrap();
    sanitized
  }
}

fn compile_lab_set(labs: &LabSet) -> Result<String, fmt::Error> {
  if labs == &LabSet::ALL {
    return Ok("LabSet::ALL".to_owned());
  }
  if labs == &LabSet::NONE {
    return Ok("LabSet::NONE".to_owned());
  }
  let mut str = "LabSet::from_bits(&[".to_owned();
  for (i, word) in labs.bits.iter().enumerate() {
    if i != 0 {
      write!(str, ", ")?;
    }
    write!(str, "0x{:x}", word)?;
  }
  str.push_str("])");
  Ok(str)
}
