// Despite the file name, this is not actually a JIT (yet).

use crate::{
  ast::{self, Tree},
  ops::Op,
  run::{
    self, Lab, Port,
    Tag::{self, *},
    Wire,
  },
};
use std::{
  collections::{hash_map::Entry, HashMap},
  fmt::{self, Write},
};

pub fn compile_book(book: &ast::Book, host: &ast::Host) -> Result<String, fmt::Error> {
  let mut code = Code::default();

  writeln!(code, "#![allow(non_upper_case_globals)]")?;
  writeln!(code, "#[allow(unused_imports)]")?;
  writeln!(code, "use crate::{{ast::{{Host, DefRef}}, run::{{*, Tag::*}}, ops::Op::*, jit::*}};\n")?;

  writeln!(code, "pub fn host() -> Host {{")?;
  code.indent(|code| {
    writeln!(code, "let mut host = Host::default();")?;
    for raw_name in book.keys() {
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

  for (raw_name, net) in book.iter() {
    compile_def(&mut code, raw_name, net)?;
  }

  Ok(code.code)
}

fn compile_def(code: &mut Code, raw_name: &str, net: &ast::Net) -> fmt::Result {
  let mut state = State::default();

  state.write_tree(&net.root, "rt".to_string())?;

  for (i, (a, b)) in net.rdex.iter().enumerate() {
    state.write_redex(a, b, format!("rd{i}"))?;
  }

  let name = sanitize_name(raw_name);
  writeln!(code, "pub fn call_{name}(net: &mut Net, rt: Port) {{")?;
  code.indent(|code| {
    code.write_str("let rt = Trg::Port(rt);\n")?;
    code.write_str(&state.code.code)
    // code.write_char('\n')?;
    // code.write_str(&state.post.code)
  })?;
  writeln!(code, "}}")?;
  code.write_char('\n')?;

  return Ok(());

  #[derive(Default)]
  struct State<'a> {
    code: Code,
    // post: Code,
    vars: HashMap<&'a str, String>,
  }

  impl<'a> State<'a> {
    // fn create_pair(&mut self, n: String) -> Result<(String, String), fmt::Error> {
    //   let i = self.pair_count;
    //   self.pair_count += 1;
    //   let n0 = format!("{n}0");
    //   let n1 = format!("{n}1");

    //   writeln!(self.code, "let mut {n} = (Lcl::Todo({i}), Lcl::Todo({i}));")?;
    //   writeln!(self.code, "let {n0} = Trg::Lcl(&mut {n}.0);")?;
    //   writeln!(self.code, "let {n1} = Trg::Lcl(&mut {n}.1);")?;

    //   writeln!(self.post, "let (Lcl::Bound({n0}), Lcl::Bound({n1})) = {n} else {{ unreachable!() }};")?;
    //   writeln!(self.post, "net.link_trg({n0}, {n1});")?;

    //   Ok((n0, n1))
    // }
    fn write_redex(&mut self, a: &'a Tree, b: &'a Tree, name: String) -> fmt::Result {
      let t = match (a, b) {
        (Tree::Era, t) | (t, Tree::Era) => {
          writeln!(self.code, "let {name} = Trg::Port(Port::ERA);")?;
          t
        }
        (Tree::Ref { nam }, t) | (t, Tree::Ref { nam }) => {
          writeln!(self.code, "let {name} = Trg::Port(Port::new_ref(&DEF_{nam}));")?;
          t
        }
        (Tree::Num { val }, t) | (t, Tree::Num { val }) => {
          writeln!(self.code, "let {name} = Trg::Port(Port::new_num({val}));")?;
          t
        }
        _ => panic!("Invalid redex"),
      };
      self.write_tree(t, name)
    }
    fn write_tree(&mut self, tree: &'a Tree, trg: String) -> fmt::Result {
      match tree {
        Tree::Era => {
          writeln!(self.code, "net.link_trg_port({trg}, Port::ERA);")?;
        }
        Tree::Ref { nam } => {
          writeln!(self.code, "net.link_trg_port({trg}, Port::new_ref(&DEF_{nam}));")?;
        }
        Tree::Num { val } => {
          writeln!(self.code, "net.link_trg_port({trg}, Port::new_num({val}));")?;
        }
        Tree::Ctr { lab, lft, rgt } => {
          let x = format!("{trg}x");
          let y = format!("{trg}y");
          writeln!(self.code, "let ({x}, {y}) = net.do_ctr({trg}, {lab});")?;
          self.write_tree(lft, x)?;
          self.write_tree(rgt, y)?;
        }
        Tree::Var { nam } => match self.vars.entry(nam) {
          Entry::Occupied(e) => {
            writeln!(self.code, "net.link_trg({}, {trg});", e.remove())?;
          }
          Entry::Vacant(e) => {
            e.insert(trg);
          }
        },
        Tree::Op2 { opr, lft, rgt } => {
          if let Tree::Num { val } = &**lft {
            let r = format!("{trg}r");
            writeln!(self.code, "let {r} = net.do_op2_num({trg}, {opr:?}, {val});")?;
            self.write_tree(rgt, r)?;
          } else {
            let b = format!("{trg}b");
            let r = format!("{trg}r");
            writeln!(self.code, "let ({b}, {r}) = net.do_op2({trg}, {opr:?});")?;
            self.write_tree(lft, b)?;
            self.write_tree(rgt, r)?;
          }
        }
        Tree::Op1 { opr, lft, rgt } => {
          let r = format!("{trg}r");
          writeln!(self.code, "let {r} = net.do_op1({trg}, {opr:?}, {lft});")?;
          self.write_tree(rgt, r)?;
        }
        Tree::Mat { .. } => {
          todo!()
          // let (r0, r1) = self.create_pair(format!("{trg}r"))?;
          // let s = format!("{trg}s");
          // if let Tree::Ctr { lab: 0, lft: zero, rgt: succ } = &**sel {
          //   let z = format!("{trg}z");
          //   if let Tree::Ctr { lab: 0, lft: inp, rgt: succ } = &**succ {
          //     let i = format!("{trg}i");
          //     writeln!(self.code, "let ({z}, {i}, {s}) = net.do_mat_con({trg}, {r0});")?;
          //     self.write_tree(zero, z)?;
          //     self.write_tree(inp, i)?;
          //     self.write_tree(succ, s)?;
          //   } else {
          //     let z = format!("{trg}z");
          //     writeln!(self.code, "let ({z}, {s}) = net.do_mat_con({trg}, {r0});")?;
          //     self.write_tree(zero, z)?;
          //     self.write_tree(succ, s)?;
          //   }
          // } else {
          //   writeln!(self.code, "let {s} = net.do_mat({trg}, {r0});")?;
          //   self.write_tree(sel, s)?;
          // }
          // self.write_tree(ret, r1)?;
        }
      }
      Ok(())
    }
  }
}

// pub(crate) enum Lcl<'a> {
//   Bound(Trg<'a, 'a>),
//   Todo(usize),
// }

// A target pointer, with implied ownership.
#[derive(Clone)]
pub enum Trg {
  // Lcl(&'t mut Lcl<'l>),
  Wire(Wire), // we don't own the pointer, so we point to its location
  Port(Port), // we own the pointer, so we store it directly
}

impl Trg {
  #[inline(always)]
  pub fn target(&self) -> Port {
    match self {
      Trg::Wire(dir) => dir.load_target(),
      Trg::Port(port) => port.clone(),
    }
  }
}

#[derive(Debug, Clone)]
pub enum Instruction {
  Const(Port, usize),
  Link(usize, usize),
  Set(usize, Port),
  Ctr(Lab, usize, usize, usize),
  Op2(Op, usize, usize, usize),
  Op1(Op, u64, usize, usize),
  Mat(usize, usize, usize),
}

impl<'a> run::Net<'a> {
  #[inline(always)]
  pub fn free_trg(&mut self, trg: Trg) {
    match trg {
      Trg::Wire(wire) => self.half_free(wire.loc()),
      Trg::Port(_) => {}
    }
  }
  // Links two targets, using atomics when necessary, based on implied ownership.
  #[inline]
  pub fn link_trg_port(&mut self, a: Trg, b: Port) {
    match a {
      Trg::Wire(a) => self.link_wire_port(a, b),
      Trg::Port(a) => self.link_port_port(a, b),
      // Trg::Lcl(Lcl::Bound(_)) => unsafe { unreachable_unchecked() },
      // Trg::Lcl(t) => {
      //   *t = Lcl::Bound(Trg::Port(b));
      // }
    }
  }

  // Links two targets, using atomics when necessary, based on implied ownership.
  #[inline(always)]
  pub fn link_trg(&mut self, a: Trg, b: Trg) {
    match (a, b) {
      (Trg::Wire(a), Trg::Wire(b)) => self.link_wire_wire(a, b),
      (Trg::Wire(a), Trg::Port(b)) => self.link_wire_port(a, b),
      (Trg::Port(a), Trg::Wire(b)) => self.link_wire_port(b, a),
      (Trg::Port(a), Trg::Port(b)) => self.link_port_port(a, b),
      // (Trg::Lcl(Lcl::Bound(_)), _) | (_, Trg::Lcl(Lcl::Bound(_))) => unsafe { unreachable_unchecked() },
      // (Trg::Lcl(a), Trg::Lcl(b)) => {
      //   let (&Lcl::Todo(an), &Lcl::Todo(bn)) = (&*a, &*b) else { unsafe { unreachable_unchecked() } };
      //   let (a, b) = if an < bn { (a, b) } else { (b, a) };
      //   *b = Lcl::Bound(Trg::Lcl(a));
      // }
      // _ => todo!(), // (Trg::Lcl(t), u) | (u, Trg::Lcl(t)) => *t = Lcl::Bound(u),
    }
  }

  #[inline(always)]
  /// {#lab x y}
  pub fn do_ctr(&mut self, trg: Trg, lab: Lab) -> (Trg, Trg) {
    let port = trg.target();
    if port.is_ctr(lab) {
      self.free_trg(trg);
      let node = port.consume_node();
      self.quik.anni += 1;
      (Trg::Wire(node.p1), Trg::Wire(node.p2))
    // TODO: fast copy?
    } else if port.tag() == Num || port.tag() == Ref && lab >= port.lab() {
      self.quik.comm += 1;
      (Trg::Port(port.clone()), Trg::Port(port))
    } else {
      let n = self.create_node(Ctr, lab);
      self.link_trg_port(trg, n.p0);
      (Trg::Port(n.p1), Trg::Port(n.p2))
    }
  }
  #[inline(always)]
  /// <op #b x>
  pub fn do_op2_num(&mut self, trg: Trg, op: Op, b: u64) -> Trg {
    let port = trg.target();
    if port.tag() == Num {
      self.quik.oper += 2;
      self.free_trg(trg);
      Trg::Port(Port::new_num(op.op(port.num(), b)))
    } else if port == Port::ERA {
      Trg::Port(Port::ERA)
    } else {
      let n = self.create_node(Op2, op as Lab);
      self.link_trg_port(trg, n.p0);
      n.p1.wire().set_target(Port::new_num(b));
      Trg::Port(n.p2)
    }
  }
  #[inline(always)]
  /// <op x y>
  pub fn do_op2(&mut self, trg: Trg, op: Op) -> (Trg, Trg) {
    let port = trg.target();
    if port.tag() == Num {
      self.quik.oper += 1;
      self.free_trg(trg);
      let n = self.create_node(Op1, op as Lab);
      n.p1.wire().set_target(Port::new_num(port.num()));
      (Trg::Port(n.p0), Trg::Port(n.p2))
    } else if port == Port::ERA {
      (Trg::Port(Port::ERA), Trg::Port(Port::ERA))
    } else {
      let n = self.create_node(Op2, op as Lab);
      self.link_trg_port(trg, n.p0);
      (Trg::Port(n.p1), Trg::Port(n.p2))
    }
  }
  #[inline(always)]
  /// <a op x>
  pub fn do_op1(&mut self, trg: Trg, op: Op, a: u64) -> Trg {
    let port = trg.target();
    if trg.target().tag() == Num {
      self.quik.oper += 1;
      self.free_trg(trg);
      Trg::Port(Port::new_num(op.op(a, port.num())))
    } else if port == Port::ERA {
      Trg::Port(Port::ERA)
    } else {
      let n = self.create_node(Op1, op as Lab);
      self.link_trg_port(trg, n.p0);
      n.p1.wire().set_target(Port::new_num(a));
      Trg::Port(n.p2)
    }
  }
  #[inline(always)]
  /// ?<(x (y z)) out>
  pub fn do_mat_con_con(&mut self, trg: Trg, out: Trg) -> (Trg, Trg, Trg) {
    let port = trg.target();
    if trg.target().tag() == Num {
      self.quik.oper += 1;
      self.free_trg(trg);
      let num = port.num();
      if num == 0 {
        (out, Trg::Port(Port::ERA), Trg::Port(Port::ERA))
      } else {
        (Trg::Port(Port::ERA), Trg::Port(Port::new_num(num - 1)), out)
      }
    } else if port == Port::ERA {
      self.link_trg_port(out, Port::ERA);
      (Trg::Port(Port::ERA), Trg::Port(Port::ERA), Trg::Port(Port::ERA))
    } else {
      let m = self.create_node(Mat, 0);
      let c1 = self.create_node(Ctr, 0);
      let c2 = self.create_node(Ctr, 0);
      self.link_port_port(m.p1, c1.p0);
      self.link_port_port(c1.p2, c2.p0);
      self.link_trg_port(out, m.p2);
      (Trg::Port(c1.p1), Trg::Port(c2.p1), Trg::Port(c2.p2))
    }
  }
  #[inline(always)]
  /// ?<(x y) out>
  pub fn do_mat_con<'t, 'l>(&mut self, trg: Trg, out: Trg) -> (Trg, Trg) {
    let port = trg.target();
    if trg.target().tag() == Num {
      self.quik.oper += 1;
      self.free_trg(trg);
      let num = port.num();
      if num == 0 {
        (out, Trg::Port(Port::ERA))
      } else {
        let c2 = self.create_node(Ctr, 0);
        c2.p1.wire().set_target(Port::new_num(num - 1));
        self.link_trg_port(out, c2.p2);
        (Trg::Port(Port::ERA), Trg::Port(c2.p0))
      }
    } else if port == Port::ERA {
      self.link_trg_port(out, Port::ERA);
      (Trg::Port(Port::ERA), Trg::Port(Port::ERA))
    } else {
      let m = self.create_node(Mat, 0);
      let c1 = self.create_node(Ctr, 0);
      self.link_port_port(m.p1, c1.p0);
      self.link_trg_port(out, m.p2);
      self.link_trg_port(trg, m.p0);
      (Trg::Port(c1.p1), Trg::Port(c1.p2))
    }
  }
  #[inline(always)]
  /// ?<x y>
  pub fn do_mat<'t, 'l>(&mut self, trg: Trg) -> (Trg, Trg) {
    let port = trg.target();
    if port.tag() == Num {
      self.quik.oper += 1;
      self.free_trg(trg);
      let num = port.num();
      let c1 = self.create_node(Ctr, 0);
      if num == 0 {
        self.link_port_port(c1.p2, Port::ERA);
        (Trg::Port(c1.p0), Trg::Wire(self.create_wire(c1.p1)))
      } else {
        let c2 = self.create_node(Ctr, 0);
        self.link_port_port(c1.p1, Port::ERA);
        self.link_port_port(c1.p2, c2.p0);
        self.link_port_port(c2.p1, Port::new_num(num - 1));
        (Trg::Port(c1.p0), Trg::Wire(self.create_wire(c2.p2)))
      }
    } else if port == Port::ERA {
      self.quik.eras += 1;
      self.free_trg(trg);
      (Trg::Port(Port::ERA), Trg::Port(Port::ERA))
    } else {
      let m = self.create_node(Mat, 0);
      self.link_trg_port(trg, m.p0);
      (Trg::Port(m.p1), Trg::Port(m.p2))
    }
  }
  #[inline(always)]
  pub fn make(&mut self, tag: Tag, lab: Lab, x: Trg, y: Trg) -> Trg {
    let n = self.create_node(tag, lab);
    self.link_trg_port(x, n.p1);
    self.link_trg_port(y, n.p2);
    Trg::Port(n.p0)
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
  name.to_owned()
}
