// Despite the file name, this is not actually a JIT (yet).

use crate::{
  ast::{self, calculate_min_safe_labels, Book, DefRef, Net, Tree},
  ops::Op,
  run::{
    self, ANode, Lab, Ptr,
    Tag::{self, *},
  },
};
use std::{
  collections::{hash_map::Entry, HashMap},
  fmt::{self, format, Write},
  hint::unreachable_unchecked,
};

pub fn compile_book(book: &ast::Book) -> Result<String, fmt::Error> {
  let mut code = Code::default();

  writeln!(code, "use crate::{{ast::{{Host, DefRef}}, run::{{*, Tag::*}}, ops::Op::*, jit::*}};\n")?;

  writeln!(code, "pub fn host() -> Host {{")?;
  code.indent(|code| {
    writeln!(code, "let mut host = Host::default();")?;
    for (raw_name, net) in book.iter() {
      let name = sanitize_name(raw_name);
      writeln!(code, r##"host.defs.insert(r#"{raw_name}"#.to_owned(), DefRef::Static(&DEF_{name}));"##)?;
      writeln!(code, r##"host.back.insert(Ptr::new_ref(&DEF_{name}).loc(), r#"{raw_name}"#.to_owned());"##)?;
    }
    writeln!(code, "host")
  })?;
  writeln!(code, "}}\n")?;

  for (raw_name, lab) in calculate_min_safe_labels(book) {
    let name = sanitize_name(raw_name);
    writeln!(code, "pub static DEF_{name}: Def = Def {{ lab: {lab}, inner: DefType::Native(call_{name}) }};")?;
  }

  writeln!(code, "")?;

  for (raw_name, net) in book.iter() {
    compile_def(&mut code, book, raw_name, net)?;
  }

  Ok(code.code)
}

fn compile_def(code: &mut Code, book: &ast::Book, raw_name: &str, net: &ast::Net) -> fmt::Result {
  let mut state = State::default();

  state.call_tree(&net.root, format!("rt"))?;

  for (i, (a, b)) in net.rdex.iter().enumerate() {
    state.call_redex(a, b, format!("rd{i}"))?;
  }

  let name = sanitize_name(raw_name);
  writeln!(code, "pub fn call_{name}(net: &mut Net, rt: Ptr) {{")?;
  code.indent(|code| {
    code.write_str("let rt = Trg::Ptr(rt);\n")?;
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
    pair_count: usize,
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
    //   writeln!(self.post, "net.safe_link({n0}, {n1});")?;

    //   Ok((n0, n1))
    // }
    fn call_redex(&mut self, a: &'a Tree, b: &'a Tree, name: String) -> fmt::Result {
      let t = match (a, b) {
        (Tree::Era, t) | (t, Tree::Era) => {
          writeln!(self.code, "let {name} = Trg::Ptr(Ptr::ERA);")?;
          t
        }
        (Tree::Ref { nam }, t) | (t, Tree::Ref { nam }) => {
          writeln!(self.code, "let {name} = Trg::Ptr(Ptr::new_ref(&DEF_{nam}));")?;
          t
        }
        (Tree::Num { val }, t) | (t, Tree::Num { val }) => {
          writeln!(self.code, "let {name} = Trg::Ptr(Ptr::new_num({val}));")?;
          t
        }
        _ => panic!("Invalid redex"),
      };
      self.call_tree(t, name)
    }
    fn call_tree(&mut self, tree: &'a Tree, trg: String) -> fmt::Result {
      match tree {
        Tree::Era => {
          writeln!(self.code, "net.half_safe_link({trg}, Ptr::ERA);")?;
        }
        Tree::Ref { nam } => {
          writeln!(self.code, "net.half_safe_link({trg}, Ptr::new_ref(&DEF_{nam}));")?;
        }
        Tree::Num { val } => {
          writeln!(self.code, "net.half_safe_link({trg}, Ptr::new_num({val}));")?;
        }
        Tree::Ctr { lab, lft, rgt } => {
          let x = format!("{trg}x");
          let y = format!("{trg}y");
          writeln!(self.code, "let ({x}, {y}) = net.quick_ctr({trg}, {lab});")?;
          self.call_tree(lft, x)?;
          self.call_tree(rgt, y)?;
        }
        Tree::Var { nam } => match self.vars.entry(&nam) {
          Entry::Occupied(e) => {
            writeln!(self.code, "net.safe_link({}, {trg});", e.remove())?;
          }
          Entry::Vacant(e) => {
            e.insert(trg);
          }
        },
        Tree::Op2 { opr, lft, rgt } => {
          if let Tree::Num { val } = &**lft {
            let r = format!("{trg}r");
            writeln!(self.code, "let {r} = net.quick_op2_num({trg}, {opr:?}, {val});")?;
            self.call_tree(rgt, r)?;
          } else {
            let b = format!("{trg}b");
            let r = format!("{trg}r");
            writeln!(self.code, "let ({b}, {r}) = net.quick_op2({trg}, {opr:?});")?;
            self.call_tree(lft, b)?;
            self.call_tree(rgt, r)?;
          }
        }
        Tree::Op1 { opr, lft, rgt } => {
          let r = format!("{trg}r");
          writeln!(self.code, "let {r} = net.quick_op1({trg}, {opr:?}, {lft});")?;
          self.call_tree(rgt, r)?;
        }
        Tree::Mat { sel, ret } => {
          todo!()
          // let (r0, r1) = self.create_pair(format!("{trg}r"))?;
          // let s = format!("{trg}s");
          // if let Tree::Ctr { lab: 0, lft: zero, rgt: succ } = &**sel {
          //   let z = format!("{trg}z");
          //   if let Tree::Ctr { lab: 0, lft: inp, rgt: succ } = &**succ {
          //     let i = format!("{trg}i");
          //     writeln!(self.code, "let ({z}, {i}, {s}) = net.quick_mat_con({trg}, {r0});")?;
          //     self.call_tree(zero, z)?;
          //     self.call_tree(inp, i)?;
          //     self.call_tree(succ, s)?;
          //   } else {
          //     let z = format!("{trg}z");
          //     writeln!(self.code, "let ({z}, {s}) = net.quick_mat_con({trg}, {r0});")?;
          //     self.call_tree(zero, z)?;
          //     self.call_tree(succ, s)?;
          //   }
          // } else {
          //   writeln!(self.code, "let {s} = net.quick_mat({trg}, {r0});")?;
          //   self.call_tree(sel, s)?;
          // }
          // self.call_tree(ret, r1)?;
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
pub(crate) enum Trg {
  // Lcl(&'t mut Lcl<'l>),
  Dir(Ptr), // we don't own the pointer, so we point to its location
  Ptr(Ptr), // we own the pointer, so we store it directly
}

impl Trg {
  #[inline(always)]
  pub fn target(&self) -> Ptr {
    match self {
      Trg::Dir(dir) => dir.target().load(),
      Trg::Ptr(ptr) => *ptr,
    }
  }

  #[inline(always)]
  pub fn take(self) -> Ptr {
    match self {
      Trg::Dir(dir) => dir.target().swap(Ptr::LOCK),
      Trg::Ptr(ptr) => ptr,
    }
  }
}

impl<'a> run::Net<'a> {
  // Links two targets, using atomics when necessary, based on implied ownership.
  #[inline(always)]
  pub(crate) fn half_safe_link(&mut self, a: Trg, b: Ptr) {
    match a {
      Trg::Dir(a_dir) => self.half_atomic_link(a_dir, b),
      Trg::Ptr(a_ptr) => self.link(a_ptr, b),
      // Trg::Lcl(Lcl::Bound(_)) => unsafe { unreachable_unchecked() },
      // Trg::Lcl(t) => {
      //   *t = Lcl::Bound(Trg::Ptr(b));
      // }
    }
  }

  // Links two targets, using atomics when necessary, based on implied ownership.
  #[inline(always)]
  pub(crate) fn safe_link(&mut self, a: Trg, b: Trg) {
    match (a, b) {
      (Trg::Dir(a_dir), Trg::Dir(b_dir)) => self.atomic_link(a_dir, b_dir),
      (Trg::Dir(a_dir), Trg::Ptr(b_ptr)) => self.half_atomic_link(a_dir, b_ptr),
      (Trg::Ptr(a_ptr), Trg::Dir(b_dir)) => self.half_atomic_link(b_dir, a_ptr),
      (Trg::Ptr(a_ptr), Trg::Ptr(b_ptr)) => self.link(a_ptr, b_ptr),
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
  pub(crate) fn quick_ctr(&mut self, trg: Trg, lab: Lab) -> (Trg, Trg) {
    let ptr = trg.target();
    if ptr.is_ctr(lab) {
      self.rwts.anni += 1;
      let got = trg.take();
      (Trg::Dir(got.p1()), Trg::Dir(got.p2()))
    } else if ptr.tag() == Num || ptr.tag() == Ref && lab >= ptr.lab() {
      (Trg::Ptr(ptr), Trg::Ptr(ptr))
    } else {
      let loc = self.heap.alloc();
      let n = Ptr::new(Ctr, lab, loc);
      self.half_safe_link(trg, n);
      (Trg::Ptr(n.p1()), Trg::Ptr(n.p2()))
    }
  }
  #[inline(always)]
  /// <op #b x>
  pub(crate) fn quick_op2_num(&mut self, trg: Trg, op: Op, b: u64) -> Trg {
    let ptr = trg.target();
    if ptr.tag() == Num {
      self.rwts.oper += 2;
      let got = trg.take();
      Trg::Ptr(Ptr::new_num(op.op(got.num(), b)))
    } else if ptr == Ptr::ERA {
      Trg::Ptr(Ptr::ERA)
    } else {
      let n = Ptr::new(Op2, op as Lab, self.heap.alloc());
      self.half_safe_link(trg, n);
      n.p1().target().store(Ptr::new_num(b));
      Trg::Ptr(n.p2())
    }
  }
  #[inline(always)]
  /// <op x y>
  pub(crate) fn quick_op2(&mut self, trg: Trg, op: Op) -> (Trg, Trg) {
    let ptr = trg.target();
    if ptr.tag() == Num {
      self.rwts.oper += 1;
      let got = trg.take();
      let n = Ptr::new(Op1, op as Lab, self.heap.alloc());
      n.p1().target().store(Ptr::new_num(got.num()));
      (Trg::Ptr(n), Trg::Ptr(n.p2()))
    } else if ptr == Ptr::ERA {
      (Trg::Ptr(Ptr::ERA), Trg::Ptr(Ptr::ERA))
    } else {
      let n = Ptr::new(Op2, op as Lab, self.heap.alloc());
      self.half_safe_link(trg, n);
      (Trg::Ptr(n.p1()), Trg::Ptr(n.p2()))
    }
  }
  #[inline(always)]
  /// <a op x>
  pub(crate) fn quick_op1(&mut self, trg: Trg, op: Op, a: u64) -> Trg {
    let ptr = trg.target();
    if trg.target().tag() == Num {
      self.rwts.oper += 1;
      let got = trg.take();
      Trg::Ptr(Ptr::new_num(op.op(a, got.num())))
    } else if ptr == Ptr::ERA {
      Trg::Ptr(Ptr::ERA)
    } else {
      let n = Ptr::new(Op1, op as Lab, self.heap.alloc());
      self.half_safe_link(trg, n);
      n.p1().target().store(Ptr::new_num(a));
      Trg::Ptr(n.p2())
    }
  }
  #[inline(always)]
  /// ?<(x (y z)) out>
  pub(crate) fn quick_mat_con_con(&mut self, trg: Trg, out: Trg) -> (Trg, Trg, Trg) {
    let ptr = trg.target();
    if trg.target().tag() == Num {
      self.rwts.oper += 1;
      let num = trg.take().num();
      if num == 0 {
        (out, Trg::Ptr(Ptr::ERA), Trg::Ptr(Ptr::ERA))
      } else {
        (Trg::Ptr(Ptr::ERA), Trg::Ptr(Ptr::new_num(num - 1)), out)
      }
    } else if ptr == Ptr::ERA {
      self.half_safe_link(out, Ptr::ERA);
      (Trg::Ptr(Ptr::ERA), Trg::Ptr(Ptr::ERA), Trg::Ptr(Ptr::ERA))
    } else {
      let m = Ptr::new(Mat, 0, self.heap.alloc());
      let c1 = Ptr::new(Ctr, 0, self.heap.alloc());
      let c2 = Ptr::new(Ctr, 0, self.heap.alloc());
      m.p1().target().store(c1);
      c1.p2().target().store(c2);
      self.half_safe_link(out, m.p2());
      (Trg::Ptr(c1.p1()), Trg::Ptr(c2.p1()), Trg::Ptr(c2.p2()))
    }
  }
  #[inline(always)]
  /// ?<(x y) out>
  pub(crate) fn quick_mat_con<'t, 'l>(&mut self, trg: Trg, out: Trg) -> (Trg, Trg) {
    let ptr = trg.target();
    if trg.target().tag() == Num {
      self.rwts.oper += 1;
      let num = trg.take().num();
      if num == 0 {
        (out, Trg::Ptr(Ptr::ERA))
      } else {
        let c2 = Ptr::new(Ctr, 0, self.heap.alloc());
        c2.p1().target().store(Ptr::new_num(num - 1));
        self.half_safe_link(out, c2.p2());
        (Trg::Ptr(Ptr::ERA), Trg::Ptr(c2))
      }
    } else if ptr == Ptr::ERA {
      self.half_safe_link(out, Ptr::ERA);
      (Trg::Ptr(Ptr::ERA), Trg::Ptr(Ptr::ERA))
    } else {
      let m = Ptr::new(Mat, 0, self.heap.alloc());
      let c1 = Ptr::new(Ctr, 0, self.heap.alloc());
      m.p1().target().store(c1);
      self.half_safe_link(out, m.p2());
      (Trg::Ptr(c1.p1()), Trg::Ptr(c1.p2()))
    }
  }
  #[inline(always)]
  /// ?<x out>
  pub(crate) fn quick_mat<'t, 'l>(&mut self, trg: Trg, out: Trg) -> Trg {
    let ptr = trg.target();
    if trg.target().tag() == Num {
      self.rwts.oper += 1;
      let num = trg.take().num();
      let c1 = Ptr::new(Ctr, 0, self.heap.alloc());
      if num == 0 {
        self.half_safe_link(out, c1.p1());
        c1.p2().target().store(Ptr::ERA);
      } else {
        let c2 = Ptr::new(Ctr, 0, self.heap.alloc());
        c1.p1().target().store(Ptr::ERA);
        c1.p2().target().store(c2);
        c2.p1().target().store(Ptr::new_num(num - 1));
        self.half_safe_link(out, c2.p2());
      }
      Trg::Ptr(c1)
    } else if ptr == Ptr::ERA {
      self.half_safe_link(out, Ptr::ERA);
      Trg::Ptr(Ptr::ERA)
    } else {
      let m = Ptr::new(Mat, 0, self.heap.alloc());
      self.half_safe_link(out, m.p2());
      Trg::Ptr(m.p1())
    }
  }
  #[inline(always)]
  pub(crate) fn make(&mut self, tag: Tag, lab: Lab, x: Trg, y: Trg) -> Trg {
    let n = Ptr::new(tag, lab, self.heap.alloc());
    self.half_safe_link(x, n.p1());
    self.half_safe_link(y, n.p2());
    Trg::Ptr(n)
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
