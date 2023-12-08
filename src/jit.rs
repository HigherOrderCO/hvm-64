// Despite the file name, this is not actually a JIT (yet).

use crate::{
  ast::{self, Book, Net, Tree},
  ops::Op,
  run::{
    self, ANode, Lab, Ptr,
    Tag::{self, *},
    Trg,
  },
};
use std::{
  collections::{hash_map::Entry, HashMap},
  fmt::{self, format, Write},
};

pub fn compile_book(book: &ast::Book) -> String {
  let mut code = Code::default();

  writeln!(code, "use crate::run::*;\n");

  for (name, def) in book.iter() {
    let name = sanitize_name(name);
    let lab = u16::MAX; // TODO
    writeln!(code, "pub static DEF_{name}: Def = Def {{ lab: {lab}, inner: DefType::Native(call_{name}) }};").unwrap();
  }

  return code.code;
  // writeln!(code, "");

  // for (fid, def) in book.defs.iter() {
  //   if def.node.len() > 0 {
  //     code.push_str(&compile_term(&book, 1, *fid));
  //     (write!(code, "\n"));
  //   }
  // }

  // (write!(code, "}}"));

  // return code;
}

pub fn compile_term(code: &mut Code, book: &ast::Book, name: &str, net: &ast::Net) -> fmt::Result {
  let mut state = State::default();

  state.call_tree(&net.root, format!("rt"));

  for (i, rd) in net.rdex.iter().enumerate() {
    let (rd0, rd1) = state.create_pair(format!("rd_{i}_"));
    state.call_tree(&rd.0, rd0);
    state.call_tree(&rd.1, rd1);
  }

  let name = sanitize_name(name);
  writeln!(code, "fn call_{name}(net: &mut Net, rt: Ptr) {{")?;
  code.indent(|code| {
    code.write_str(&state.code.code);
    code.write_char('\n');
    code.write_str(&state.post.code);
  });
  writeln!(code, "}}")?;
  code.write_char('\n');

  return Ok(());

  #[derive(Default)]
  struct State<'a> {
    code: Code,
    post: Code,
    vars: HashMap<&'a str, String>,
    pair_count: usize,
  }

  impl<'a> State<'a> {
    fn create_pair(&mut self, n: String) -> (String, String) {
      let i = self.pair_count;
      self.pair_count += 1;
      let n0 = format!("{n}0");
      let n1 = format!("{n}1");

      writeln!(self.code, "let mut {n} = (Lcl::Todo(i), Lcl::Todo(i));");
      writeln!(self.code, "let {n0} = Trg::Lcl(&mut {n}.0);");
      writeln!(self.code, "let {n1} = Trg::Lcl(&mut {n}.1);");

      writeln!(self.post, "let (Lcl::Bound({n0}), Lcl::Bound({n1})) = {n} else {{ unreachable!() }};");
      writeln!(self.post, "net.safe_link(n0, n1);");

      (n0, n1)
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
          writeln!(self.code, "let ({x}, {y}) = net.quick_ctr({trg}, {lab});");
          self.call_tree(lft, x);
          self.call_tree(rgt, y);
        }
        Tree::Var { nam } => match self.vars.entry(&nam) {
          Entry::Occupied(e) => {
            writeln!(self.code, "net.safe_link({}, {trg}", e.remove())?;
          }
          Entry::Vacant(e) => {
            e.insert(trg);
          }
        },
        Tree::Op2 { opr, lft, rgt } => {
          if let Tree::Num { val } = &**lft {
            let r = format!("{trg}r");
            writeln!(self.code, "let {r} = net.quick_op2_num({trg}, {opr:?}, {val});");
            self.call_tree(rgt, r);
          } else {
            let b = format!("{trg}b");
            let r = format!("{trg}r");
            writeln!(self.code, "let ({b}, {r}) = net.quick_op2({trg}, {opr:?});");
            self.call_tree(lft, b);
            self.call_tree(rgt, r);
          }
        }
        Tree::Op1 { opr, lft, rgt } => {
          let r = format!("{trg}r");
          writeln!(self.code, "let {r} = net.quick_op1({trg}, {opr:?}, {lft});");
          self.call_tree(rgt, r);
        }
        Tree::Mat { sel, ret } => {
          let (r0, r1) = self.create_pair(format!("{trg}r"));
          let s = format!("{trg}s");
          if let Tree::Ctr { lab: 0, lft: zero, rgt: succ } = &**sel {
            let z = format!("{trg}z");
            if let Tree::Ctr { lab: 0, lft: inp, rgt: succ } = &**succ {
              let i = format!("{trg}i");
              writeln!(self.code, "let ({z}, {i}, {s}) = net.quick_mat_con({trg}, {r0});");
              self.call_tree(zero, z)?;
              self.call_tree(inp, i)?;
              self.call_tree(succ, s)?;
            } else {
              let z = format!("{trg}z");
              writeln!(self.code, "let ({z}, {s}) = net.quick_mat_con({trg}, {r0});");
              self.call_tree(zero, z)?;
              self.call_tree(succ, s)?;
            }
          } else {
            writeln!(self.code, "let {s} = net.quick_mat({trg}, {r0});");
            self.call_tree(sel, s)?;
          }
          self.call_tree(ret, r1);
        }
      }
      Ok(())
    }
  }
}

impl<'a> run::Net<'a> {
  #[inline]
  /// {#lab x y}
  fn quick_ctr<'b>(&mut self, trg: Trg<'b>, lab: Lab) -> (Trg<'b>, Trg<'b>) {
    let ptr = trg.target();
    if ptr.is_ctr(lab) {
      self.rwts.anni += 1;
      let got = trg.take();
      (Trg::Dir(got.p1()), Trg::Dir(got.p2()))
    } else if ptr.tag() == Num || ptr.tag() == Ref && lab >= ptr.lab() {
      (Trg::Ptr(ptr), Trg::Ptr(ptr))
    } else {
      let n = Ptr::new(Ctr, lab, self.heap.alloc());
      self.half_safe_link(trg, n);
      (Trg::Ptr(n.p1()), Trg::Ptr(n.p2()))
    }
  }
  #[inline]
  /// <op #b x>
  fn quick_op2_num<'b>(&mut self, trg: Trg<'b>, op: Op, b: u64) -> Trg<'b> {
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
  #[inline]
  /// <op x y>
  fn quick_op2<'b>(&mut self, trg: Trg<'b>, op: Op) -> (Trg<'b>, Trg<'b>) {
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
  #[inline]
  /// <a op x>
  fn quick_op1<'b>(&mut self, trg: Trg<'b>, op: Op, a: u64) -> Trg<'b> {
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
  #[inline]
  /// ?<(x (y z)) out>
  fn quick_mat_con_con<'b>(&mut self, trg: Trg<'b>, out: Trg<'b>) -> (Trg<'b>, Trg<'b>, Trg<'b>) {
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
  #[inline]
  /// ?<(x y) out>
  fn quick_mat_con<'b>(&mut self, trg: Trg<'b>, out: Trg<'b>) -> (Trg<'b>, Trg<'b>) {
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
  #[inline]
  /// ?<x out>
  fn quick_mat<'b>(&mut self, trg: Trg<'b>, out: Trg<'b>) -> Trg<'b> {
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
  fn make<'b>(&mut self, tag: Tag, lab: Lab, x: Trg<'b>, y: Trg<'b>) -> Trg<'b> {
    let n = Ptr::new(tag, lab, self.heap.alloc());
    self.half_safe_link(x, n.p1());
    self.half_safe_link(y, n.p2());
    Trg::Ptr(n)
  }
}

trait Sentinel {}
// pub fn compile_term(book: &ast::Book, tab: usize, fid: run::Val) -> String {
//   // returns a fresh variable: 'v<NUM>'
//   fn fresh(newx: &mut usize) -> String {
//     *newx += 1;
//     format!("k{}", newx)
//   }

//   fn call_redex(
//     book: &run::Book,
//     tab: usize,
//     newx: &mut usize,
//     vars: &mut HashMap<run::Ptr, String>,
//     def: &run::Def,
//     rdex: (run::Ptr, run::Ptr),
//   ) -> String {
//     let (rf, rx) = adjust_redex(rdex.0, rdex.1);
//     let rf_name = format!("_{}", fresh(newx));
//     let mut code = String::new();
//     (write!(code, "{}let {} : Trg = Trg::Ptr({});\n", ident(tab), rf_name, &atom(rf)));
//     code.push_str(&burn(book, tab, None, newx, vars, def, rx, &Target { nam: rf_name }));
//     return code;
//   }

//   fn call(
//     book: &run::Book,
//     tab: usize,
//     tail: Option<run::Val>,
//     newx: &mut usize,
//     vars: &mut HashMap<run::Ptr, String>,
//     fid: run::Val,
//     trg: &Target,
//   ) -> String {
//     //let newx = &mut 0;
//     //let vars = &mut HashMap::new();

//     let def = &book.get(fid).unwrap();

//     // Tail call
//     // TODO: when I manually edited a file to implement tail call, the single-core performance
//     // increased a lot, but it resulted in a single thread withholding all redexes and, thus,
//     // the program went single-core mode again. I believe a smarter redex sharing structure is
//     // necessary for us to implement tail calls in a way that doesn't sacrify parallelism.
//     //if tail.is_some() && def.rdex.len() > 0 && def.rdex[0].0.is_ref() && def.rdex[0].0.loc() == tail.unwrap() {
//     //println!("tco {}", ast::val_to_name(tail.unwrap() as run::Val));
//     //let mut code = String::new();
//     //for rdex in &def.rdex[1..] {
//     //code.push_str(&call_redex(book, tab, newx, vars, def, *rdex));
//     //}
//     //code.push_str(&burn(book, tab, Some(fid), newx, vars, def, def.node[0].1, &trg));
//     //code.push_str(&call_redex(book, tab, newx, vars, def, def.rdex[0]));
//     //return code;
//     //}

//     // Normal call
//     let mut code = String::new();
//     for rdex in &def.rdex {
//       code.push_str(&call_redex(book, tab, newx, vars, def, *rdex));
//     }
//     code.push_str(&burn(book, tab, Some(fid), newx, vars, def, def.node[0].1, &trg));
//     return code;
//   }

//   fn burn(
//     book: &run::Book,
//     tab: usize,
//     tail: Option<run::Val>,
//     newx: &mut usize,
//     vars: &mut HashMap<run::Ptr, String>,
//     def: &run::Def,
//     ptr: run::Ptr,
//     trg: &Target,
//   ) -> String {
//     //println!("burn {:08x} {}", ptr.0, x);
//     let mut code = String::new();

//     // (<?(ifz ifs) ret> ret) ~ (#X R)
//     // ------------------------------- fast match
//     // if X == 0:
//     //   ifz ~ R
//     //   ifs ~ *
//     // else:
//     //   ifz ~ *
//     //   ifs ~ (#(X-1) R)
//     // When ifs is REF, tail-call optimization is applied.
//     if ptr.tag() == run::LAM {
//       let mat = def.node[ptr.loc() as usize].0;
//       let rty = def.node[ptr.loc() as usize].1;
//       if mat.tag() == run::MAT {
//         let cse = def.node[mat.loc() as usize].0;
//         let rtx = def.node[mat.loc() as usize].1;
//         let got = def.node[rty.loc() as usize];
//         let rtz = if rty.tag() == run::VR1 { got.0 } else { got.1 };
//         if cse.tag() == run::LAM && rtx.is_var() && rtx == rtz {
//           let ifz = def.node[cse.loc() as usize].0;
//           let ifs = def.node[cse.loc() as usize].1;
//           let c_z = Target { nam: fresh(newx) };
//           let c_s = Target { nam: fresh(newx) };
//           let num = Target { nam: format!("{}x", trg.show()) };
//           let res = Target { nam: format!("{}y", trg.show()) };
//           let lam = fresh(newx);
//           let mat = fresh(newx);
//           let cse = fresh(newx);
//           (write!(code, "{}let {} : Trg;\n", ident(tab), &c_z.show()));
//           (write!(code, "{}let {} : Trg;\n", ident(tab), &c_s.show()));
//           (write!(code, "{}// fast match\n", ident(tab)));
//           (write!(
//             code,
//             "{}if {}.tag() == LAM && self.heap.get({}.loc(), P1).is_num() {{\n",
//             ident(tab),
//             trg.get(),
//             trg.get()
//           ));
//           (write!(code, "{}self.rwts.anni += 2;\n", ident(tab + 1)));
//           (write!(code, "{}self.rwts.oper += 1;\n", ident(tab + 1)));
//           (write!(code, "{}let got = {};\n", ident(tab + 1), trg.take()));
//           (write!(code, "{}let {} = Trg::Dir(Ptr::new(VR1, 0, got.loc()));\n", ident(tab + 1), num.show()));
//           (write!(code, "{}let {} = Trg::Dir(Ptr::new(VR2, 0, got.loc()));\n", ident(tab + 1), res.show()));
//           (write!(code, "{}if {}.val() == 0 {{\n", ident(tab + 1), num.get()));
//           (write!(code, "{}{};\n", ident(tab + 2), num.take()));
//           (write!(code, "{}{} = {};\n", ident(tab + 2), &c_z.show(), res.show()));
//           (write!(code, "{}{} = Trg::Ptr({});\n", ident(tab + 2), &c_s.show(), "ERAS"));
//           (write!(code, "{}}} else {{\n", ident(tab + 1)));
//           (write!(
//             code,
//             "{}{};\n",
//             ident(tab + 2),
//             num.swap(&format!("Ptr::big(NUM, {}.val() - 1)", num.get()))
//           ));
//           (write!(code, "{}{} = Trg::Ptr({});\n", ident(tab + 2), &c_z.show(), "ERAS"));
//           (write!(code, "{}{} = {};\n", ident(tab + 2), &c_s.show(), trg.show()));
//           (write!(code, "{}}}\n", ident(tab + 1)));
//           (write!(code, "{}}} else {{\n", ident(tab)));
//           (write!(code, "{}let {} = self.alloc(1);\n", ident(tab + 1), lam));
//           (write!(code, "{}let {} = self.alloc(1);\n", ident(tab + 1), mat));
//           (write!(code, "{}let {} = self.alloc(1);\n", ident(tab + 1), cse));
//           (write!(code, "{}self.heap.set({}, P1, Ptr::new(MAT, 0, {}));\n", ident(tab + 1), lam, mat));
//           (write!(code, "{}self.heap.set({}, P2, Ptr::new(VR2, 0, {}));\n", ident(tab + 1), lam, mat));
//           (write!(code, "{}self.heap.set({}, P1, Ptr::new(LAM, 0, {}));\n", ident(tab + 1), mat, cse));
//           (write!(code, "{}self.heap.set({}, P2, Ptr::new(VR2, 0, {}));\n", ident(tab + 1), mat, lam));
//           (write!(
//             code,
//             "{}self.safe_link(Trg::Ptr(Ptr::new(LAM, 0, {})), {});\n",
//             ident(tab + 1),
//             lam,
//             trg.show()
//           ));
//           (write!(code, "{}{} = Trg::Ptr(Ptr::new(VR1, 0, {}));\n", ident(tab + 1), &c_z.show(), cse));
//           (write!(code, "{}{} = Trg::Ptr(Ptr::new(VR2, 0, {}));\n", ident(tab + 1), &c_s.show(), cse));
//           (write!(code, "{}}}\n", ident(tab)));
//           code.push_str(&burn(book, tab, None, newx, vars, def, ifz, &c_z));
//           code.push_str(&burn(book, tab, tail, newx, vars, def, ifs, &c_s));
//           return code;
//         }
//       }
//     }

//     // #A ~ <+ #B r>
//     // ----------------- fast op
//     // r <~ #(op(+,A,B))
//     if ptr.is_op2() {
//       let val = def.node[ptr.loc() as usize].0;
//       let ret = def.node[ptr.loc() as usize].1;
//       if let Some(val) = got(vars, def, val) {
//         let val = Target { nam: val };
//         let nxt = Target { nam: fresh(newx) };
//         let op2 = fresh(newx);
//         (write!(code, "{}let {} : Trg;\n", ident(tab), &nxt.show()));
//         (write!(code, "{}// fast op\n", ident(tab)));
//         (write!(code, "{}if {}.is_num() && {}.is_num() {{\n", ident(tab), trg.get(), val.get()));
//         (write!(code, "{}self.rwts.oper += 2;\n", ident(tab + 1))); // OP2 + OP1
//         (write!(code, "{}let vx = {};\n", ident(tab + 1), trg.take()));
//         (write!(code, "{}let vy = {};\n", ident(tab + 1), val.take()));
//         (write!(
//           code,
//           "{}{} = Trg::Ptr(Ptr::big(NUM, self.op({},vx.val(),vy.val())));\n",
//           ident(tab + 1),
//           &nxt.show(),
//           ptr.lab()
//         ));
//         (write!(code, "{}}} else {{\n", ident(tab)));
//         (write!(code, "{}let {} = self.alloc(1);\n", ident(tab + 1), op2));
//         (write!(
//           code,
//           "{}self.safe_link(Trg::Ptr(Ptr::new(VR1, 0, {})), {});\n",
//           ident(tab + 1),
//           op2,
//           val.show()
//         ));
//         (write!(
//           code,
//           "{}self.safe_link(Trg::Ptr(Ptr::new(OP2, {}, {})), {});\n",
//           ident(tab + 1),
//           ptr.lab(),
//           op2,
//           trg.show()
//         ));
//         (write!(code, "{}{} = Trg::Ptr(Ptr::new(VR2, 0, {}));\n", ident(tab + 1), &nxt.show(), op2));
//         (write!(code, "{}}}\n", ident(tab)));
//         code.push_str(&burn(book, tab, None, newx, vars, def, ret, &nxt));
//         return code;
//       }
//     }

//     // {p1 p2} <~ #N
//     // ------------- fast copy
//     // p1 <~ #N
//     // p2 <~ #N
//     if ptr.is_dup() {
//       let x1 = Target { nam: format!("{}x", trg.show()) };
//       let x2 = Target { nam: format!("{}y", trg.show()) };
//       let p1 = def.node[ptr.loc() as usize].0;
//       let p2 = def.node[ptr.loc() as usize].1;
//       let lc = fresh(newx);
//       (write!(code, "{}let {} : Trg;\n", ident(tab), &x1.show()));
//       (write!(code, "{}let {} : Trg;\n", ident(tab), &x2.show()));
//       (write!(code, "{}// fast copy\n", ident(tab)));
//       (write!(code, "{}if {}.tag() == NUM {{\n", ident(tab), trg.get()));
//       (write!(code, "{}self.rwts.comm += 1;\n", ident(tab + 1)));
//       (write!(code, "{}let got = {};\n", ident(tab + 1), trg.take()));
//       (write!(code, "{}{} = Trg::Ptr(got);\n", ident(tab + 1), &x1.show()));
//       (write!(code, "{}{} = Trg::Ptr(got);\n", ident(tab + 1), &x2.show()));
//       (write!(code, "{}}} else {{\n", ident(tab)));
//       (write!(code, "{}let {} = self.alloc(1);\n", ident(tab + 1), lc));
//       (write!(code, "{}{} = Trg::Ptr(Ptr::new(VR1, 0, {}));\n", ident(tab + 1), &x1.show(), lc));
//       (write!(code, "{}{} = Trg::Ptr(Ptr::new(VR2, 0, {}));\n", ident(tab + 1), &x2.show(), lc));
//       (write!(
//         code,
//         "{}self.safe_link(Trg::Ptr(Ptr::new({}, {}, {})), {});\n",
//         ident(tab + 1),
//         tag(ptr.tag()),
//         ptr.lab(),
//         lc,
//         trg.show()
//       ));
//       (write!(code, "{}}}\n", ident(tab)));
//       code.push_str(&burn(book, tab, None, newx, vars, def, p2, &x2));
//       code.push_str(&burn(book, tab, None, newx, vars, def, p1, &x1));
//       return code;
//     }

//     // (p1 p2) <~ (x1 x2)
//     // ------------------ fast apply
//     // p1 <~ x1
//     // p2 <~ x2
//     if ptr.is_ctr() && ptr.tag() == run::LAM {
//       let x1 = Target { nam: format!("{}x", trg.show()) };
//       let x2 = Target { nam: format!("{}y", trg.show()) };
//       let p1 = def.node[ptr.loc() as usize].0;
//       let p2 = def.node[ptr.loc() as usize].1;
//       let lc = fresh(newx);
//       (write!(code, "{}let {} : Trg;\n", ident(tab), &x1.show()));
//       (write!(code, "{}let {} : Trg;\n", ident(tab), &x2.show()));
//       (write!(code, "{}// fast apply\n", ident(tab)));
//       (write!(code, "{}if {}.tag() == {} {{\n", ident(tab), trg.get(), tag(ptr.tag())));
//       (write!(code, "{}self.rwts.anni += 1;\n", ident(tab + 1)));
//       (write!(code, "{}let got = {};\n", ident(tab + 1), trg.take()));
//       (write!(code, "{}{} = Trg::Dir(Ptr::new(VR1, 0, got.loc()));\n", ident(tab + 1), &x1.show()));
//       (write!(code, "{}{} = Trg::Dir(Ptr::new(VR2, 0, got.loc()));\n", ident(tab + 1), &x2.show()));
//       (write!(code, "{}}} else {{\n", ident(tab)));
//       (write!(code, "{}let {} = self.alloc(1);\n", ident(tab + 1), lc));
//       (write!(code, "{}{} = Trg::Ptr(Ptr::new(VR1, 0, {}));\n", ident(tab + 1), &x1.show(), lc));
//       (write!(code, "{}{} = Trg::Ptr(Ptr::new(VR2, 0, {}));\n", ident(tab + 1), &x2.show(), lc));
//       (write!(
//         code,
//         "{}self.safe_link(Trg::Ptr(Ptr::new({}, 0, {})), {});\n",
//         ident(tab + 1),
//         tag(ptr.tag()),
//         lc,
//         trg.show()
//       ));
//       (write!(code, "{}}}\n", ident(tab)));
//       code.push_str(&burn(book, tab, None, newx, vars, def, p2, &x2));
//       code.push_str(&burn(book, tab, None, newx, vars, def, p1, &x1));
//       return code;
//     }

//     //// TODO: implement inlining correctly
//     //// NOTE: enabling this makes dec_bits_tree hang; investigate
//     //if ptr.is_ref() && tail.is_some() {
//     //(write!(code, "{}// inline @{}\n", ident(tab), ast::val_to_name(ptr.loc() as run::Val)));
//     //(write!(code, "{}if !{}.is_skp() {{\n", ident(tab), trg.get()));
//     //(write!(code, "{}self.rwts.dref += 1;\n", ident(tab+1)));
//     //code.push_str(&call(book, tab+1, tail, newx, &mut HashMap::new(), ptr.loc(), trg));
//     //(write!(code, "{}}} else {{\n", ident(tab)));
//     //code.push_str(&make(tab+1, newx, vars, def, ptr, &trg.show()));
//     //(write!(code, "{}}}\n", ident(tab)));
//     //return code;
//     //}

//     // ATOM <~ *
//     // --------- fast erase
//     // nothing
//     if ptr.is_num() || ptr.is_era() {
//       (write!(code, "{}// fast erase\n", ident(tab)));
//       (write!(code, "{}if {}.is_skp() {{\n", ident(tab), trg.get()));
//       (write!(code, "{}{};\n", ident(tab + 1), trg.take()));
//       (write!(code, "{}self.rwts.eras += 1;\n", ident(tab + 1)));
//       (write!(code, "{}}} else {{\n", ident(tab)));
//       code.push_str(&make(tab + 1, newx, vars, def, ptr, &trg.show()));
//       (write!(code, "{}}}\n", ident(tab)));
//       return code;
//     }

//     code.push_str(&make(tab, newx, vars, def, ptr, &trg.show()));
//     return code;
//   }

//   fn make(
//     tab: usize,
//     newx: &mut usize,
//     vars: &mut HashMap<run::Ptr, String>,
//     def: &run::Def,
//     ptr: run::Ptr,
//     trg: &String,
//   ) -> String {
//     //println!("make {:08x} {}", ptr.0, x);
//     let mut code = String::new();
//     if ptr.is_nod() {
//       let lc = fresh(newx);
//       let p1 = def.node[ptr.loc() as usize].0;
//       let p2 = def.node[ptr.loc() as usize].1;
//       (write!(code, "{}let {} = self.alloc(1);\n", ident(tab), lc));
//       code.push_str(&make(tab, newx, vars, def, p2, &format!("Trg::Ptr(Ptr::new(VR2, 0, {}))", lc)));
//       code.push_str(&make(tab, newx, vars, def, p1, &format!("Trg::Ptr(Ptr::new(VR1, 0, {}))", lc)));
//       (write!(
//         code,
//         "{}self.safe_link(Trg::Ptr(Ptr::new({}, {}, {})), {});\n",
//         ident(tab),
//         tag(ptr.tag()),
//         ptr.lab(),
//         lc,
//         trg
//       ));
//     } else if ptr.is_var() {
//       match got(vars, def, ptr) {
//         None => {
//           //println!("-var fst");
//           vars.insert(ptr, trg.clone());
//         }
//         Some(got) => {
//           //println!("-var snd");
//           (write!(code, "{}self.safe_link({}, {});\n", ident(tab), trg, got));
//         }
//       }
//     } else {
//       (write!(code, "{}self.safe_link({}, Trg::Ptr({}));\n", ident(tab), trg, atom(ptr)));
//     }
//     return code;
//   }

//   fn got(vars: &HashMap<run::Ptr, String>, def: &run::Def, ptr: run::Ptr) -> Option<String> {
//     if ptr.is_var() {
//       let got = def.node[ptr.loc() as usize];
//       let slf = if ptr.tag() == run::VR1 { got.0 } else { got.1 };
//       return vars.get(&slf).cloned();
//     } else {
//       return None;
//     }
//   }

//   let fun = ast::val_to_name(fid);
//   let def = &book.get(fid).unwrap();

//   let mut code = String::new();
//   (write!(code, "{}pub fn F_{}(&mut self, ptr: Ptr, trg: Trg<'b>) -> bool {{\n", ident(tab), fun));
//   if def.safe {
//     (write!(code, "{}if self.get(trg).is_dup() {{\n", ident(tab + 1)));
//     (write!(code, "{}self.copy(self.swap(trg, NULL), ptr);\n", ident(tab + 2)));
//     (write!(code, "{}return true;\n", ident(tab + 2)));
//     (write!(code, "{}}}\n", ident(tab + 1)));
//   }
//   code.push_str(&call(book, tab + 1, None, &mut 0, &mut HashMap::new(), fid, &Target {
//     nam: "trg".to_string(),
//   }));
//   (write!(code, "{}return true;\n", ident(tab + 1)));
//   (write!(code, "{}}}\n", ident(tab)));

//   return code;
// }

// // TODO: HVM-Lang must always output in this form.
// fn adjust_redex(rf: run::Ptr, rx: run::Ptr) -> (run::Ptr, run::Ptr) {
//   if rf.is_nilary() && !rx.is_nilary() {
//     return (rf, rx);
//   } else if !rf.is_nilary() && rx.is_nilary() {
//     return (rx, rf);
//   } else {
//     println!("Invalid redex. Compiled HVM requires that ALL defs are in the form:");
//     println!("@name = ROOT");
//     println!("  & ATOM ~ TERM");
//     println!("  & ATOM ~ TERM");
//     println!("  & ATOM ~ TERM");
//     println!("  ...");
//     println!("Where ATOM must be either a ref (`@foo`), a num (`#123`), or an era (`*`).");
//     println!("If you used HVM-Lang, please report on https://github.com/HigherOrderCO/hvm-lang.");
//     panic!("Invalid HVMC file.");
//   }
// }

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
      self.write_str(s)?;
    }

    Ok(())
  }
}

fn sanitize_name(name: &str) -> String {
  todo!()
}
