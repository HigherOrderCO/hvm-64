// Despite the file name, this is not actually a JIT (yet).

use crate::run;
use crate::ast;

use std::collections::HashMap;

pub fn compile_book(book: &run::Book) -> String {
  let mut code = String::new();

  code.push_str(&format!("use crate::run::{{*}};\n"));
  code.push_str(&format!("\n"));

  for (fid, def) in book.defs.iter() {
    if def.node.len() > 0 {
      let name = &ast::val_to_name(*fid as run::Val);
      code.push_str(&format!("pub const F_{:4} : Val = 0x{:06x};\n", name, fid));
    }
  }

  code.push_str(&format!("\n"));

  code.push_str(&format!("impl<'a, const LAZY: bool> NetFields<'a, LAZY> where [(); LAZY as usize]: {{\n"));
  code.push_str(&format!("\n"));

  code.push_str(&format!("{}pub fn call_native(&mut self, book: &Book, ptr: Ptr, x: Ptr) -> bool {{\n", ident(1)));
  code.push_str(&format!("{}match ptr.val() {{\n", ident(2)));
  for (fid, def) in book.defs.iter() {
    if def.node.len() > 0 {
      let fun = ast::val_to_name(*fid);
      code.push_str(&format!("{}F_{} => {{ return self.F_{}(ptr, Trg::Ptr(x)); }}\n", ident(3), fun, fun));
    }
  }
  code.push_str(&format!("{}_ => {{ return false; }}\n", ident(3)));
  code.push_str(&format!("{}}}\n", ident(2)));
  code.push_str(&format!("{}}}\n", ident(1)));
  code.push_str(&format!("\n"));

  for (fid, def) in book.defs.iter() {
    if def.node.len() > 0 {
      code.push_str(&compile_term(&book, 1, *fid));
      code.push_str(&format!("\n"));
    }
  }

  code.push_str(&format!("}}"));

  return code;

}

pub fn ident(tab: usize) -> String {
  return "  ".repeat(tab);
}

pub fn tag(tag: run::Tag) -> &'static str {
  match tag {
    run::VR1 => "VR1",
    run::VR2 => "VR2",
    run::RD1 => "RD1",
    run::RD2 => "RD2",
    run::REF => "REF",
    run::ERA => "ERA",
    run::NUM => "NUM",
    run::OP2 => "OP2",
    run::OP1 => "OP1",
    run::MAT => "MAT",
    run::LAM => "LAM",
    run::TUP => "TUP",
    run::DUP => "DUP",
    _ => unreachable!(),
  }
}

pub fn atom(ptr: run::Ptr) -> String {
  if ptr.is_ref() {
    return format!("Ptr::big(REF, F_{})", ast::val_to_name(ptr.val()));
  } else {
    return format!("Ptr::new({}, 0x{:x}, 0x{:x})", tag(ptr.tag()), ptr.lab(), ptr.loc());
  }
}

struct Target {
  nam: String
}

impl Target {
  fn show(&self) -> String {
    format!("{}", self.nam)
  }

  fn get(&self) -> String {
    format!("self.get({})", self.nam)
  }

  fn swap(&self, value: &str) -> String {
    format!("self.swap({}, {})", self.nam, value)
  }

  fn take(&self) -> String {
    self.swap(&"NULL")
  }
}

pub fn compile_term(book: &run::Book, tab: usize, fid: run::Val) -> String {

  // returns a fresh variable: 'v<NUM>'
  fn fresh(newx: &mut usize) -> String {
    *newx += 1;
    format!("k{}", newx)
  }

  fn call_redex(
    book : &run::Book,
    tab  : usize,
    newx : &mut usize,
    vars : &mut HashMap<run::Ptr, String>,
    def  : &run::Def,
    rdex : (run::Ptr, run::Ptr),
  ) -> String {
    let (rf, rx) = adjust_redex(rdex.0, rdex.1);
    let rf_name  = format!("_{}", fresh(newx));
    let mut code = String::new();
    code.push_str(&format!("{}let {} : Trg = Trg::Ptr({});\n", ident(tab), rf_name, &atom(rf)));
    code.push_str(&burn(book, tab, None, newx, vars, def, rx, &Target { nam: rf_name }));
    return code;
  }

  fn call(
    book : &run::Book,
    tab  : usize,
    tail : Option<run::Val>,
    newx : &mut usize,
    vars : &mut HashMap<run::Ptr, String>,
    fid  : run::Val,
    trg  : &Target,
  ) -> String {
    //let newx = &mut 0;
    //let vars = &mut HashMap::new();

    let def = &book.get(fid).unwrap();

    // Tail call
    // TODO: when I manually edited a file to implement tail call, the single-core performance
    // increased a lot, but it resulted in a single thread withholding all redexes and, thus, 
    // the program went single-core mode again. I believe a smarter redex sharing structure is
    // necessary for us to implement tail calls in a way that doesn't sacrify parallelism.
    //if tail.is_some() && def.rdex.len() > 0 && def.rdex[0].0.is_ref() && def.rdex[0].0.loc() == tail.unwrap() {
      //println!("tco {}", ast::val_to_name(tail.unwrap() as run::Val));
      //let mut code = String::new();
      //for rdex in &def.rdex[1..] {
        //code.push_str(&call_redex(book, tab, newx, vars, def, *rdex));
      //}
      //code.push_str(&burn(book, tab, Some(fid), newx, vars, def, def.node[0].1, &trg));
      //code.push_str(&call_redex(book, tab, newx, vars, def, def.rdex[0]));
      //return code;
    //}

    // Normal call
    let mut code = String::new();
    for rdex in &def.rdex {
      code.push_str(&call_redex(book, tab, newx, vars, def, *rdex));
    }
    code.push_str(&burn(book, tab, Some(fid), newx, vars, def, def.node[0].2, &trg));
    return code;
  }
  
  fn burn(
    book : &run::Book,
    tab  : usize,
    tail : Option<run::Val>,
    newx : &mut usize,
    vars : &mut HashMap<run::Ptr, String>,
    def  : &run::Def,
    ptr  : run::Ptr,
    trg  : &Target,
  ) -> String {
    //println!("burn {:08x} {}", ptr.0, x);
    let mut code = String::new();

    // (<?(ifz ifs) ret> ret) ~ (#X R)
    // ------------------------------- fast match
    // if X == 0:
    //   ifz ~ R
    //   ifs ~ *
    // else:
    //   ifz ~ *
    //   ifs ~ (#(X-1) R)
    // When ifs is REF, tail-call optimization is applied.
    if ptr.tag() == run::LAM {
      let mat = def.node[ptr.loc() as usize].1;
      let rty = def.node[ptr.loc() as usize].2;
      if mat.tag() == run::MAT {
        let cse = def.node[mat.loc() as usize].1;
        let rtx = def.node[mat.loc() as usize].2;
        let got = def.node[rty.loc() as usize];
        let rtz = if rty.tag() == run::VR1 { got.1 } else { got.2 };
        if cse.tag() == run::LAM && rtx.is_var() && rtx == rtz {
          let ifz = def.node[cse.loc() as usize].1;
          let ifs = def.node[cse.loc() as usize].2;
          let c_z = Target { nam: fresh(newx) };
          let c_s = Target { nam: fresh(newx) };
          let num = Target { nam: format!("{}x", trg.show()) };
          let res = Target { nam: format!("{}y", trg.show()) };
          let lam = fresh(newx);
          let mat = fresh(newx);
          let cse = fresh(newx);
          code.push_str(&format!("{}let {} : Trg;\n", ident(tab), &c_z.show()));
          code.push_str(&format!("{}let {} : Trg;\n", ident(tab), &c_s.show()));
          code.push_str(&format!("{}// fast match\n", ident(tab)));
          code.push_str(&format!("{}if {}.tag() == LAM && self.heap.get({}.loc(), P1).is_num() {{\n", ident(tab), trg.get(), trg.get()));
          code.push_str(&format!("{}self.rwts.anni += 2;\n", ident(tab+1)));
          code.push_str(&format!("{}self.rwts.oper += 1;\n", ident(tab+1)));
          code.push_str(&format!("{}let got = {};\n", ident(tab+1), trg.take()));
          code.push_str(&format!("{}let {} = Trg::Dir(Ptr::new(VR1, 0, got.loc()));\n", ident(tab+1), num.show()));
          code.push_str(&format!("{}let {} = Trg::Dir(Ptr::new(VR2, 0, got.loc()));\n", ident(tab+1), res.show()));
          code.push_str(&format!("{}if {}.val() == 0 {{\n", ident(tab+1), num.get()));
          code.push_str(&format!("{}{};\n", ident(tab+2), num.take()));
          code.push_str(&format!("{}{} = {};\n", ident(tab+2), &c_z.show(), res.show()));
          code.push_str(&format!("{}{} = Trg::Ptr({});\n", ident(tab+2), &c_s.show(), "ERAS"));
          code.push_str(&format!("{}}} else {{\n", ident(tab+1)));
          code.push_str(&format!("{}{};\n", ident(tab+2), num.swap(&format!("Ptr::big(NUM, {}.val() - 1)", num.get()))));
          code.push_str(&format!("{}{} = Trg::Ptr({});\n", ident(tab+2), &c_z.show(), "ERAS"));
          code.push_str(&format!("{}{} = {};\n", ident(tab+2), &c_s.show(), trg.show()));
          code.push_str(&format!("{}}}\n", ident(tab+1)));
          code.push_str(&format!("{}}} else {{\n", ident(tab)));
          code.push_str(&format!("{}let {} = self.alloc();\n", ident(tab+1), lam));
          code.push_str(&format!("{}let {} = self.alloc();\n", ident(tab+1), mat));
          code.push_str(&format!("{}let {} = self.alloc();\n", ident(tab+1), cse));
          code.push_str(&format!("{}self.heap.set({}, P1, Ptr::new(MAT, 0, {}));\n", ident(tab+1), lam, mat));
          code.push_str(&format!("{}self.heap.set({}, P2, Ptr::new(VR2, 0, {}));\n", ident(tab+1), lam, mat));
          code.push_str(&format!("{}self.heap.set({}, P1, Ptr::new(LAM, 0, {}));\n", ident(tab+1), mat, cse));
          code.push_str(&format!("{}self.heap.set({}, P2, Ptr::new(VR2, 0, {}));\n", ident(tab+1), mat, lam));
          code.push_str(&format!("{}self.safe_link(Trg::Ptr(Ptr::new(LAM, 0, {})), {});\n", ident(tab+1), lam, trg.show()));
          code.push_str(&format!("{}{} = Trg::Ptr(Ptr::new(VR1, 0, {}));\n", ident(tab+1), &c_z.show(), cse));
          code.push_str(&format!("{}{} = Trg::Ptr(Ptr::new(VR2, 0, {}));\n", ident(tab+1), &c_s.show(), cse));
          code.push_str(&format!("{}}}\n", ident(tab)));
          code.push_str(&burn(book, tab, None, newx, vars, def, ifz, &c_z));
          code.push_str(&burn(book, tab, tail, newx, vars, def, ifs, &c_s));
          return code;
        }
      }
    }

    // #A ~ <+ #B r>
    // ----------------- fast op
    // r <~ #(op(+,A,B))
    if ptr.is_op2() {
      let val = def.node[ptr.loc() as usize].1;
      let ret = def.node[ptr.loc() as usize].2;
      if let Some(val) = got(vars, def, val) {
        let val = Target { nam: val };
        let nxt = Target { nam: fresh(newx) };
        let op2 = fresh(newx);
        code.push_str(&format!("{}let {} : Trg;\n", ident(tab), &nxt.show()));
        code.push_str(&format!("{}// fast op\n", ident(tab)));
        code.push_str(&format!("{}if {}.is_num() && {}.is_num() {{\n", ident(tab), trg.get(), val.get()));
        code.push_str(&format!("{}self.rwts.oper += 2;\n", ident(tab+1))); // OP2 + OP1
        code.push_str(&format!("{}let vx = {};\n", ident(tab+1), trg.take()));
        code.push_str(&format!("{}let vy = {};\n", ident(tab+1), val.take()));
        code.push_str(&format!("{}{} = Trg::Ptr(Ptr::big(NUM, self.op({},vx.val(),vy.val())));\n", ident(tab+1), &nxt.show(), ptr.lab()));
        code.push_str(&format!("{}}} else {{\n", ident(tab)));
        code.push_str(&format!("{}let {} = self.alloc();\n", ident(tab+1), op2));
        code.push_str(&format!("{}self.safe_link(Trg::Ptr(Ptr::new(VR1, 0, {})), {});\n", ident(tab+1), op2, val.show()));
        code.push_str(&format!("{}self.safe_link(Trg::Ptr(Ptr::new(OP2, {}, {})), {});\n", ident(tab+1), ptr.lab(), op2, trg.show()));
        code.push_str(&format!("{}{} = Trg::Ptr(Ptr::new(VR2, 0, {}));\n", ident(tab+1), &nxt.show(), op2));
        code.push_str(&format!("{}}}\n", ident(tab)));
        code.push_str(&burn(book, tab, None, newx, vars, def, ret, &nxt));
        return code;
      }
    }

    // {p1 p2} <~ #N
    // ------------- fast copy
    // p1 <~ #N
    // p2 <~ #N
    if ptr.is_dup() {
      let x1 = Target { nam: format!("{}x", trg.show()) };
      let x2 = Target { nam: format!("{}y", trg.show()) };
      let p1 = def.node[ptr.loc() as usize].1;
      let p2 = def.node[ptr.loc() as usize].2;
      let lc = fresh(newx);
      code.push_str(&format!("{}let {} : Trg;\n", ident(tab), &x1.show()));
      code.push_str(&format!("{}let {} : Trg;\n", ident(tab), &x2.show()));
      code.push_str(&format!("{}// fast copy\n", ident(tab)));
      code.push_str(&format!("{}if {}.tag() == NUM {{\n", ident(tab), trg.get()));
      code.push_str(&format!("{}self.rwts.comm += 1;\n", ident(tab+1)));
      code.push_str(&format!("{}let got = {};\n", ident(tab+1), trg.take()));
      code.push_str(&format!("{}{} = Trg::Ptr(got);\n", ident(tab+1), &x1.show()));
      code.push_str(&format!("{}{} = Trg::Ptr(got);\n", ident(tab+1), &x2.show()));
      code.push_str(&format!("{}}} else {{\n", ident(tab)));
      code.push_str(&format!("{}let {} = self.alloc();\n", ident(tab+1), lc));
      code.push_str(&format!("{}{} = Trg::Ptr(Ptr::new(VR1, 0, {}));\n", ident(tab+1), &x1.show(), lc));
      code.push_str(&format!("{}{} = Trg::Ptr(Ptr::new(VR2, 0, {}));\n", ident(tab+1), &x2.show(), lc));
      code.push_str(&format!("{}self.safe_link(Trg::Ptr(Ptr::new({}, {}, {})), {});\n", ident(tab+1), tag(ptr.tag()), ptr.lab(), lc, trg.show()));
      code.push_str(&format!("{}}}\n", ident(tab)));
      code.push_str(&burn(book, tab, None, newx, vars, def, p2, &x2));
      code.push_str(&burn(book, tab, None, newx, vars, def, p1, &x1));
      return code;
    }

    // (p1 p2) <~ (x1 x2)
    // ------------------ fast apply
    // p1 <~ x1
    // p2 <~ x2
    if ptr.is_ctr() && ptr.tag() == run::LAM {
      let x1 = Target { nam: format!("{}x", trg.show()) };
      let x2 = Target { nam: format!("{}y", trg.show()) };
      let p1 = def.node[ptr.loc() as usize].1;
      let p2 = def.node[ptr.loc() as usize].2;
      let lc = fresh(newx);
      code.push_str(&format!("{}let {} : Trg;\n", ident(tab), &x1.show()));
      code.push_str(&format!("{}let {} : Trg;\n", ident(tab), &x2.show()));
      code.push_str(&format!("{}// fast apply\n", ident(tab)));
      code.push_str(&format!("{}if {}.tag() == {} {{\n", ident(tab), trg.get(), tag(ptr.tag())));
      code.push_str(&format!("{}self.rwts.anni += 1;\n", ident(tab+1)));
      code.push_str(&format!("{}let got = {};\n", ident(tab+1), trg.take()));
      code.push_str(&format!("{}{} = Trg::Dir(Ptr::new(VR1, 0, got.loc()));\n", ident(tab+1), &x1.show()));
      code.push_str(&format!("{}{} = Trg::Dir(Ptr::new(VR2, 0, got.loc()));\n", ident(tab+1), &x2.show()));
      code.push_str(&format!("{}}} else {{\n", ident(tab)));
      code.push_str(&format!("{}let {} = self.alloc();\n", ident(tab+1), lc));
      code.push_str(&format!("{}{} = Trg::Ptr(Ptr::new(VR1, 0, {}));\n", ident(tab+1), &x1.show(), lc));
      code.push_str(&format!("{}{} = Trg::Ptr(Ptr::new(VR2, 0, {}));\n", ident(tab+1), &x2.show(), lc));
      code.push_str(&format!("{}self.safe_link(Trg::Ptr(Ptr::new({}, 0, {})), {});\n", ident(tab+1), tag(ptr.tag()), lc, trg.show()));
      code.push_str(&format!("{}}}\n", ident(tab)));
      code.push_str(&burn(book, tab, None, newx, vars, def, p2, &x2));
      code.push_str(&burn(book, tab, None, newx, vars, def, p1, &x1));
      return code;
    }

    //// TODO: implement inlining correctly
    //// NOTE: enabling this makes dec_bits_tree hang; investigate
    //if ptr.is_ref() && tail.is_some() {
      //code.push_str(&format!("{}// inline @{}\n", ident(tab), ast::val_to_name(ptr.loc() as run::Val)));
      //code.push_str(&format!("{}if !{}.is_skp() {{\n", ident(tab), trg.get()));
      //code.push_str(&format!("{}self.rwts.dref += 1;\n", ident(tab+1)));
      //code.push_str(&call(book, tab+1, tail, newx, &mut HashMap::new(), ptr.loc(), trg));
      //code.push_str(&format!("{}}} else {{\n", ident(tab)));
      //code.push_str(&make(tab+1, newx, vars, def, ptr, &trg.show()));
      //code.push_str(&format!("{}}}\n", ident(tab)));
      //return code;
    //}

    // ATOM <~ *
    // --------- fast erase
    // nothing
    if ptr.is_num() || ptr.is_era() {
      code.push_str(&format!("{}// fast erase\n", ident(tab)));
      code.push_str(&format!("{}if {}.is_skp() {{\n", ident(tab), trg.get()));
      code.push_str(&format!("{}{};\n", ident(tab+1), trg.take()));
      code.push_str(&format!("{}self.rwts.eras += 1;\n", ident(tab+1)));
      code.push_str(&format!("{}}} else {{\n", ident(tab)));
      code.push_str(&make(tab+1, newx, vars, def, ptr, &trg.show()));
      code.push_str(&format!("{}}}\n", ident(tab)));
      return code;
    }

    code.push_str(&make(tab, newx, vars, def, ptr, &trg.show()));
    return code;
  }

  fn make(
    tab  : usize,
    newx : &mut usize,
    vars : &mut HashMap<run::Ptr, String>,
    def  : &run::Def,
    ptr  : run::Ptr,
    trg  : &String,
  ) -> String {
    //println!("make {:08x} {}", ptr.0, x);
    let mut code = String::new();
    if ptr.is_nod() {
      let lc = fresh(newx);
      let p1 = def.node[ptr.loc() as usize].1;
      let p2 = def.node[ptr.loc() as usize].2;
      code.push_str(&format!("{}let {} = self.alloc();\n", ident(tab), lc));
      code.push_str(&make(tab, newx, vars, def, p2, &format!("Trg::Ptr(Ptr::new(VR2, 0, {}))", lc)));
      code.push_str(&make(tab, newx, vars, def, p1, &format!("Trg::Ptr(Ptr::new(VR1, 0, {}))", lc)));
      code.push_str(&format!("{}self.safe_link(Trg::Ptr(Ptr::new({}, {}, {})), {});\n", ident(tab), tag(ptr.tag()), ptr.lab(), lc, trg));
    } else if ptr.is_var() {
      match got(vars, def, ptr) {
        None => {
          //println!("-var fst");
          vars.insert(ptr, trg.clone());
        },
        Some(got) => {
          //println!("-var snd");
          code.push_str(&format!("{}self.safe_link({}, {});\n", ident(tab), trg, got));
        }
      }
    } else {
      code.push_str(&format!("{}self.safe_link({}, Trg::Ptr({}));\n", ident(tab), trg, atom(ptr)));
    }
    return code;
  }

  fn got(
    vars : &HashMap<run::Ptr, String>,
    def  : &run::Def,
    ptr  : run::Ptr,
  ) -> Option<String> {
    if ptr.is_var() {
      let got = def.node[ptr.loc() as usize];
      let slf = if ptr.tag() == run::VR1 { got.1 } else { got.2 };
      return vars.get(&slf).cloned();
    } else {
      return None;
    }
  }

  let fun = ast::val_to_name(fid);
  let def = &book.get(fid).unwrap();

  let mut code = String::new();
  // Given a label, returns true if the definition contains that dup label, directly or not
  code.push_str(&format!("{}pub fn L_{}(&mut self, lab: Lab) -> bool {{\n", ident(tab), fun));
  for dup in &def.labs {
    code.push_str(&format!("{}if lab == 0x{:x} {{ return true; }}\n", ident(tab+1), dup));
  }
  code.push_str(&format!("{}return false;\n", ident(tab+1)));
  code.push_str(&format!("{}}}\n", ident(tab)));
  // Calls the definition, performing inline rewrites when possible, and expanding it when not
  code.push_str(&format!("{}pub fn F_{}(&mut self, ptr: Ptr, trg: Trg) -> bool {{\n", ident(tab), fun));
  code.push_str(&format!("{}if self.get(trg).is_dup() && !self.L_{}(self.get(trg).lab()) {{\n", ident(tab+1), fun));
  code.push_str(&format!("{}self.copy(self.swap(trg, NULL), ptr);\n", ident(tab+2)));
  code.push_str(&format!("{}return true;\n", ident(tab+2)));
  code.push_str(&format!("{}}}\n", ident(tab+1)));
  code.push_str(&call(book, tab+1, None, &mut 0, &mut HashMap::new(), fid, &Target { nam: "trg".to_string() }));
  code.push_str(&format!("{}return true;\n", ident(tab+1)));
  code.push_str(&format!("{}}}\n", ident(tab)));

  return code;
}

// TODO: HVM-Lang must always output in this form.
fn adjust_redex(rf: run::Ptr, rx: run::Ptr) -> (run::Ptr, run::Ptr) {
  if rf.is_skp() && !rx.is_skp() {
    return (rf, rx);
  } else if !rf.is_skp() && rx.is_skp() {
    return (rx, rf);
  } else {
    println!("Invalid redex. Compiled HVM requires that ALL defs are in the form:");
    println!("@name = ROOT");
    println!("  & ATOM ~ TERM");
    println!("  & ATOM ~ TERM");
    println!("  & ATOM ~ TERM");
    println!("  ...");
    println!("Where ATOM must be either a ref (`@foo`), a num (`#123`), or an era (`*`).");
    println!("If you used HVM-Lang, please report on https://github.com/HigherOrderCO/hvm-lang.");
    panic!("Invalid HVMC file.");
  }
}
