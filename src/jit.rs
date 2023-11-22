// Despite the file name, this is not actually a JIT (yet).

use crate::run;
use crate::ast;

use std::collections::HashMap;

pub fn compile_book(book: &run::Book) -> String {
  let mut code = String::new();

  code.push_str(&format!("use crate::run::{{*}};\n"));
  code.push_str(&format!("\n"));

  for fid in 0 .. book.defs.len() as run::Val {
    if book.defs[fid as usize].node.len() > 0 {
      let name = &ast::val_to_name(fid as run::Val);
      code.push_str(&format!("pub const F_{:4} : Loc = 0x{:06x};\n", name, fid));
    }
  }
  code.push_str(&format!("\n"));

  code.push_str(&format!("impl<'a> Net<'a> {{\n"));
  code.push_str(&format!("\n"));

  code.push_str(&format!("{}pub fn call_native(&mut self, book: &Book, ptr: Ptr, x: Ptr) -> bool {{\n", ident(1)));
  code.push_str(&format!("{}match ptr.loc() {{\n", ident(2)));
  for fid in 0 .. book.defs.len() as run::Val {
    if book.defs[fid as usize].node.len() > 0 {
      let fun = ast::val_to_name(fid);
      code.push_str(&format!("{}F_{} => {{ return self.F_{}(ptr, x); }}\n", ident(3), fun, fun));
    }
  }
  code.push_str(&format!("{}_ => {{ return false; }}\n", ident(3)));
  code.push_str(&format!("{}}}\n", ident(2)));
  code.push_str(&format!("{}}}\n", ident(1)));
  code.push_str(&format!("\n"));

  for fid in 0 .. book.defs.len() as run::Loc {
    if book.defs[fid as usize].node.len() > 0 {
      code.push_str(&compile_term(&book, 1, fid));
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
    return format!("Ptr::new(REF, 0, F_{})", ast::val_to_name(ptr.loc() as run::Val));
  } else {
    return format!("Ptr::new({}, 0, 0x{:x})", tag(ptr.tag()), ptr.loc());
  }
}

#[derive(Clone)]
enum Target {
  External { nam: String },
  Internal { nam: String },
}

impl Target {
  fn name(&self) -> String {
    match self {
      Target::External { nam } => format!("{}", nam),
      Target::Internal { nam } => format!("{}", nam),
    }
  }

  fn get(&self) -> String {
    match self {
      Target::External { nam } => format!("self.get_target({})", nam),
      Target::Internal { nam } => format!("{}", nam),
    }
  }

  fn take(&self) -> String {
    match self {
      Target::External { nam } => format!("self.swap_target({}, NULL)", nam),
      Target::Internal { nam } => format!(""),
    }
  }

  fn link(&self, tab: usize, lnks: &mut Vec<(bool,String)>, to: &Target) {
    match (self, to) {
      (Target::Internal { nam: a_nam }, Target::Internal { nam: b_nam }) => {
        lnks.push((false, format!("{}self.link({}, {});\n", ident(tab), a_nam, b_nam)));
      }
      (Target::Internal { nam: a_nam }, Target::External { nam: b_nam }) => {
        lnks.push((true, format!("{}self.half_atomic_link({}, {});\n", ident(tab), b_nam, a_nam)));
      }
      (Target::External { nam: a_nam }, Target::Internal { nam: b_nam }) => {
        lnks.push((true, format!("{}self.half_atomic_link({}, {});\n", ident(tab), a_nam, b_nam)));
      }
      (Target::External { nam: a_nam }, Target::External { nam: b_nam }) => {
        lnks.push((true, format!("{}self.atomic_link({}, {});\n", ident(tab), a_nam, b_nam)));
      }
    }
  }
}

pub fn compile_term(book: &run::Book, tab: usize, fid: run::Loc) -> String {

  // returns a fresh variable: 'v<NUM>'
  fn fresh(newx: &mut usize) -> String {
    *newx += 1;
    format!("k{}", newx)
  }

  fn func(
    book : &run::Book,
    tab  : usize,
    fid  : run::Loc,
  ) -> String {

    // Gets function
    let def = &book.defs[fid as usize];
    let fun = ast::val_to_name(fid as run::Val);
    let trg = Target::Internal { nam: "trg".to_string() };

    // Inits code
    let mut code = String::new();

    // Slow path
    let newx = &mut 0;
    let vars = &mut HashMap::new();
    let lnks = &mut Vec::new();
    code.push_str(&format!("{}pub fn F_{}_slow(&mut self, ptr: Ptr, trg: Ptr) -> bool {{\n", ident(tab), fun));
    for (rf, rx) in &def.rdex {
      let (rf, rx) = adjust_redex(*rf, *rx);
      code.push_str(&make(tab+1, newx, vars, lnks, def, rx, &Target::Internal { nam: atom(rf) }));
    }
    code.push_str(&make(tab+1, newx, vars, lnks, def, def.node[0].1, &trg));
    for (_, lnk) in lnks.iter().rev() { code.push_str(&lnk); }
    code.push_str(&format!("{}return true;\n", ident(tab+1)));
    code.push_str(&format!("{}}}\n\n", ident(tab)));

    // Fast path
    let newx = &mut 0;
    let vars = &mut HashMap::new();
    let lnks = &mut Vec::new();
    code.push_str(&format!("{}#[inline(always)]\n", ident(tab)));
    code.push_str(&format!("{}pub fn F_{}_fast(&mut self, ptr: Ptr, trg: Ptr) -> bool {{\n", ident(tab), fun));
    code.push_str(&is_fast(tab+1, def, def.node[0].1, &trg));
    for (rf, rx) in &def.rdex {
      let (rf, rx) = adjust_redex(*rf, *rx);
      code.push_str(&make(tab+1, newx, vars, lnks, def, rx, &Target::Internal { nam: atom(rf) }));
    }
    code.push_str(&go_fast(tab+1, newx, vars, lnks, def, def.node[0].1, &trg));
    for (ext, lnk) in lnks.iter().rev() { if !*ext { code.push_str(&lnk); } }
    for (ext, lnk) in lnks.iter().rev() { if  *ext { code.push_str(&lnk); } }
    code.push_str(&format!("{}return true;\n", ident(tab+1)));
    code.push_str(&format!("{}}}\n\n", ident(tab)));

    // Caller
    code.push_str(&format!("{}pub fn F_{}(&mut self, ptr: Ptr, trg: Ptr) -> bool {{\n", ident(tab), fun));
    code.push_str(&format!("{}if self.F_{}_fast(ptr, trg) {{\n", ident(tab+1), fun));
    code.push_str(&format!("{}return true;\n", ident(tab+2)));
    code.push_str(&format!("{}}}\n", ident(tab+1)));
    code.push_str(&format!("{}self.F_{}_slow(ptr, trg);\n", ident(tab+1), fun));
    code.push_str(&format!("{}return true;\n", ident(tab+1)));
    code.push_str(&format!("{}}}\n\n", ident(tab)));

    return code;
  }
  
  fn is_fast(
    tab : usize,
    def : &run::Def,
    ptr : run::Ptr,
    trg : &Target,
  ) -> String {
    let mut code = String::new();

    // (p1 p2) <~ (x1 x2)
    // ------------------ fast apply
    // p1 <~ x1
    // p2 <~ x2
    if ptr.is_ctr() && ptr.tag() == run::LAM {
      let p1 = def.node[ptr.loc() as usize].0;
      let p2 = def.node[ptr.loc() as usize].1;
      let x1 = Target::External { nam: format!("{}x", trg.name()) };
      let x2 = Target::External { nam: format!("{}y", trg.name()) };
      code.push_str(&format!("{}let {} : Ptr;\n", ident(tab), &x1.name()));
      code.push_str(&format!("{}let {} : Ptr;\n", ident(tab), &x2.name()));
      code.push_str(&format!("{}// fast apply\n", ident(tab)));
      code.push_str(&format!("{}if {}.tag() == {} {{\n", ident(tab), trg.get(), tag(ptr.tag())));
      code.push_str(&format!("{}let got = {};\n", ident(tab+1), trg.get()));
      code.push_str(&format!("{}{} = Ptr::new(VR1, 0, got.loc());\n", ident(tab+1), &x1.name()));
      code.push_str(&format!("{}{} = Ptr::new(VR2, 0, got.loc());\n", ident(tab+1), &x2.name()));
      code.push_str(&format!("{}}} else {{\n", ident(tab)));
      code.push_str(&format!("{}return false;\n", ident(tab+1)));
      code.push_str(&format!("{}}}\n", ident(tab)));
      code.push_str(&is_fast(tab, def, p1, &x1));
      code.push_str(&is_fast(tab, def, p2, &x2));
      return code;
    }

    return code;
  }

  fn go_fast(
    tab  : usize,
    newx : &mut usize,
    vars : &mut HashMap<run::Ptr, Target>,
    lnks : &mut Vec<(bool, String)>,
    def  : &run::Def,
    ptr  : run::Ptr,
    trg  : &Target,
  ) -> String {
    let mut code = String::new();

    // (p1 p2) <~ (x1 x2)
    // ------------------ fast apply
    // p1 <~ x1
    // p2 <~ x2
    if ptr.is_ctr() && ptr.tag() == run::LAM {
      let p1 = def.node[ptr.loc() as usize].0;
      let p2 = def.node[ptr.loc() as usize].1;
      let x1 = Target::External { nam: format!("{}x", trg.name()) };
      let x2 = Target::External { nam: format!("{}y", trg.name()) };
      code.push_str(&go_fast(tab, newx, vars, lnks, def, p1, &x1));
      code.push_str(&go_fast(tab, newx, vars, lnks, def, p2, &x2));
      code.push_str(&format!("{}self.rwts.anni += 1;\n", ident(tab)));
      let taken = trg.take();
      if taken.len() > 0 { code.push_str(&format!("{}{};\n", ident(tab), taken)); }
      return code;
    }

    code.push_str(&make(tab, newx, vars, lnks, def, ptr, trg));
    return code;
  }

  fn make(
    tab  : usize,
    newx : &mut usize,
    vars : &mut HashMap<run::Ptr, Target>,
    lnks : &mut Vec<(bool, String)>,
    def  : &run::Def,
    ptr  : run::Ptr,
    trg  : &Target,
  ) -> String {
    //println!("make {:08x} {}", ptr.0, x);
    let mut code = String::new();
    if ptr.is_nod() {
      let lc = fresh(newx);
      let p1 = def.node[ptr.loc() as usize].0;
      let p2 = def.node[ptr.loc() as usize].1;
      let x1 = Target::Internal { nam: fresh(newx) };
      let x2 = Target::Internal { nam: fresh(newx) };
      code.push_str(&format!("{}let {} = self.alloc(1);\n", ident(tab), lc));
      code.push_str(&format!("{}let {} = {};\n", ident(tab), &x1.name(), format!("Ptr::new(VR1, 0, {})", lc)));
      code.push_str(&format!("{}let {} = {};\n", ident(tab), &x2.name(), format!("Ptr::new(VR2, 0, {})", lc)));
      code.push_str(&make(tab, newx, vars, lnks, def, p1, &x1));
      code.push_str(&make(tab, newx, vars, lnks, def, p2, &x2));
      trg.link(tab, lnks, &Target::Internal { nam: format!("Ptr::new({}, {}, {})", tag(ptr.tag()), ptr.lab(), lc) });
    } else if ptr.is_var() {
      if let Some(got) = find(vars, def, ptr) {
        trg.link(tab, lnks, &got);
      } else {
        vars.insert(ptr, trg.clone());
      }
    } else {
      trg.link(tab, lnks, &Target::Internal { nam: atom(ptr) });
    }
    return code;
  }

  fn find(
    vars : &HashMap<run::Ptr, Target>,
    def  : &run::Def,
    ptr  : run::Ptr,
  ) -> Option<Target> {
    if ptr.is_var() {
      let got = def.node[ptr.loc() as usize];
      let slf = if ptr.tag() == run::VR1 { got.0 } else { got.1 };
      return vars.get(&slf).cloned();
    } else {
      return None;
    }
  }

  let mut code = String::new();
  code.push_str(&func(book, tab, fid));

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
