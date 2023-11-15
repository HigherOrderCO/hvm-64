use std::collections::HashMap;
use crate::ir::{Con, Function, Ins, Stmt};
use crate::run::{self, Book, Def, Ptr, Val};
use crate::ast;

pub fn compile_term(book: &Book, fid: Val) -> Function {
  type Target = String;

  struct Lowering<'book> {
    newx: usize,
    book: &'book Book,
    vars: HashMap<Ptr, String>,
    target: Target,
    stmts: Vec<Stmt>,
  }

  impl Lowering<'_> {
    /// returns a fresh variable: 'v<NUM>'
    fn fresh(&mut self) -> Ins {
      let name = format!("v{}", self.newx);
      self.newx += 1;
      Ins::Var { name }
    }

    /// returns a fresh variable: 'v<NUM>'
    fn fresh_name(&mut self) -> String {
      let name = format!("v{}", self.newx);
      self.newx += 1;
      name
    }

    fn call(&mut self, fid: Val) {
      let def = &self.book.defs[fid as usize];
      let mut code = self.burn(def, def.node[0].1, self.target.clone());
      for (rf, rx) in &def.rdex {
        let (rf, rx) = adjust_redex(*rf, *rx);
        let ref_name = self.fresh_name();
        self.stmts.push(Stmt::Let {
          name: ref_name.clone(),
          value: assert_is_atom(rf),
        });
        self.burn(def, rx, ref_name);
      }
    }

    /// @loop = (?<(#0 (x y)) R> R) & @loop ~ (x y)
    fn burn(&mut self, def: &Def, ptr: Ptr, target: Target) {

    }
  }

  let mut lowering = Lowering {
    newx: 0,
    book,
    target: "x".to_string(),
    vars: HashMap::new(),
    stmts: vec![],
  };

  lowering.call(fid);
  lowering.stmts.push(Stmt::Return(Ins::True));

  Function {
    name: ast::val_to_name(fid),
    body: lowering.stmts,
  }
}

fn assert_is_atom(ptr: Ptr) -> Ins {
  if ptr.is_ref() {
    Ins::NewPtr {
      tag: Ins::from(Con::REF).into(),
      value: Ins::from(Con::F(ast::val_to_name(ptr.val()))).into(),
    }
  } else {
    Ins::NewPtr {
      tag: Ins::from(compile_tag(ptr.tag())).into(),
      value: Ins::Hex(ptr.val()).into(),
    }
  }
}

 fn compile_tag(tag: run::Tag) -> Con {
  match tag {
    run::VR1 => Con::VR1,
    run::VR2 => Con::VR2,
    run::RD1 => Con::RD1,
    run::RD2 => Con::RD2,
    run::REF => Con::REF,
    run::ERA => Con::ERA,
    run::NUM => Con::NUM,
    run::OP2 => Con::OP2,
    run::OP1 => Con::OP1,
    run::MAT => Con::MAT,
    run::CT0 => Con::CT0,
    run::CT1 => Con::CT1,
    run::CT2 => Con::CT2,
    run::CT3 => Con::CT3,
    run::CT4 => Con::CT4,
    run::CT5 => Con::CT5,
    _ => unreachable!(),
  }
}

// TODO: HVM-Lang must always output in this form.
fn adjust_redex(rf: Ptr, rx: Ptr) -> (Ptr, Ptr) {
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
