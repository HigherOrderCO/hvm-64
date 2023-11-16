//! JIT compilation of Cranelift IR.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::str::FromStr;

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, Linkage, Module};
use Stmt::SetHeap;

use crate::ast;
use crate::run::{self, Book, Def, Ptr, Val};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeRepr {
  HvmPtr,
  Ptr,
  USize,
  U8,
  U32,
  Bool,
}

/// Constant values
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Const {
  F(String),
  P1,   // [crate::run::P1]
  P2,   // [crate::run::P2]
  NULL, // [crate::run::NULL]
  ROOT, // [crate::run::ROOT]
  ERAS, // [crate::run::ERAS]
  VR1,  // [crate::run::VR1]
  VR2,  // [crate::run::VR2]
  RD1,  // [crate::run::RD1]
  RD2,  // [crate::run::RD2]
  REF,  // [crate::run::REF]
  ERA,  // [crate::run::ERA]
  NUM,  // [crate::run::NUM]
  OP1,  // [crate::run::OP1]
  OP2,  // [crate::run::OP2]
  MAT,  // [crate::run::MAT]
  CT0,  // [crate::run::CT0]
  CT1,  // [crate::run::CT1]
  CT2,  // [crate::run::CT2]
  CT3,  // [crate::run::CT3]
  CT4,  // [crate::run::CT4]
  CT5,  // [crate::run::CT5]
  USE,  // [crate::run::USE]
  ADD,  // [crate::run::ADD]
  SUB,  // [crate::run::SUB]
  MUL,  // [crate::run::MUL]
  DIV,  // [crate::run::DIV]
  MOD,  // [crate::run::MOD]
  EQ,   // [crate::run::EQ]
  NE,   // [crate::run::NE]
  LT,   // [crate::run::LT]
  GT,   // [crate::run::GT]
  AND,  // [crate::run::AND]
  OR,   // [crate::run::OR]
  XOR,  // [crate::run::XOR]
  NOT,  // [crate::run::NOT]
  RSH,  // [crate::run::RSH]
  LSH,  // [crate::run::LSH]
}

/// Constant values
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Prop {
  Anni,
  Oper,
  Eras,
  Comm,
  Var(String),
}

pub struct Constant {
  pub name: String,
  pub value: u32,
}

/// Represents the entire IR.
pub struct Program {
  pub functions: Vec<Function>,
  pub values: Vec<Constant>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
  pub name: String,
  pub body: Vec<Stmt>,
}

/// Represents a single statement in the IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
  Let {
    name: String,
    value: Instr,
  },
  Val {
    name: String,
    type_repr: TypeRepr,
  },
  Assign {
    name: Prop,
    value: Instr,
  },
  Instr(Instr),
  Free(Instr),
  Return(Instr),
  /// self.heap.set(idx, port, value)
  SetHeap {
    idx: Instr,
    port: Instr,
    value: Instr,
  },
  /// self.link(lhs, rhs)
  Link {
    lhs: Instr,
    rhs: Instr,
  },
}

/// Represents a single instruction in the IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instr {
  True,
  False,
  Int(u32),
  Const(Const),
  Prop(Prop),

  /// SPECIAL TERM: Call
  /// if cond then else els
  If {
    cond: Box<Instr>,
    then: Vec<Stmt>,
    otherwise: Vec<Stmt>,
  },
  /// !ins
  Not {
    ins: Box<Instr>,
  },
  /// lhs op rhs
  Bin {
    op: String,
    lhs: Box<Instr>,
    rhs: Box<Instr>,
  },

  // VALUE FUNCTIONS:
  // These are the functions that are exposed to the user.
  /// ins.val()
  Val {
    ins: Box<Instr>,
  },
  /// ins.tag()
  Tag {
    ins: Box<Instr>,
  },
  /// ins.is_num()
  IsNum {
    ins: Box<Instr>,
  },
  /// ins.is_skp()
  IsSkp {
    ins: Box<Instr>,
  },
  /// Ptr::new(tag, value)
  NewPtr {
    tag: Box<Instr>,
    value: Box<Instr>,
  },

  // FUNCTIONS:
  // These are the functions that are internal to the IR.
  /// self.ops(lhs, op, rhs)
  Op {
    lhs: Box<Instr>,
    rhs: Box<Instr>,
  },
  /// self.alloc(n)
  Alloc {
    size: usize,
  },
  /// self.heap.get(idx, port)
  GetHeap {
    idx: Box<Instr>,
    port: Box<Instr>,
  },
}

impl From<Const> for Instr {
  fn from(value: Const) -> Self {
    Instr::Const(value)
  }
}

impl From<String> for Instr {
  fn from(value: String) -> Self {
    Instr::Prop(Prop::Var(value))
  }
}

impl From<Prop> for Instr {
  fn from(value: Prop) -> Self {
    Instr::Prop(value)
  }
}

pub fn compile_book(book: &Book) -> Program {
  let mut functions = vec![];
  let mut values = vec![];

  for fid in 0..book.defs.len() as run::Val {
    let name = ast::val_to_name(fid as Val);
    if book.defs[fid as usize].node.len() > 0 {
      functions.push(compile_term(book, fid as Val));
      values.push(Constant {
        name,
        value: fid as u32,
      });
    }
  }

  Program { functions, values }
}

pub fn compile_term(book: &Book, fid: Val) -> Function {
  let mut lowering = Lowering {
    newx: Rc::new(Cell::new(0)),
    book,
    target: Instr::from("argument".to_string()),
    vars: Rc::new(RefCell::new(HashMap::new())),
    stmts: vec![],
  };

  lowering.call(fid);
  lowering.stmts.push(Stmt::Return(Instr::True));

  Function {
    name: ast::val_to_name(fid),
    body: lowering.stmts,
  }
}

fn assert_is_atom(ptr: Ptr) -> Instr {
  if ptr.is_ref() {
    Instr::NewPtr {
      tag: Instr::from(Const::REF).into(),
      value: Instr::from(Const::F(ast::val_to_name(ptr.val()))).into(),
    }
  } else {
    Instr::NewPtr {
      tag: Instr::from(compile_tag(ptr.tag())).into(),
      value: Instr::Int(ptr.val()).into(),
    }
  }
}

fn compile_tag(tag: run::Tag) -> Const {
  match tag {
    run::VR1 => Const::VR1,
    run::VR2 => Const::VR2,
    run::RD1 => Const::RD1,
    run::RD2 => Const::RD2,
    run::REF => Const::REF,
    run::ERA => Const::ERA,
    run::NUM => Const::NUM,
    run::OP2 => Const::OP2,
    run::OP1 => Const::OP1,
    run::MAT => Const::MAT,
    run::CT0 => Const::CT0,
    run::CT1 => Const::CT1,
    run::CT2 => Const::CT2,
    run::CT3 => Const::CT3,
    run::CT4 => Const::CT4,
    run::CT5 => Const::CT5,
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

type Target = String;

#[derive(Clone)]
struct Lowering<'book> {
  newx: Rc<Cell<usize>>,
  book: &'book Book,
  vars: Rc<RefCell<HashMap<Ptr, Instr>>>,
  target: Instr,
  stmts: Vec<Stmt>,
}

impl Lowering<'_> {
  /// returns a fresh variable: 'v<NUM>'
  fn fresh(&mut self) -> Instr {
    let name = format!("v{}", self.newx.get());
    self.newx.set(self.newx.get() + 1);
    Instr::Prop(Prop::Var(name))
  }

  /// returns a fresh variable: 'v<NUM>'
  fn fresh_name(&mut self) -> String {
    let name = format!("v{}", self.newx.get());
    self.newx.set(self.newx.get() + 1);
    name
  }

  /// Compiles a function call
  fn call(&mut self, fid: Val) {
    let def = &self.book.defs[fid as usize];
    self.burn(def, def.node[0].1, self.target.clone());
    for (rf, rx) in &def.rdex {
      let (rf, rx) = adjust_redex(*rf, *rx);
      let ref_name = self.fresh_name();
      self.stmts.push(Stmt::Let {
        name: ref_name.clone(),
        value: assert_is_atom(rf),
      });
      self.burn(def, rx, ref_name.into());
    }
  }

  /// Declares a variable without name
  fn declare_fresh(&mut self, type_repr: TypeRepr) -> String {
    let name = self.fresh_name();
    self.stmts.push(Stmt::Val {
      name: name.clone(),
      type_repr,
    });
    name
  }

  /// Declares a variable
  fn declare(&mut self, name: String, type_repr: TypeRepr) -> String {
    self.stmts.push(Stmt::Val {
      name: name.clone(),
      type_repr,
    });
    name
  }

  /// Defines a variable without name
  fn define_fresh(&mut self, value: Instr) -> String {
    let name = self.fresh_name();
    self.stmts.push(Stmt::Let {
      name: name.clone(),
      value,
    });
    name
  }

  /// Defines a variable
  fn define(&mut self, name: String, value: Instr) -> String {
    self.stmts.push(Stmt::Let {
      name: name.clone(),
      value,
    });
    name
  }

  /// Assigns a value to a property
  fn assign(&mut self, prop: Prop, value: Instr) {
    self.stmts.push(Stmt::Assign { name: prop, value });
  }

  /// Fork returning Self
  fn fork(&self) -> Self {
    let mut new_fork = self.clone();
    new_fork.stmts = vec![];
    new_fork
  }

  /// Fork returning vec of statements
  fn fork_on(&self, f: impl FnOnce(&mut Self)) -> Vec<Stmt> {
    let mut fork = self.fork();
    f(&mut fork);
    fork.stmts
  }

  /// Generates code
  fn make(&mut self, def: &Def, ptr: Ptr, target: Instr) {
    if ptr.is_nod() {
      let lc = self.define_fresh(Instr::Alloc { size: 1 });
      let (p1, p2) = def.node[ptr.val() as usize];
      self.make(def, p1, Instr::new_ptr(Const::VR1, lc.clone().into()));
      self.make(def, p2, Instr::new_ptr(Const::VR2, lc.clone().into()));
      self
        .stmts
        .push(Instr::new_ptr(compile_tag(ptr.tag()), lc.into()).link(target.clone()));
    } else if ptr.is_var() {
      match self.get(def, ptr) {
        None => {
          self.vars.borrow_mut().insert(ptr, target.clone());
        }
        Some(value) => {
          self.stmts.push(target.clone().link(value));
        }
      }
    } else {
      self.stmts.push(target.clone().link(assert_is_atom(ptr)));
    }
  }

  /// Get value from vars
  fn get(&self, def: &Def, ptr: Ptr) -> Option<Instr> {
    if ptr.is_var() {
      let got = def.node[ptr.val() as usize];
      let slf = if ptr.tag() == run::VR1 { got.0 } else { got.1 };
      self.vars.borrow().get(&slf).cloned()
    } else {
      None
    }
  }

  /// @loop = (?<(#0 (x y)) R> R) & @loop ~ (x y)
  ///
  /// This function basically concentrates all the optimizations
  fn burn(&mut self, def: &Def, ptr: Ptr, target: Instr) {
    // (<?(ifz ifs) ret> ret) ~ (#X R)
    // ------------------------------- fast match
    // if X == 0:
    //   ifz ~ R
    //   ifs ~ *
    // else:
    //   ifz ~ *
    //   ifs ~ (#(X-1) R)
    // When ifs is REF, tail-call optimization is applied.
    if fast_match(self, def, ptr, target.clone()) {
      return;
    }

    // <x <y r>> ~ #N
    // --------------------- fast op
    // r <~ #(op(op(N,x),y))
    if fast_op(self, def, ptr, target.clone()) {
      return;
    }

    // {p1 p2} <~ #N
    // ------------- fast copy
    // p1 <~ #N
    // p2 <~ #N
    if fast_copy(self, def, ptr, target.clone()) {
      return;
    }

    // (p1 p2) <~ (x1 x2)
    // ------------------ fast apply
    // p1 <~ x1
    // p2 <~ x2
    if fast_apply(self, def, ptr, target.clone()) {
      return;
    }

    // ATOM <~ *
    // --------- fast erase
    // nothing
    if fast_erase(self, def, ptr, target.clone()) {
      return;
    }

    self.make(def, ptr, target);
  }
}

/// (<?(ifz ifs) ret> ret) ~ (#X R)
/// ------------------------------- fast match
/// if X == 0:
///   ifz ~ R
///   ifs ~ *
/// else:
///   ifz ~ *
///   ifs ~ (#(X-1) R)
/// When ifs is REF, tail-call optimization is applied.
fn fast_match(lowering: &mut Lowering, def: &Def, ptr: Ptr, target: Instr) -> bool {
  if ptr.tag() == run::CT0 {
    let (mat, rty) = def.node[ptr.val() as usize];
    if mat.tag() == run::MAT {
      let got @ (cse, rtx) = def.node[rty.val() as usize];
      let rtz = if rty.tag() == run::VR1 { got.0 } else { got.1 };
      if cse.tag() == run::CT0 && rtx.is_var() && rtx == rtz {
        let (ifz, ifs) = def.node[cse.val() as usize];
        let c_z = lowering.declare_fresh(TypeRepr::HvmPtr);
        let c_s = lowering.declare_fresh(TypeRepr::HvmPtr);
        // FAST MATCH
        // if tag(target) = CT0 && is-num(get-heap(val(target))
        lowering.stmts.push(Stmt::Instr(Instr::If {
          cond: Instr::from(target.clone())
            .eq(Instr::from(Const::CT0))
            .and(
              Instr::GetHeap {
                idx: Instr::from(target.clone()).val().into(),
                port: Instr::from(Const::P1).into(),
              }
              .is_num(),
            )
            .into(),
          then: lowering.fork_on(|lowering| {
            // self.anni += 2
            lowering.assign(Prop::Anni, Instr::from(Prop::Anni).add(Instr::Int(2)));

            // self.oper += 1
            lowering.assign(Prop::Oper, Instr::from(Prop::Anni).add(Instr::Int(1)));

            // let num = self.heap.get(target.val(), P1)
            let num = lowering.define_fresh(Instr::GetHeap {
              idx: Instr::from(target.clone()).val().into(),
              port: Instr::from(Const::P1).into(),
            });

            // let res = self.heap.get(target.val(), P2)
            let res = lowering.define_fresh(Instr::GetHeap {
              idx: Instr::from(target.clone()).val().into(),
              port: Instr::from(Const::P2).into(),
            });

            // if num.val() == 0
            //   self.free(target.val())
            //   c_z = res
            //   c_s = ERAS
            // else
            //   self.heap.set(target.val(), P1, Ptr::new(NUM, num.val() - 1))
            //   c_z = ERAS
            //   c_s = target
            lowering.stmts.push(Stmt::Instr(Instr::If {
              cond: Instr::from(num.clone()).eq(Instr::Int(0)).into(),
              then: lowering.fork_on(|lowering| {
                lowering
                  .stmts
                  .push(Stmt::Free(Instr::from(target.clone()).val()));
                lowering.assign(Prop::Var(c_z.clone()), Instr::from(res.clone()));
                lowering.assign(Prop::Var(c_s.clone()), Instr::from(Const::ERAS));
              }),
              otherwise: lowering.fork_on(|lowering| {
                lowering.stmts.push(Stmt::SetHeap {
                  idx: Instr::from(target.clone()).val().into(),
                  port: Instr::from(Const::P1).into(),
                  value: Instr::new_ptr(
                    Instr::from(Const::NUM),
                    Instr::from(num).sub(Instr::Int(1)),
                  )
                  .into(),
                });
                lowering.assign(Prop::Var(c_z.clone()), Instr::from(Const::ERAS));
                lowering.assign(Prop::Var(c_s.clone()), Instr::from(target.clone()));
              }),
            }))
          }),
          otherwise: lowering.fork_on(|lowering| {
            let lam = lowering.define_fresh(Instr::Alloc { size: 1 });
            let mat = lowering.define_fresh(Instr::Alloc { size: 1 });
            let cse = lowering.define_fresh(Instr::Alloc { size: 1 });
            lowering.stmts.push(SetHeap {
              // self.heap.set(lam, P1, Ptr::new(MAT, mat));
              idx: Instr::from(lam.clone()).into(),
              port: Instr::from(Const::P1).into(),
              value: Instr::new_ptr(Const::MAT, Instr::from(mat.clone())).into(),
            });
            lowering.stmts.push(SetHeap {
              // self.heap.set(lam, P2, Ptr::new(VR2, mat));
              idx: Instr::from(lam.clone()).into(),
              port: Instr::from(Const::P2).into(),
              value: Instr::new_ptr(Const::VR2, Instr::from(mat.clone())).into(),
            });
            lowering.stmts.push(SetHeap {
              // self.heap.set(mat, P1, Ptr::new(CT0, cse));
              idx: Instr::from(mat.clone()),
              port: Instr::from(Const::P1),
              value: Instr::new_ptr(Const::CT0, Instr::from(cse.clone())),
            });
            lowering.stmts.push(SetHeap {
              // self.heap.set(mat, P2, Ptr::new(VR2, cse));
              idx: Instr::from(mat.clone()),
              port: Instr::from(Const::P2),
              value: Instr::new_ptr(Const::VR2, Instr::from(lam.clone())),
            });
            lowering.stmts.push(
              Instr::new_ptr(Const::CT0, Instr::from(lam.clone()))
                .link(Instr::from(target.clone())),
            );
            lowering.assign(
              Prop::Var(c_z.clone()),
              Instr::new_ptr(Const::VR1, Instr::from(cse.clone())),
            );
            lowering.assign(
              Prop::Var(c_s.clone()),
              Instr::new_ptr(Const::VR2, Instr::from(cse.clone())),
            );
          }),
        }));
        lowering.burn(def, ifz, Instr::from(c_z).clone());
        lowering.burn(def, ifs, Instr::from(c_s).clone());
        return true;
      }
    }
  }
  false
}

// <x <y r>> ~ #N
// --------------------- fast op
// r <~ #(op(op(N,x),y))
fn fast_op(lowering: &mut Lowering, def: &Def, ptr: Ptr, target: Instr) -> bool {
  if ptr.is_op2() {
    let (v_x, cnt) = def.node[ptr.val() as usize];
    if cnt.is_op2() {
      let (v_y, ret) = def.node[cnt.val() as usize];
      if let (Some(v_x), Some(v_y)) = (lowering.get(def, v_x), lowering.get(def, v_y)) {
        let nxt = lowering.declare_fresh(TypeRepr::HvmPtr);
        // FAST OP
        // if is-num(target) && is-num(v-x) && is-num(v-y)
        //   self.oper += 4
        //   nxt = Ptr::new(NUM, self.op(self.op(val(target), val(v-x)), val(v-y)))
        // else
        //   let opx = self.alloc(1)
        //   let opy = self.alloc(1)
        //   self.heap.set(opx, P2, Ptr::new(OP2, opy))
        //   self.link(Ptr::new(VR1, opx), v-x)
        //   self.link(Ptr::new(VR1, opy), v-y)
        //   self.link(Ptr::new(OP2, opx), target)
        //   nxt = Ptr::new(VR2, opy)
        lowering.stmts.push(Stmt::Instr(Instr::If {
          cond: Instr::from(target.clone())
            .is_num()
            .and(v_x.clone().is_num())
            .and(v_y.clone().is_num())
            .into(),
          then: lowering.fork_on(|lowering| {
            lowering.assign(Prop::Oper, Instr::from(Prop::Oper).add(Instr::Int(4)));
            lowering.assign(
              Prop::Var(nxt.clone()),
              Instr::new_ptr(Const::NUM, Instr::Op {
                lhs: Instr::Op {
                  lhs: Instr::from(target.clone()).val().into(),
                  rhs: v_x.clone().val().into(),
                }
                .into(),
                rhs: v_y.clone().val().into(),
              }),
            );
          }),
          otherwise: lowering.fork_on(|lowering| {
            let opx = lowering.define_fresh(Instr::Alloc { size: 1 });
            let opy = lowering.define_fresh(Instr::Alloc { size: 1 });

            // self.heap.set(opx, P2, Ptr::new(OP2, opy))
            lowering.stmts.push(SetHeap {
              idx: Instr::from(opx.clone()).into(),
              port: Instr::from(Const::P2).into(),
              value: Instr::new_ptr(Const::OP2, Instr::from(opy.clone())).into(),
            });

            // self.link(Ptr::new(VR1, opx), v-x)
            lowering.stmts.push(
              Instr::new_ptr(Const::VR1, Instr::from(opx.clone())).link(Instr::from(v_x.clone())),
            );

            // self.link(Ptr::new(VR1, opy), v-y)
            lowering.stmts.push(
              Instr::new_ptr(Const::VR1, Instr::from(opy.clone())).link(Instr::from(v_y.clone())),
            );

            // self.link(Ptr::new(OP2, opx), target)
            lowering.stmts.push(
              Instr::new_ptr(Const::OP2, Instr::from(opx.clone()))
                .link(Instr::from(target.clone())),
            );

            // nxt = Ptr::new(VR2, opy)
            lowering.assign(
              Prop::Var(nxt.clone()),
              Instr::new_ptr(Const::VR2, Instr::from(opy.clone())),
            );
          }),
        }));

        lowering.burn(def, ret, Instr::from(nxt).clone());
        return true;
      }
    }
  }
  false
}

/// {p1 p2} <~ #N
/// ------------- fast copy
/// p1 <~ #N
/// p2 <~ #N
fn fast_copy(lowering: &mut Lowering, def: &Def, ptr: Ptr, target: Instr) -> bool {
  if ptr.is_ctr() && ptr.tag() > run::CT0 {
    let x1 = lowering.declare_fresh(TypeRepr::HvmPtr);
    let x2 = lowering.declare_fresh(TypeRepr::HvmPtr);
    let (p1, p2) = def.node[ptr.val() as usize];

    // FAST COPY
    // if tag(target) = NUM
    //   self.comm += 1
    //   x1 = target
    //   x2 = target
    // else
    //   let lc = self.alloc(1)
    //   x1 = Ptr::new(VR1, lc)
    //   x2 = Ptr::new(VR2, lc)
    //   self.link(Ptr::new(ptr.tag(), lc), target)
    lowering.stmts.push(Stmt::Instr(Instr::If {
      cond: Instr::from(target.clone())
        .tag()
        .eq(Instr::from(Const::NUM))
        .into(),
      then: lowering.fork_on(|lowering| {
        lowering.assign(Prop::Comm, Instr::from(Prop::Comm).add(Instr::Int(1)));
        lowering.assign(Prop::Var(x1.clone()), Instr::from(target.clone()));
        lowering.assign(Prop::Var(x2.clone()), Instr::from(target.clone()));
      }),
      otherwise: lowering.fork_on(|lowering| {
        let lc = lowering.define_fresh(Instr::Alloc { size: 1 });
        lowering.assign(
          Prop::Var(x1.clone()),
          Instr::new_ptr(Const::VR1, lc.clone().into()),
        );
        lowering.assign(
          Prop::Var(x2.clone()),
          Instr::new_ptr(Const::VR2, lc.clone().into()),
        );
        lowering.stmts.push(
          Instr::new_ptr(compile_tag(ptr.tag()), lc.into()).link(Instr::from(target.clone())),
        );
      }),
    }));

    lowering.burn(def, p1, x1.clone().into());
    lowering.burn(def, p2, x2.clone().into());
    return true;
  }
  false
}

/// (p1 p2) <~ (x1 x2)
/// ------------------ fast apply
/// p1 <~ x1
/// p2 <~ x2
fn fast_apply(lowering: &mut Lowering, def: &Def, ptr: Ptr, target: Instr) -> bool {
  if ptr.is_ctr() && ptr.tag() == run::CT0 {
    let (p1, p2) = def.node[ptr.val() as usize];

    let x1 = lowering.declare_fresh(TypeRepr::HvmPtr);
    let x2 = lowering.declare_fresh(TypeRepr::HvmPtr);

    // FAST APPLY
    // if tag(target) = ptr.tag()
    //   self.anni += 1
    //   x1 = heap-get(val(target), P1)
    //   x2 = heap-get(val(target), P2)
    // else
    //   let lc = self.alloc(1)
    //   x1 = Ptr::new(VR1, lc)
    //   x2 = Ptr::new(VR2, lc)
    //   self.link(Ptr::new(ptr.tag(), lc), target)
    lowering.stmts.push(Stmt::Instr(Instr::If {
      cond: Instr::from(target.clone())
        .tag()
        .eq(Instr::from(compile_tag(ptr.tag())))
        .into(),
      then: lowering.fork_on(|lowering| {
        lowering.assign(Prop::Anni, Instr::from(Prop::Anni).add(Instr::Int(1)));
        lowering.assign(Prop::Var(x1.clone()), Instr::GetHeap {
          idx: Instr::from(target.clone()).val().into(),
          port: Instr::from(Const::P1).into(),
        });
        lowering.assign(Prop::Var(x2.clone()), Instr::GetHeap {
          idx: Instr::from(target.clone()).val().into(),
          port: Instr::from(Const::P2).into(),
        });
        lowering
          .stmts
          .push(Stmt::Free(Instr::from(target.clone()).val().into()))
      }),
      otherwise: lowering.fork_on(|lowering| {
        let lc = lowering.define_fresh(Instr::Alloc { size: 1 });
        lowering.assign(
          Prop::Var(x1.clone()),
          Instr::new_ptr(Const::VR1, lc.clone().into()),
        );
        lowering.assign(
          Prop::Var(x2.clone()),
          Instr::new_ptr(Const::VR2, lc.clone().into()),
        );
        lowering.stmts.push(
          Instr::new_ptr(compile_tag(ptr.tag()), lc.into()).link(Instr::from(target.clone())),
        );
      }),
    }));

    lowering.burn(def, p1, x1.clone().into());
    lowering.burn(def, p2, x2.clone().into());
    return true;
  }
  false
}

/// ATOM <~ *
/// --------- fast erase
/// nothing
fn fast_erase(lowering: &mut Lowering, def: &Def, ptr: Ptr, target: Instr) -> bool {
  if ptr.is_num() || ptr.is_era() {
    // FAST ERASE
    lowering.stmts.push(Stmt::Instr(Instr::If {
      cond: Instr::from(target.clone()).is_skp().into(),
      then: lowering.fork_on(|lowering| {
        lowering.assign(Prop::Eras, Instr::from(Prop::Eras).add(Instr::Int(1)));
      }),
      otherwise: lowering.fork_on(|lowering| {
        lowering.make(def, ptr, target.clone());
      }),
    }));
    return true;
  }
  false
}

struct JitLowering {
  /// The function builder context, which is reused across multiple
  /// FunctionBuilder instances.
  builder_context: FunctionBuilderContext,

  /// The main Cranelift context, which holds the state for codegen. Cranelift
  /// separates this from `Module` to allow for parallel compilation, with a
  /// context per thread, though this isn't in the simple demo here.
  ctx: codegen::Context,

  /// The data description, which is to data objects what `ctx` is to functions.
  data_description: DataDescription,

  /// The module, with the jit backend, which manages the JIT'd
  /// functions.
  module: JITModule,
}

impl Default for JitLowering {
  fn default() -> Self {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
      panic!("host machine is not supported: {}", msg);
    });
    let isa = isa_builder
      .finish(settings::Flags::new(flag_builder))
      .unwrap();
    let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

    let module = JITModule::new(builder);
    Self {
      builder_context: FunctionBuilderContext::new(),
      ctx: module.make_context(),
      data_description: DataDescription::new(),
      module,
    }
  }
}

impl JitLowering {
  /// Create a zero-initialized data section.
  fn create_data(&mut self, name: &str, contents: Vec<u8>) -> Result<&[u8], String> {
    // The steps here are analogous to `compile`, except that data is much
    // simpler than functions.
    self.data_description.define(contents.into_boxed_slice());
    let id = self
      .module
      .declare_data(name, Linkage::Export, true, false)
      .map_err(|e| e.to_string())?;

    self
      .module
      .define_data(id, &self.data_description)
      .map_err(|e| e.to_string())?;
    self.data_description.clear();
    self.module.finalize_definitions().unwrap();
    let buffer = self.module.get_finalized_data(id);
    // TODO: Can we move the unsafe into cranelift?
    Ok(unsafe { core::slice::from_raw_parts(buffer.0, buffer.1) })
  }
}

/// The state of the program being lowered.
struct LoweringProgram {
  constants: HashMap<String, i64>,
}

impl LoweringProgram {
  pub fn lower_constant(&self, constant: Const) -> i64 {
    match constant {
      Const::F(name) => self
        .constants
        .get(&name)
        .expect("Can't find value for const")
        .clone(),
      Const::P1 => crate::run::P1 as i64,
      Const::P2 => crate::run::P2 as i64,
      Const::NULL => crate::run::NULL.0 as i64,
      Const::ROOT => crate::run::ERAS.0 as i64,
      Const::ERAS => crate::run::ERAS.0 as i64,
      Const::VR1 => crate::run::VR1 as i64,
      Const::VR2 => crate::run::VR2 as i64,
      Const::RD1 => crate::run::RD1 as i64,
      Const::RD2 => crate::run::RD2 as i64,
      Const::REF => crate::run::REF as i64,
      Const::ERA => crate::run::ERA as i64,
      Const::NUM => crate::run::NUM as i64,
      Const::OP1 => crate::run::OP1 as i64,
      Const::OP2 => crate::run::OP2 as i64,
      Const::MAT => crate::run::MAT as i64,
      Const::CT0 => crate::run::CT0 as i64,
      Const::CT1 => crate::run::CT1 as i64,
      Const::CT2 => crate::run::CT2 as i64,
      Const::CT3 => crate::run::CT3 as i64,
      Const::CT4 => crate::run::CT4 as i64,
      Const::CT5 => crate::run::CT5 as i64,
      Const::USE => crate::run::USE as i64,
      Const::ADD => crate::run::ADD as i64,
      Const::SUB => crate::run::SUB as i64,
      Const::MUL => crate::run::MUL as i64,
      Const::DIV => crate::run::DIV as i64,
      Const::MOD => crate::run::MOD as i64,
      Const::EQ => crate::run::EQ as i64,
      Const::NE => crate::run::NE as i64,
      Const::LT => crate::run::LT as i64,
      Const::GT => crate::run::GT as i64,
      Const::AND => crate::run::AND as i64,
      Const::OR => crate::run::OR as i64,
      Const::XOR => crate::run::XOR as i64,
      Const::NOT => crate::run::NOT as i64,
      Const::RSH => crate::run::RSH as i64,
      Const::LSH => crate::run::LSH as i64,
    }
  }
}

macro_rules! declare_external_function {
  ($name:ident ($($argn:ident : $args:expr),*) -> $ret:expr) => {
    pub fn $name (&mut self, $( $argn : Value ),*) -> Value {
      let mut sig = self.module.make_signature();
      $( sig.params.push(AbiParam::new($args)); )*
      sig.returns.push(AbiParam::new($ret));

      let id = self.module.declare_function(stringify!($name), Linkage::Import, &sig).unwrap();
      let local_id = self.module.declare_func_in_func(id, &mut self.builder.func);

      let mut arguments = Vec::new();
      $( arguments.push($argn); )*

      let call = self.builder.ins().call(local_id, &arguments);
      self.builder.inst_results(call)[0]
    }
  };
}

/// A collection of state used for translating from toy-language AST nodes
/// into Cranelift IR.
struct FunctionLowering<'a> {
  int: types::Type,
  builder: FunctionBuilder<'a>,
  variables: HashMap<String, Variable>,
  module: &'a mut JITModule,
}

impl FunctionLowering<'_> {
  declare_external_function! {
    GET_HEAP (idx: types::I32, port: types::I32) -> types::I32
  }

  fn lower_instr(&mut self, program: &LoweringProgram, instr: Instr) -> Value {
    match instr {
      Instr::True => self.builder.ins().iconst(self.int, 0),
      Instr::False => self.builder.ins().iconst(self.int, 0),
      Instr::Int(v) => self.builder.ins().iconst(self.int, v as i64),
      Instr::Const(constant) => {
        let value_constant = program.lower_constant(constant);
        self.builder.ins().iconst(self.int, value_constant)
      }
      Instr::Prop(prop) => todo!(),
      Instr::If {
        cond,
        then,
        otherwise,
      } => todo!(),
      Instr::Not { ins } => todo!(),
      Instr::Bin { op, lhs, rhs } => todo!(),
      Instr::Val { ins } => todo!(),
      Instr::Tag { ins } => todo!(),
      Instr::IsNum { ins } => todo!(),
      Instr::IsSkp { ins } => todo!(),
      Instr::NewPtr { tag, value } => todo!(),
      Instr::Op { lhs, rhs } => todo!(),
      Instr::Alloc { size } => todo!(),
      Instr::GetHeap { idx, port } => todo!(),
    }
  }
}

impl Program {
  #[cfg(not(feature = "hvm_cli_options"))]
  pub fn compile_to_rust_fns(self) -> String {
    String::new()
  }

  #[cfg(feature = "hvm_cli_options")]
  pub fn compile_to_rust_fns(self) -> String {
    use rust_format::Formatter;
    use quote::ToTokens;
    let tokens = self.to_token_stream();
    rust_format::RustFmt::new()
      .format_str(tokens.to_string())
      .unwrap()
  }
}

#[cfg(feature = "hvm_cli_options")]
mod rust_codegen {
  use proc_macro2::{Ident, Span, TokenStream};
  use quote::{format_ident, quote, ToTokens, TokenStreamExt};

  use super::*;

  impl ToTokens for Program {
    fn to_tokens(&self, tokens: &mut TokenStream) {
      let constants = &self.values;
      let functions = &self.functions;
      let cases = functions.iter().map(|function| {
        let name = format_ident!("F_{}", function.name);

        quote! { #name => self.#name(book, ptr, argument), }
      });

      tokens.append_all(quote! {
        use crate::run::*;

        #( #constants )*

        impl Net {
          #( #functions )*

          pub fn call_native(&mut self, book: &Book, ptr: Ptr, argument: Ptr) -> bool {
            match ptr.val() {
              #( #cases )*
              _ => { return false; }
            }
          }
        }
      })
    }
  }

  impl ToTokens for Function {
    fn to_tokens(&self, tokens: &mut TokenStream) {
      let name = format_ident!("F_{}", self.name);
      let body = &self.body;

      tokens.append_all(quote! {
        pub fn #name(&mut self, book: &Book, ptr: Ptr, argument: Ptr) -> bool {
          #( #body )*
        }
      })
    }
  }

  impl ToTokens for TypeRepr {
    fn to_tokens(&self, tokens: &mut TokenStream) {
      tokens.append_all(match self {
        TypeRepr::HvmPtr => quote! { Ptr },
        TypeRepr::Ptr => quote! { usize },
        TypeRepr::USize => quote! { usize },
        TypeRepr::U8 => quote! { u8 },
        TypeRepr::U32 => quote! { u32 },
        TypeRepr::Bool => quote! { bool },
      })
    }
  }

  impl ToTokens for Stmt {
    fn to_tokens(&self, tokens: &mut TokenStream) {
      tokens.append_all(match self {
        Stmt::Let { name, value } => {
          let name = format_ident!("{name}");
          quote! { let #name = #value; }
        }
        Stmt::Val { name, type_repr } => {
          let name = format_ident!("{name}");
          quote! { let #name: #type_repr; }
        }
        Stmt::Assign { name, value } => match name {
          Prop::Anni => quote! { self.anni = #value; },
          Prop::Oper => quote! { self.oper = #value; },
          Prop::Eras => quote! { self.eras = #value; },
          Prop::Comm => quote! { self.comm = #value; },
          Prop::Var(var) => {
            let name = format_ident!("{}", var).to_token_stream();
            quote! { #name = #value; }
          }
        },
        Stmt::Instr(instr) => quote! { #instr; },
        Stmt::Free(value) => quote! { self.free(#value); },
        Stmt::Return(value) => quote! { return #value; },
        Stmt::SetHeap { idx, port, value } => quote! { self.heap.set(#idx, #port, #value); },
        Stmt::Link { lhs, rhs } => quote! { self.link(#lhs, #rhs); },
      })
    }
  }

  impl ToTokens for Instr {
    fn to_tokens(&self, tokens: &mut TokenStream) {
      tokens.append_all(match self {
        Instr::True => quote! { true },
        Instr::False => quote! { true },
        Instr::Int(i) => TokenStream::from_str(&format!("{i}")).unwrap(),
        Instr::Const(Const::F(name)) => format_ident!("F_{}", name).into_token_stream(),
        Instr::Const(Const::P1) => quote! { crate::run::P1 },
        Instr::Const(Const::P2) => quote! { crate::run::P2 },
        Instr::Const(Const::NULL) => quote! { crate::run::NULL },
        Instr::Const(Const::ROOT) => quote! { crate::run::ERAS },
        Instr::Const(Const::ERAS) => quote! { crate::run::ERAS },
        Instr::Const(Const::VR1) => quote! { crate::run::VR1 },
        Instr::Const(Const::VR2) => quote! { crate::run::VR2 },
        Instr::Const(Const::RD1) => quote! { crate::run::RD1 },
        Instr::Const(Const::RD2) => quote! { crate::run::RD2 },
        Instr::Const(Const::REF) => quote! { crate::run::REF },
        Instr::Const(Const::ERA) => quote! { crate::run::ERA },
        Instr::Const(Const::NUM) => quote! { crate::run::NUM },
        Instr::Const(Const::OP1) => quote! { crate::run::OP1 },
        Instr::Const(Const::OP2) => quote! { crate::run::OP2 },
        Instr::Const(Const::MAT) => quote! { crate::run::MAT },
        Instr::Const(Const::CT0) => quote! { crate::run::CT0 },
        Instr::Const(Const::CT1) => quote! { crate::run::CT1 },
        Instr::Const(Const::CT2) => quote! { crate::run::CT2 },
        Instr::Const(Const::CT3) => quote! { crate::run::CT3 },
        Instr::Const(Const::CT4) => quote! { crate::run::CT4 },
        Instr::Const(Const::CT5) => quote! { crate::run::CT5 },
        Instr::Const(Const::USE) => quote! { crate::run::USE },
        Instr::Const(Const::ADD) => quote! { crate::run::ADD },
        Instr::Const(Const::SUB) => quote! { crate::run::SUB },
        Instr::Const(Const::MUL) => quote! { crate::run::MUL },
        Instr::Const(Const::DIV) => quote! { crate::run::DIV },
        Instr::Const(Const::MOD) => quote! { crate::run::MOD },
        Instr::Const(Const::EQ) => quote! { crate::run::EQ },
        Instr::Const(Const::NE) => quote! { crate::run::NE },
        Instr::Const(Const::LT) => quote! { crate::run::LT },
        Instr::Const(Const::GT) => quote! { crate::run::GT },
        Instr::Const(Const::AND) => quote! { crate::run::AND },
        Instr::Const(Const::OR) => quote! { crate::run::OR },
        Instr::Const(Const::XOR) => quote! { crate::run::XOR },
        Instr::Const(Const::NOT) => quote! { crate::run::NOT },
        Instr::Const(Const::RSH) => quote! { crate::run::RSH },
        Instr::Const(Const::LSH) => quote! { crate::run::LSH },
        Instr::Prop(Prop::Var(name)) => format_ident!("{}", name).into_token_stream(),
        Instr::Prop(Prop::Anni) => quote! { self.anni },
        Instr::Prop(Prop::Comm) => quote! { self.comm },
        Instr::Prop(Prop::Eras) => quote! { self.eras },
        Instr::Prop(Prop::Oper) => quote! { self.oper },
        Instr::Not { ins } => quote! { !#ins },
        Instr::Val { ins } => quote! { #ins.val() },
        Instr::Tag { ins } => quote! { #ins.tag() },
        Instr::IsNum { ins } => quote! { #ins.is_num() },
        Instr::IsSkp { ins } => quote! { #ins.is_skp() },
        Instr::NewPtr { tag, value } => quote! { Ptr::new(#tag, #value) },
        Instr::Op { lhs, rhs } => quote! { self.op(#lhs, #rhs) },
        Instr::Alloc { size } => quote! { self.alloc(#size) },
        Instr::GetHeap { idx, port } => quote! { self.heap.get(#idx, #port) },
        Instr::Bin { op, lhs, rhs } => match op.as_str() {
          "==" => quote! { #lhs == #rhs },
          "!=" => quote! { #lhs != #rhs },
          "&&" => quote! { #lhs && #rhs },
          "+" => quote! { #lhs + #rhs },
          "-" => quote! { #lhs - #rhs },
          _ => panic!(),
        },
        Instr::If {
          cond,
          then,
          otherwise,
        } => quote! {
          if #cond {
            #( #then )*
          } else {
            #( #otherwise )*
          }
        },
      })
    }
  }

  impl ToTokens for Constant {
    fn to_tokens(&self, tokens: &mut TokenStream) {
      let name = format_ident!("F_{}", self.name);
      let value = TokenStream::from_str(&format!("0x{:06x}", self.value)).unwrap();

      tokens.append_all(quote! {
        pub const #name: u32 = #value;
      })
    }
  }
}

/// Utility functions
impl Instr {
  pub fn is_num(self) -> Instr {
    Instr::IsNum {
      ins: Box::new(self),
    }
  }

  pub fn is_skp(self) -> Instr {
    Instr::IsSkp {
      ins: Box::new(self),
    }
  }

  pub fn val(self) -> Instr {
    Instr::Val {
      ins: Box::new(self),
    }
  }

  pub fn tag(self) -> Instr {
    Instr::Tag {
      ins: Box::new(self),
    }
  }

  pub fn not(self) -> Instr {
    Instr::Not {
      ins: Box::new(self),
    }
  }

  pub fn bin(self, op: &str, rhs: Instr) -> Instr {
    Instr::Bin {
      op: op.to_string(),
      lhs: Box::new(self),
      rhs: Box::new(rhs),
    }
  }

  pub fn eq(self, rhs: Instr) -> Instr {
    self.bin("==", rhs)
  }

  pub fn ne(self, rhs: Instr) -> Instr {
    self.bin("!=", rhs)
  }

  pub fn and(self, rhs: Instr) -> Instr {
    self.bin("&&", rhs)
  }

  pub fn sub(self, rhs: Instr) -> Instr {
    self.bin("-", rhs)
  }

  pub fn add(self, rhs: Instr) -> Instr {
    self.bin("+", rhs)
  }

  pub fn link(self, rhs: Instr) -> Stmt {
    Stmt::Link { lhs: self, rhs }
  }

  pub fn new_ptr(tag: impl Into<Instr>, value: Instr) -> Instr {
    Instr::NewPtr {
      tag: Box::new(tag.into()),
      value: Box::new(value),
    }
  }
}
