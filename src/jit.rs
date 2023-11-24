//! JIT compilation of Cranelift IR.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::str::FromStr;
use std::sync::Arc;

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, Linkage, Module};
use Instr::SetHeap;

use crate::ast;
use crate::run::{self, Book, CallNative, Def, Ptr, Val};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeRepr {
  HvmPtr,
  Ptr,
  USize,
  U8,
  U32,
  Bool,
  Unit,
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

/// Property or variable, it's self.prop, like self.anni, self.oper, etc.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Var {
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
  pub body: Vec<Instr>,
}

/// Represents a single statement in the IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instr {
  /// Declares a variable and assigns it's value.
  Let {
    name: String,
    value: Expr,
  },
  /// Declares a variable without declaring it's value, it's useful for delaying the initialization.
  Val {
    name: String,
    type_repr: TypeRepr,
  },
  /// Assigns a value to a property.
  Assign {
    name: Var,
    value: Expr,
  },
  Expr(Expr),
  Free(Expr),
  Return(Expr),
  /// self.heap.set(idx, port, value)
  SetHeap {
    idx: Expr,
    port: Expr,
    value: Expr,
  },
  /// self.link(lhs, rhs)
  Link {
    lhs: Expr,
    rhs: Expr,
  },
}

/// Represents a single instruction in the IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
  True,
  False,
  Int(u32),
  Const(Const),
  Var(Var),

  /// SPECIAL TERM: Call
  /// if cond then else els
  If {
    cond: Box<Expr>,
    then: Vec<Instr>,
    otherwise: Vec<Instr>,
  },
  /// lhs op rhs
  Bin {
    op: String,
    lhs: Box<Expr>,
    rhs: Box<Expr>,
  },

  // VALUE FUNCTIONS:
  // These are the functions that are exposed to the user.
  /// ins.val()
  Val {
    expr: Box<Expr>,
  },
  /// ins.tag()
  Tag {
    expr: Box<Expr>,
  },
  /// ins.is_num()
  IsNum {
    expr: Box<Expr>,
  },
  /// ins.is_skp()
  IsSkp {
    expr: Box<Expr>,
  },
  /// Ptr::new(tag, value)
  NewPtr {
    tag: Box<Expr>,
    value: Box<Expr>,
  },

  // FUNCTIONS:
  // These are the functions that are internal to the IR.
  /// self.ops(lhs, op, rhs)
  Op {
    lhs: Box<Expr>,
    rhs: Box<Expr>,
  },
  /// self.alloc(n)
  Alloc {
    size: usize,
  },
  /// self.heap.get(idx, port)
  GetHeap {
    idx: Box<Expr>,
    port: Box<Expr>,
  },
}

impl From<Const> for Expr {
  fn from(value: Const) -> Self {
    Expr::Const(value)
  }
}

impl From<String> for Expr {
  fn from(value: String) -> Self {
    Expr::Var(Var::Var(value))
  }
}

impl From<Var> for Expr {
  fn from(value: Var) -> Self {
    Expr::Var(value)
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
    target: Expr::from("argument".to_string()),
    vars: Rc::new(RefCell::new(HashMap::new())),
    stmts: vec![],
  };

  lowering.call(fid);
  lowering.stmts.push(Instr::Return(Expr::True));

  Function {
    name: ast::val_to_name(fid),
    body: lowering.stmts,
  }
}

fn assert_is_atom(ptr: Ptr) -> Expr {
  if ptr.is_ref() {
    Expr::NewPtr {
      tag: Expr::from(Const::REF).into(),
      value: Expr::from(Const::F(ast::val_to_name(ptr.val()))).into(),
    }
  } else {
    Expr::NewPtr {
      tag: Expr::from(compile_tag(ptr.tag())).into(),
      value: Expr::Int(ptr.val()).into(),
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
  vars: Rc<RefCell<HashMap<Ptr, Expr>>>,
  target: Expr,
  stmts: Vec<Instr>,
}

impl Lowering<'_> {
  /// returns a fresh variable: 'v<NUM>'
  fn fresh(&mut self) -> Expr {
    let name = format!("v{}", self.newx.get());
    self.newx.set(self.newx.get() + 1);
    Expr::Var(Var::Var(name))
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
      self.stmts.push(Instr::Let {
        name: ref_name.clone(),
        value: assert_is_atom(rf),
      });
      self.burn(def, rx, ref_name.into());
    }
  }

  /// Declares a variable without name
  fn declare_fresh(&mut self, type_repr: TypeRepr) -> String {
    let name = self.fresh_name();
    self.stmts.push(Instr::Val {
      name: name.clone(),
      type_repr,
    });
    name
  }

  /// Declares a variable
  fn declare(&mut self, name: String, type_repr: TypeRepr) -> String {
    self.stmts.push(Instr::Val {
      name: name.clone(),
      type_repr,
    });
    name
  }

  /// Defines a variable without name
  fn define_fresh(&mut self, value: Expr) -> String {
    let name = self.fresh_name();
    self.stmts.push(Instr::Let {
      name: name.clone(),
      value,
    });
    name
  }

  /// Defines a variable
  fn define(&mut self, name: String, value: Expr) -> String {
    self.stmts.push(Instr::Let {
      name: name.clone(),
      value,
    });
    name
  }

  /// Assigns a value to a property
  fn assign(&mut self, prop: Var, value: Expr) {
    self.stmts.push(Instr::Assign { name: prop, value });
  }

  /// Fork returning Self
  fn fork(&self) -> Self {
    let mut new_fork = self.clone();
    new_fork.stmts = vec![];
    new_fork
  }

  /// Fork returning vec of statements
  fn fork_on(&self, f: impl FnOnce(&mut Self)) -> Vec<Instr> {
    let mut fork = self.fork();
    f(&mut fork);
    fork.stmts
  }

  /// Generates code
  fn make(&mut self, def: &Def, ptr: Ptr, target: Expr) {
    if ptr.is_nod() {
      let lc = self.define_fresh(Expr::Alloc { size: 1 });
      let (p1, p2) = def.node[ptr.val() as usize];
      self.make(def, p1, Expr::new_ptr(Const::VR1, lc.clone().into()));
      self.make(def, p2, Expr::new_ptr(Const::VR2, lc.clone().into()));
      self
        .stmts
        .push(Expr::new_ptr(compile_tag(ptr.tag()), lc.into()).link(target.clone()));
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
  fn get(&self, def: &Def, ptr: Ptr) -> Option<Expr> {
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
  fn burn(&mut self, def: &Def, ptr: Ptr, target: Expr) {
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
fn fast_match(lowering: &mut Lowering, def: &Def, ptr: Ptr, target: Expr) -> bool {
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
        lowering.stmts.push(Instr::Expr(Expr::If {
          cond: Expr::from(target.clone())
            .eq(Expr::from(Const::CT0))
            .and(
              Expr::GetHeap {
                idx: Expr::from(target.clone()).val().into(),
                port: Expr::from(Const::P1).into(),
              }
              .is_num(),
            )
            .into(),
          then: lowering.fork_on(|lowering| {
            // self.anni += 2
            lowering.assign(Var::Anni, Expr::from(Var::Anni).add(Expr::Int(2)));

            // self.oper += 1
            lowering.assign(Var::Oper, Expr::from(Var::Anni).add(Expr::Int(1)));

            // let num = self.heap.get(target.val(), P1)
            let num = lowering.define_fresh(Expr::GetHeap {
              idx: Expr::from(target.clone()).val().into(),
              port: Expr::from(Const::P1).into(),
            });

            // let res = self.heap.get(target.val(), P2)
            let res = lowering.define_fresh(Expr::GetHeap {
              idx: Expr::from(target.clone()).val().into(),
              port: Expr::from(Const::P2).into(),
            });

            // if num.val() == 0
            //   self.free(target.val())
            //   c_z = res
            //   c_s = ERAS
            // else
            //   self.heap.set(target.val(), P1, Ptr::new(NUM, num.val() - 1))
            //   c_z = ERAS
            //   c_s = target
            lowering.stmts.push(Instr::Expr(Expr::If {
              cond: Expr::from(num.clone()).eq(Expr::Int(0)).into(),
              then: lowering.fork_on(|lowering| {
                lowering
                  .stmts
                  .push(Instr::Free(Expr::from(target.clone()).val()));
                lowering.assign(Var::Var(c_z.clone()), Expr::from(res.clone()));
                lowering.assign(Var::Var(c_s.clone()), Expr::from(Const::ERAS));
              }),
              otherwise: lowering.fork_on(|lowering| {
                lowering.stmts.push(Instr::SetHeap {
                  idx: Expr::from(target.clone()).val().into(),
                  port: Expr::from(Const::P1).into(),
                  value: Expr::new_ptr(Expr::from(Const::NUM), Expr::from(num).sub(Expr::Int(1)))
                    .into(),
                });
                lowering.assign(Var::Var(c_z.clone()), Expr::from(Const::ERAS));
                lowering.assign(Var::Var(c_s.clone()), Expr::from(target.clone()));
              }),
            }))
          }),
          otherwise: lowering.fork_on(|lowering| {
            let lam = lowering.define_fresh(Expr::Alloc { size: 1 });
            let mat = lowering.define_fresh(Expr::Alloc { size: 1 });
            let cse = lowering.define_fresh(Expr::Alloc { size: 1 });
            lowering.stmts.push(SetHeap {
              // self.heap.set(lam, P1, Ptr::new(MAT, mat));
              idx: Expr::from(lam.clone()).into(),
              port: Expr::from(Const::P1).into(),
              value: Expr::new_ptr(Const::MAT, Expr::from(mat.clone())).into(),
            });
            lowering.stmts.push(SetHeap {
              // self.heap.set(lam, P2, Ptr::new(VR2, mat));
              idx: Expr::from(lam.clone()).into(),
              port: Expr::from(Const::P2).into(),
              value: Expr::new_ptr(Const::VR2, Expr::from(mat.clone())).into(),
            });
            lowering.stmts.push(SetHeap {
              // self.heap.set(mat, P1, Ptr::new(CT0, cse));
              idx: Expr::from(mat.clone()),
              port: Expr::from(Const::P1),
              value: Expr::new_ptr(Const::CT0, Expr::from(cse.clone())),
            });
            lowering.stmts.push(SetHeap {
              // self.heap.set(mat, P2, Ptr::new(VR2, cse));
              idx: Expr::from(mat.clone()),
              port: Expr::from(Const::P2),
              value: Expr::new_ptr(Const::VR2, Expr::from(lam.clone())),
            });
            lowering.stmts.push(
              Expr::new_ptr(Const::CT0, Expr::from(lam.clone())).link(Expr::from(target.clone())),
            );
            lowering.assign(
              Var::Var(c_z.clone()),
              Expr::new_ptr(Const::VR1, Expr::from(cse.clone())),
            );
            lowering.assign(
              Var::Var(c_s.clone()),
              Expr::new_ptr(Const::VR2, Expr::from(cse.clone())),
            );
          }),
        }));
        lowering.burn(def, ifz, Expr::from(c_z).clone());
        lowering.burn(def, ifs, Expr::from(c_s).clone());
        return true;
      }
    }
  }
  false
}

// <x <y r>> ~ #N
// --------------------- fast op
// r <~ #(op(op(N,x),y))
fn fast_op(lowering: &mut Lowering, def: &Def, ptr: Ptr, target: Expr) -> bool {
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
        lowering.stmts.push(Instr::Expr(Expr::If {
          cond: Expr::from(target.clone())
            .is_num()
            .and(v_x.clone().is_num())
            .and(v_y.clone().is_num())
            .into(),
          then: lowering.fork_on(|lowering| {
            lowering.assign(Var::Oper, Expr::from(Var::Oper).add(Expr::Int(4)));
            lowering.assign(
              Var::Var(nxt.clone()),
              Expr::new_ptr(Const::NUM, Expr::Op {
                lhs: Expr::Op {
                  lhs: Expr::from(target.clone()).val().into(),
                  rhs: v_x.clone().val().into(),
                }
                .into(),
                rhs: v_y.clone().val().into(),
              }),
            );
          }),
          otherwise: lowering.fork_on(|lowering| {
            let opx = lowering.define_fresh(Expr::Alloc { size: 1 });
            let opy = lowering.define_fresh(Expr::Alloc { size: 1 });

            // self.heap.set(opx, P2, Ptr::new(OP2, opy))
            lowering.stmts.push(SetHeap {
              idx: Expr::from(opx.clone()).into(),
              port: Expr::from(Const::P2).into(),
              value: Expr::new_ptr(Const::OP2, Expr::from(opy.clone())).into(),
            });

            // self.link(Ptr::new(VR1, opx), v-x)
            lowering.stmts.push(
              Expr::new_ptr(Const::VR1, Expr::from(opx.clone())).link(Expr::from(v_x.clone())),
            );

            // self.link(Ptr::new(VR1, opy), v-y)
            lowering.stmts.push(
              Expr::new_ptr(Const::VR1, Expr::from(opy.clone())).link(Expr::from(v_y.clone())),
            );

            // self.link(Ptr::new(OP2, opx), target)
            lowering.stmts.push(
              Expr::new_ptr(Const::OP2, Expr::from(opx.clone())).link(Expr::from(target.clone())),
            );

            // nxt = Ptr::new(VR2, opy)
            lowering.assign(
              Var::Var(nxt.clone()),
              Expr::new_ptr(Const::VR2, Expr::from(opy.clone())),
            );
          }),
        }));

        lowering.burn(def, ret, Expr::from(nxt).clone());
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
fn fast_copy(lowering: &mut Lowering, def: &Def, ptr: Ptr, target: Expr) -> bool {
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
    lowering.stmts.push(Instr::Expr(Expr::If {
      cond: Expr::from(target.clone())
        .tag()
        .eq(Expr::from(Const::NUM))
        .into(),
      then: lowering.fork_on(|lowering| {
        lowering.assign(Var::Comm, Expr::from(Var::Comm).add(Expr::Int(1)));
        lowering.assign(Var::Var(x1.clone()), Expr::from(target.clone()));
        lowering.assign(Var::Var(x2.clone()), Expr::from(target.clone()));
      }),
      otherwise: lowering.fork_on(|lowering| {
        let lc = lowering.define_fresh(Expr::Alloc { size: 1 });
        lowering.assign(
          Var::Var(x1.clone()),
          Expr::new_ptr(Const::VR1, lc.clone().into()),
        );
        lowering.assign(
          Var::Var(x2.clone()),
          Expr::new_ptr(Const::VR2, lc.clone().into()),
        );
        lowering
          .stmts
          .push(Expr::new_ptr(compile_tag(ptr.tag()), lc.into()).link(Expr::from(target.clone())));
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
fn fast_apply(lowering: &mut Lowering, def: &Def, ptr: Ptr, target: Expr) -> bool {
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
    lowering.stmts.push(Instr::Expr(Expr::If {
      cond: Expr::from(target.clone())
        .tag()
        .eq(Expr::from(compile_tag(ptr.tag())))
        .into(),
      then: lowering.fork_on(|lowering| {
        lowering.assign(Var::Anni, Expr::from(Var::Anni).add(Expr::Int(1)));
        lowering.assign(Var::Var(x1.clone()), Expr::GetHeap {
          idx: Expr::from(target.clone()).val().into(),
          port: Expr::from(Const::P1).into(),
        });
        lowering.assign(Var::Var(x2.clone()), Expr::GetHeap {
          idx: Expr::from(target.clone()).val().into(),
          port: Expr::from(Const::P2).into(),
        });
        lowering
          .stmts
          .push(Instr::Free(Expr::from(target.clone()).val().into()))
      }),
      otherwise: lowering.fork_on(|lowering| {
        let lc = lowering.define_fresh(Expr::Alloc { size: 1 });
        lowering.assign(
          Var::Var(x1.clone()),
          Expr::new_ptr(Const::VR1, lc.clone().into()),
        );
        lowering.assign(
          Var::Var(x2.clone()),
          Expr::new_ptr(Const::VR2, lc.clone().into()),
        );
        lowering
          .stmts
          .push(Expr::new_ptr(compile_tag(ptr.tag()), lc.into()).link(Expr::from(target.clone())));
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
fn fast_erase(lowering: &mut Lowering, def: &Def, ptr: Ptr, target: Expr) -> bool {
  if ptr.is_num() || ptr.is_era() {
    // FAST ERASE
    lowering.stmts.push(Instr::Expr(Expr::If {
      cond: Expr::from(target.clone()).is_skp().into(),
      then: lowering.fork_on(|lowering| {
        lowering.assign(Var::Eras, Expr::from(Var::Eras).add(Expr::Int(1)));
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

type JitNet = u64;
type JitBook = u64;
type JitPtr = u32;
type JitFunction = unsafe extern "C" fn(JitNet, JitBook, JitPtr, JitPtr) -> u8;

impl Program {
  /// Compile the program into a JIT function.
  pub fn compile_function(&self, function: Function) -> JitFunction {
    let mut lowering = JitLowering::default();
    let function = lowering.translate(self, function);
    unsafe { std::mem::transmute(function) }
  }

  pub fn compile_program(&self) -> CallNative {
    let mut functions = HashMap::new();

    for function in &self.functions {
      let Some(value) = self.values.iter().find(|i| i.name == function.name) else {
        panic!("Can't find value for function");
      };

      functions.insert(value.value, self.compile_function(function.clone()));
    }

    Arc::new(move |net, book, ptr, x| match functions.get(&ptr.val()) {
      Some(function) => unsafe {
        let net: u64 = std::mem::transmute(net);
        let book: u64 = std::mem::transmute(book);
        let ptr: u32 = std::mem::transmute(ptr);
        let x: u32 = std::mem::transmute(x);
        (function)(net, book, ptr, x) == 1
      },
      None => false,
    })
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

  fn translate(&mut self, program: &Program, function: Function) -> *const u8 {
    let signature = &mut self.ctx.func.signature;
    signature.params.push(AbiParam::new(types::I64)); // net
    signature.params.push(AbiParam::new(types::I64)); // book
    signature.params.push(AbiParam::new(types::I32)); // ptr
    signature.params.push(AbiParam::new(types::I32)); // argument

    let mut lowering = FunctionLowering {
      program,
      builder: FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context),
      variables: HashMap::new(),
      module: &mut self.module,
      index: 0,
    };

    let environment = lowering.declare_variable(types::I64, "environment");
    let book = lowering.declare_variable(types::I64, "book");
    let ptr = lowering.declare_variable(types::I32, "ptr");
    let argument = lowering.declare_variable(types::I32, "argument");

    lowering.variables.insert("environment".into(), environment);
    lowering.variables.insert("book".into(), book);
    lowering.variables.insert("ptr".into(), ptr);
    lowering.variables.insert("argument".into(), argument);

    let entry_block = lowering.builder.create_block();
    let mut return_value = None;
    lowering
      .builder
      .append_block_params_for_function_params(entry_block);
    lowering.builder.switch_to_block(entry_block);
    lowering.builder.seal_block(entry_block);
    for instr in function.body {
      return_value = lowering.lower_instr(instr);
    }

    if let Some(return_value) = return_value {
      lowering.builder.ins().return_(&[return_value]);
    }

    let function = self
      .module
      .declare_function(&function.name, Linkage::Export, &self.ctx.func.signature)
      .expect("Can't create new function");

    self.module.clear_context(&mut self.ctx);
    self.module.finalize_definitions().unwrap();
    self.module.get_finalized_function(function)
  }
}

impl Program {
  pub fn lower_constant(&self, constant: Const) -> i64 {
    match constant {
      Const::F(name) => {
        let constant = self.values.iter().find(|i| i.name == name);
        constant.expect("Can't find value for const").value as i64
      }
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
  ([$($name:ident ($($argn:ident : $args:expr),*) -> $ret:expr),*]) => {
    $(declare_external_function!($name ($($argn : $args),*) -> $ret);)*
  };
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
struct FunctionLowering<'program, 'jit> {
  program: &'program Program,
  builder: FunctionBuilder<'jit>,
  variables: HashMap<String, Variable>,
  module: &'jit mut JITModule,
  index: usize,
}

impl FunctionLowering<'_, '_> {
  declare_external_function!([
    VAL (val: types::I32) -> types::I32,
    TAG (val: types::I32) -> types::I32,
    IS_NUM (val: types::I32) -> types::I8,
    IS_SKP (val: types::I32) -> types::I8,
    NEW_PTR (tag: types::I8, value: types::I32) -> types::I32,
    ALLOC (net: types::I64, size: types::I64) -> types::I32,
    OP (net: types::I64, lhs: types::I32, rhs: types::I32) -> types::I32,
    LINK (net: types::I64, lhs: types::I32, rhs: types::I32) -> types::I8,
    FREE (net: types::I64, idx: types::I32) -> types::I8,
    GET_HEAP (net: types::I64, idx: types::I32, port: types::I32) -> types::I32,
    SET_HEAP (net: types::I64, idx: types::I32, port: types::I32, value: types::I32) -> types::I8,
    GET_ANNI (net: types::I64) -> types::I64,
    SET_ANNI (net: types::I64, val: types::I64) -> types::I64,
    GET_OPER (net: types::I64) -> types::I64,
    SET_OPER (net: types::I64, val: types::I64) -> types::I64,
    GET_ERAS (net: types::I64) -> types::I64,
    SET_ERAS (net: types::I64, val: types::I64) -> types::I64,
    GET_COMM (net: types::I64) -> types::I64,
    SET_COMM (net: types::I64, val: types::I64) -> types::I64
  ]);

  fn declare_variable(&mut self, int: types::Type, name: &str) -> Variable {
    let var = Variable::new(self.index);
    if !self.variables.contains_key(name) {
      self.variables.insert(name.into(), var);
      self.builder.declare_var(var, int);
      self.index += 1;
    }
    var
  }

  fn get_environment(&mut self) -> Value {
    let variable = self.variables.get("environment").unwrap();
    self.builder.use_var(*variable)
  }

  fn lower_var(&mut self, prop: Var) -> Value {
    let environment = self.get_environment();
    match prop {
      Var::Anni => self.GET_ANNI(environment),
      Var::Oper => self.GET_OPER(environment),
      Var::Eras => self.GET_ERAS(environment),
      Var::Comm => self.GET_COMM(environment),
      Var::Var(var) => {
        let variable = self.variables.get(&var).unwrap();
        self.builder.use_var(*variable)
      }
    }
  }

  fn lower_instr(&mut self, instr: Instr) -> Option<Value> {
    match instr {
      Instr::Free(target) => {
        let net = self.get_environment();
        let target = self.lower_expr(target);
        self.FREE(net, target);
      }
      Instr::SetHeap { idx, port, value } => {
        let net = self.get_environment();
        let idx = self.lower_expr(idx);
        let port = self.lower_expr(port);
        let value = self.lower_expr(value);
        self.SET_HEAP(net, idx, port, value);
      }
      Instr::Link { lhs, rhs } => {
        let net = self.get_environment();
        let lhs = self.lower_expr(lhs);
        let rhs = self.lower_expr(rhs);
        self.LINK(net, lhs, rhs);
      }

      // STATEMENTS
      Instr::Let { name, value } => {
        let value = self.lower_expr(value);
        let variable = self.declare_variable(types::I32, &name);
        self.builder.def_var(variable, value);
        self.variables.insert(name, variable);
      }
      Instr::Val { name, .. } => {
        let variable = self.declare_variable(types::I32, &name);
        self.variables.insert(name, variable);
      }
      Instr::Assign { name, value } => {
        let value = self.lower_expr(value);
        let environment = self.get_environment();
        match name {
          Var::Anni => self.SET_ANNI(environment, value),
          Var::Oper => self.SET_OPER(environment, value),
          Var::Eras => self.SET_ERAS(environment, value),
          Var::Comm => self.SET_COMM(environment, value),
          Var::Var(name) => {
            let variable = self.variables.get(&name).unwrap();
            self.builder.def_var(*variable, value);
            return None;
          }
        };
      }
      Instr::Expr(expr) => return Some(self.lower_expr(expr)),
      Instr::Return(expr) => {
        let value = self.lower_expr(expr);
        self.builder.ins().return_(&[value]);
      }
    }
    None
  }

  fn lower_expr(&mut self, expr: Expr) -> Value {
    match expr {
      Expr::True => self.builder.ins().iconst(types::I8, 1),
      Expr::False => self.builder.ins().iconst(types::I8, 0),
      Expr::Int(v) => self.builder.ins().iconst(types::I32, v as i64),
      Expr::Const(constant) => {
        let value_constant = self.program.lower_constant(constant);
        self.builder.ins().iconst(types::I8, value_constant)
      }
      Expr::Var(prop) => self.lower_var(prop),
      Expr::Bin { op, lhs, rhs } => {
        let lhs = self.lower_expr(*lhs);
        let rhs = self.lower_expr(*rhs);
        match op.as_str() {
          "+" => self.builder.ins().iadd(lhs, rhs),
          "-" => self.builder.ins().isub(lhs, rhs),
          "*" => self.builder.ins().imul(lhs, rhs),
          "/" => self.builder.ins().sdiv(lhs, rhs),
          "==" => self.builder.ins().icmp(IntCC::Equal, lhs, rhs),
          "!=" => self.builder.ins().icmp(IntCC::NotEqual, lhs, rhs),
          "<" => self.builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs),
          ">" => self.builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs),
          "&&" => self.builder.ins().band(lhs, rhs),
          "||" => self.builder.ins().bor(lhs, rhs),
          _ => panic!("Unknown operator {}", op),
        }
      }
      Expr::Val { expr } => {
        let expr = self.lower_expr(*expr);
        self.VAL(expr)
      }
      Expr::Tag { expr } => {
        let expr = self.lower_expr(*expr);
        self.TAG(expr)
      }
      Expr::IsNum { expr } => {
        let expr = self.lower_expr(*expr);
        self.IS_NUM(expr)
      }
      Expr::IsSkp { expr } => {
        let expr = self.lower_expr(*expr);
        self.IS_SKP(expr)
      }
      Expr::NewPtr { tag, value } => {
        let tag = self.lower_expr(*tag);
        let value = self.lower_expr(*value);
        self.NEW_PTR(tag, value)
      }
      Expr::Op { lhs, rhs } => {
        let net = self.get_environment();
        let lhs = self.lower_expr(*lhs);
        let rhs = self.lower_expr(*rhs);
        self.OP(net, lhs, rhs)
      }
      Expr::Alloc { size } => {
        let net = self.get_environment();
        let size = self.builder.ins().iconst(types::I64, size as i64);
        self.ALLOC(net, size)
      }
      Expr::GetHeap { idx, port } => {
        let net = self.get_environment();
        let idx = self.lower_expr(*idx);
        let port = self.lower_expr(*port);
        self.GET_HEAP(net, idx, port)
      }
      Expr::If {
        cond,
        then,
        otherwise,
      } => {
        let cond = self.lower_expr(*cond);

        let then_bb = self.builder.create_block();
        let otherwise_bb = self.builder.create_block();
        let merge_block = self.builder.create_block();

        // If-else constructs in the language have a return value.
        //
        // In traditional SSA form, this would produce a PHI between
        // the then and else bodies. Cranelift uses block parameters,
        // so set up a parameter in the merge block, and we'll pass
        // the return values to it from the branches.
        self.builder.append_block_param(merge_block, types::I64);
        self
          .builder
          .ins()
          .brif(cond, then_bb, &[], otherwise_bb, &[]);

        self.builder.switch_to_block(then_bb);
        let mut then_value = self.builder.ins().iconst(types::I64, 0);
        for instr in then {
          if let Some(value) = self.lower_instr(instr) {
            then_value = value;
          }
        }
        self.builder.ins().jump(merge_block, &[then_value]);

        self.builder.switch_to_block(otherwise_bb);
        let mut otherwise_value = self.builder.ins().iconst(types::I64, 0);
        for instr in otherwise {
          if let Some(value) = self.lower_instr(instr) {
            otherwise_value = value;
          }
        }
        self.builder.ins().jump(merge_block, &[otherwise_value]);

        self.builder.switch_to_block(merge_block);
        self.builder.seal_block(merge_block);
        self.builder.block_params(merge_block)[0]
      }
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
    use quote::ToTokens;
    use rust_format::Formatter;
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
        TypeRepr::Unit => quote! { () },
      })
    }
  }

  impl ToTokens for Instr {
    fn to_tokens(&self, tokens: &mut TokenStream) {
      tokens.append_all(match self {
        Instr::Let { name, value } => {
          let name = format_ident!("{name}");
          quote! { let #name = #value; }
        }
        Instr::Val { name, type_repr } => {
          let name = format_ident!("{name}");
          quote! { let #name: #type_repr; }
        }
        Instr::Assign { name, value } => match name {
          Var::Anni => quote! { self.anni = #value; },
          Var::Oper => quote! { self.oper = #value; },
          Var::Eras => quote! { self.eras = #value; },
          Var::Comm => quote! { self.comm = #value; },
          Var::Var(var) => {
            let name = format_ident!("{}", var).to_token_stream();
            quote! { #name = #value; }
          }
        },
        Instr::Expr(instr) => quote! { #instr; },
        Instr::Free(value) => quote! { self.free(#value); },
        Instr::Return(value) => quote! { return #value; },
        Instr::SetHeap { idx, port, value } => quote! { self.heap.set(#idx, #port, #value); },
        Instr::Link { lhs, rhs } => quote! { self.link(#lhs, #rhs); },
      })
    }
  }

  impl ToTokens for Expr {
    fn to_tokens(&self, tokens: &mut TokenStream) {
      tokens.append_all(match self {
        Expr::True => quote! { true },
        Expr::False => quote! { true },
        Expr::Int(i) => TokenStream::from_str(&format!("{i}")).unwrap(),
        Expr::Const(Const::F(name)) => format_ident!("F_{}", name).into_token_stream(),
        Expr::Const(Const::P1) => quote! { crate::run::P1 },
        Expr::Const(Const::P2) => quote! { crate::run::P2 },
        Expr::Const(Const::NULL) => quote! { crate::run::NULL },
        Expr::Const(Const::ROOT) => quote! { crate::run::ERAS },
        Expr::Const(Const::ERAS) => quote! { crate::run::ERAS },
        Expr::Const(Const::VR1) => quote! { crate::run::VR1 },
        Expr::Const(Const::VR2) => quote! { crate::run::VR2 },
        Expr::Const(Const::RD1) => quote! { crate::run::RD1 },
        Expr::Const(Const::RD2) => quote! { crate::run::RD2 },
        Expr::Const(Const::REF) => quote! { crate::run::REF },
        Expr::Const(Const::ERA) => quote! { crate::run::ERA },
        Expr::Const(Const::NUM) => quote! { crate::run::NUM },
        Expr::Const(Const::OP1) => quote! { crate::run::OP1 },
        Expr::Const(Const::OP2) => quote! { crate::run::OP2 },
        Expr::Const(Const::MAT) => quote! { crate::run::MAT },
        Expr::Const(Const::CT0) => quote! { crate::run::CT0 },
        Expr::Const(Const::CT1) => quote! { crate::run::CT1 },
        Expr::Const(Const::CT2) => quote! { crate::run::CT2 },
        Expr::Const(Const::CT3) => quote! { crate::run::CT3 },
        Expr::Const(Const::CT4) => quote! { crate::run::CT4 },
        Expr::Const(Const::CT5) => quote! { crate::run::CT5 },
        Expr::Const(Const::USE) => quote! { crate::run::USE },
        Expr::Const(Const::ADD) => quote! { crate::run::ADD },
        Expr::Const(Const::SUB) => quote! { crate::run::SUB },
        Expr::Const(Const::MUL) => quote! { crate::run::MUL },
        Expr::Const(Const::DIV) => quote! { crate::run::DIV },
        Expr::Const(Const::MOD) => quote! { crate::run::MOD },
        Expr::Const(Const::EQ) => quote! { crate::run::EQ },
        Expr::Const(Const::NE) => quote! { crate::run::NE },
        Expr::Const(Const::LT) => quote! { crate::run::LT },
        Expr::Const(Const::GT) => quote! { crate::run::GT },
        Expr::Const(Const::AND) => quote! { crate::run::AND },
        Expr::Const(Const::OR) => quote! { crate::run::OR },
        Expr::Const(Const::XOR) => quote! { crate::run::XOR },
        Expr::Const(Const::NOT) => quote! { crate::run::NOT },
        Expr::Const(Const::RSH) => quote! { crate::run::RSH },
        Expr::Const(Const::LSH) => quote! { crate::run::LSH },
        Expr::Var(Var::Var(name)) => format_ident!("{}", name).into_token_stream(),
        Expr::Var(Var::Anni) => quote! { self.anni },
        Expr::Var(Var::Comm) => quote! { self.comm },
        Expr::Var(Var::Eras) => quote! { self.eras },
        Expr::Var(Var::Oper) => quote! { self.oper },
        Expr::Val { expr: ins } => quote! { #ins.val() },
        Expr::Tag { expr: ins } => quote! { #ins.tag() },
        Expr::IsNum { expr: ins } => quote! { #ins.is_num() },
        Expr::IsSkp { expr: ins } => quote! { #ins.is_skp() },
        Expr::NewPtr { tag, value } => quote! { Ptr::new(#tag, #value) },
        Expr::Op { lhs, rhs } => quote! { self.op(#lhs, #rhs) },
        Expr::Alloc { size } => quote! { self.alloc(#size) },
        Expr::GetHeap { idx, port } => quote! { self.heap.get(#idx, #port) },
        Expr::Bin { op, lhs, rhs } => match op.as_str() {
          "==" => quote! { #lhs == #rhs },
          "!=" => quote! { #lhs != #rhs },
          "&&" => quote! { #lhs && #rhs },
          "+" => quote! { #lhs + #rhs },
          "-" => quote! { #lhs - #rhs },
          _ => panic!(),
        },
        Expr::If {
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
impl Expr {
  pub fn is_num(self) -> Expr {
    Expr::IsNum {
      expr: Box::new(self),
    }
  }

  pub fn is_skp(self) -> Expr {
    Expr::IsSkp {
      expr: Box::new(self),
    }
  }

  pub fn val(self) -> Expr {
    Expr::Val {
      expr: Box::new(self),
    }
  }

  pub fn tag(self) -> Expr {
    Expr::Tag {
      expr: Box::new(self),
    }
  }

  pub fn bin(self, op: &str, rhs: Expr) -> Expr {
    Expr::Bin {
      op: op.to_string(),
      lhs: Box::new(self),
      rhs: Box::new(rhs),
    }
  }

  pub fn eq(self, rhs: Expr) -> Expr {
    self.bin("==", rhs)
  }

  pub fn ne(self, rhs: Expr) -> Expr {
    self.bin("!=", rhs)
  }

  pub fn and(self, rhs: Expr) -> Expr {
    self.bin("&&", rhs)
  }

  pub fn sub(self, rhs: Expr) -> Expr {
    self.bin("-", rhs)
  }

  pub fn add(self, rhs: Expr) -> Expr {
    self.bin("+", rhs)
  }

  pub fn link(self, rhs: Expr) -> Instr {
    Instr::Link { lhs: self, rhs }
  }

  pub fn new_ptr(tag: impl Into<Expr>, value: Expr) -> Expr {
    Expr::NewPtr {
      tag: Box::new(tag.into()),
      value: Box::new(value),
    }
  }
}
