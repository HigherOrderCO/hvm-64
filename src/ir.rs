//! This file represents the intermediate representation of the compiler as an imperative
//! language, to be translated into the target language.

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
pub enum Con {
  F(String),
  P1,
  // [crate::run::P1]
  P2,
  // [crate::run::P2]
  NULL,
  // [crate::run::NULL]
  ROOT,
  // [crate::run::ROOT]
  ERAS,
  // [crate::run::ERAS]
  VR1,
  // [crate::run::VR1]
  VR2,
  // [crate::run::VR2]
  RD1,
  // [crate::run::RD1]
  RD2,
  // [crate::run::RD2]
  REF,
  // [crate::run::REF]
  ERA,
  // [crate::run::ERA]
  NUM,
  // [crate::run::NUM]
  OP1,
  // [crate::run::OP1]
  OP2,
  // [crate::run::OP2]
  MAT,
  // [crate::run::MAT]
  CT0,
  // [crate::run::CT0]
  CT1,
  // [crate::run::CT1]
  CT2,
  // [crate::run::CT2]
  CT3,
  // [crate::run::CT3]
  CT4,
  // [crate::run::CT4]
  CT5,
  // [crate::run::CT5]
  USE,
  // [crate::run::USE]
  ADD,
  // [crate::run::ADD]
  SUB,
  // [crate::run::SUB]
  MUL,
  // [crate::run::MUL]
  DIV,
  // [crate::run::DIV]
  MOD,
  // [crate::run::MOD]
  EQ,
  // [crate::run::EQ]
  NE,
  // [crate::run::NE]
  LT,
  // [crate::run::LT]
  GT,
  // [crate::run::GT]
  AND,
  // [crate::run::AND]
  OR,
  // [crate::run::OR]
  XOR,
  // [crate::run::XOR]
  NOT,
  // [crate::run::NOT]
  RSH,
  // [crate::run::RSH]
  LSH,  // [crate::run::LSH]
}

/// Constant values
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Prop {
  Anni,
  Eras,
  Comm,
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
    value: Ins,
  },
  Ins(Ins),
  Return(Ins),
}

impl From<Con> for Ins {
  fn from(value: Con) -> Self {
    Ins::Con(value)
  }
}

/// Represents a single instruction in the IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ins {
  True,
  False,
  Int(u32),
  Hex(u32),
  Con(Con),

  /// SPECIAL TERM: Call
  /// It's used to call generated functions
  Call {
    name: String,
    args: Vec<Ins>,
  },

  /// if cond then else els
  If {
    cond: Box<Ins>,
    then: Vec<Stmt>,
    otherwise: Vec<Stmt>,
  },
  /// name
  Var {
    name: String,
  },
  /// !ins
  Not {
    ins: Box<Ins>,
  },
  /// ptr += value
  Inc {
    ptr: Box<Ins>,
    value: Box<Ins>,
  },
  /// lhs op rhs
  Bin {
    op: String,
    lhs: Box<Ins>,
    rhs: Box<Ins>,
  },

  // VALUE FUNCTIONS:
  // These are the functions that are exposed to the user.
  /// ins.val()
  Val {
    ins: Box<Ins>,
  },
  /// ins.tag()
  Tag {
    ins: Box<Ins>,
  },
  /// ins.is_num()
  IsNum {
    ins: Box<Ins>,
  },
  /// ins.is_skp()
  IsSkp {
    ins: Box<Ins>,
  },
  /// Ptr::new(tag, value)
  NewPtr {
    tag: Box<Ins>,
    value: Box<Ins>,
  },

  // FUNCTIONS:
  // These are the functions that are internal to the IR.
  /// self.ops(lhs, op, rhs)
  Op {
    lhs: Box<Ins>,
    op: Box<Ins>,
    rhs: Box<Ins>,
  },
  /// self.link(lhs, rhs)
  Link {
    lhs: Box<Ins>,
    rhs: Box<Ins>,
  },
  /// self.alloc(n)
  Alloc {
    size: usize,
  },
  /// self.heap.get(idx, port)
  GetHeap {
    idx: Box<Ins>,
    port: Box<Ins>,
  },
  /// self.heap.set(idx, port, value)
  SetHeap {
    idx: Box<Ins>,
    port: Box<Ins>,
    value: Box<Ins>,
  },
  /// self.prop
  Prop(Prop),
}
