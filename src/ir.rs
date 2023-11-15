//! This file represents the intermediate representation of the compiler as an imperative
//! language, to be translated into the target language.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeRepr {
  HvmPtr,
  Ptr,
  Usize,
  Bool,
}

/// Constant values
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Con {}

/// Constant values
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Prop {
    Anni,
    Eras,
    Comm
}

/// Represents a single instruction in the IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ins {
  Con(Con),

  /// if cond then else els
  If {
    cond: Box<Ins>,
    then: Box<Ins>,
    els: Box<Ins>,
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
  /// let name = value; next
  Let {
    name: String,
    type_repr: TypeRepr,
    value: Box<Ins>,
    next: Box<Ins>,
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
