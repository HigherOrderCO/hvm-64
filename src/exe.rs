//! JIT Runtime functions. These are used combined with the JIT compiler to achieve fast
//! execution of the program. The JIT compiler will generate calls to these functions
//! to perform operations that are not possible to do in the JIT compiler itself.

use crate::run::{self, Port, Ptr, Tag, Val};

type Net = *mut run::Net;

/// [crate::jit::Expr::Val]
pub unsafe extern "C" fn VAL(ptr: Ptr) -> Val {
  ptr.val()
}

/// [crate::jit::Expr::Tag]
pub unsafe extern "C" fn TAG(ptr: Ptr) -> u32 {
  ptr.tag() as u32
}

/// [crate::jit::Expr::IsNum]
pub unsafe extern "C" fn IS_NUM(ptr: Ptr) -> u32 {
  if ptr.is_num() {
    1
  } else {
    0
  }
}

/// [crate::jit::Expr::IsSkp]
pub unsafe extern "C" fn IS_SKP(ptr: Ptr) -> u32 {
  if ptr.is_skp() {
    1
  } else {
    0
  }
}

/// [crate::jit::Expr::NewPtr]
pub unsafe extern "C" fn NEW_PTR(tag: Tag, val: Val) -> Ptr {
  Ptr::new(tag, val)
}

/// [crate::jit::Expr::Alloc]
pub unsafe extern "C" fn ALLOC(net: Net, size: usize) -> Val {
  let net = &mut *net;
  net.alloc(size)
}

/// [crate::jit::Expr::Op]
pub unsafe extern "C" fn OP(net: Net, a: Val, b: Val) -> Val {
  let net = &mut *net;
  net.op(a, b)
}

/// [crate::jit::Expr::GetHeap]
pub unsafe extern "C" fn GET_HEAP(net: Net, idx: Val, port: Port) -> Ptr {
  let net = &mut *net;
  let heap = net.heap.get(idx, port);
  heap
}

/// [crate::jit::Instr::SetHeap]
pub unsafe extern "C" fn SET_HEAP(net: Net, idx: Val, port: Port, val: Ptr) {
  let net = &mut *net;
  net.heap.set(idx, port, val);
}

/// [crate::jit::Instr::Link]
pub unsafe extern "C" fn LINK(net: Net, a: Ptr, b: Ptr) {
  let net = &mut *net;
  net.link(a, b)
}

/// [crate::jit::Instr::Free]
pub unsafe extern "C" fn FREE(net: Net, val: Val) {
  let net = &mut *net;
  net.free(val)
}

pub unsafe extern "C" fn GET_ANNI(net: Net) -> usize {
  let net = &mut *net;
  net.anni
}

pub unsafe extern "C" fn SET_ANNI(net: Net, val: usize) -> usize {
  let net = &mut *net;
  net.anni = val;
  val
}

pub unsafe extern "C" fn GET_OPER(net: Net) -> usize {
  let net = &mut *net;
  net.oper
}

pub unsafe extern "C" fn SET_OPER(net: Net, val: usize) -> usize {
  let net = &mut *net;
  net.oper = val;
  val
}

pub unsafe extern "C" fn GET_ERAS(net: Net) -> usize {
  let net = &mut *net;
  net.eras
}

pub unsafe extern "C" fn SET_ERAS(net: Net, val: usize) -> usize {
  let net = &mut *net;
  net.eras = val;
  val
}

pub unsafe extern "C" fn GET_COMM(net: Net) -> usize {
  let net = &mut *net;
  net.comm
}

pub unsafe extern "C" fn SET_COMM(net: Net, val: usize) -> usize {
  let net = &mut *net;
  net.comm = val;
  val
}