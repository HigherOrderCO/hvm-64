use std::str::FromStr;

use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote, ToTokens, TokenStreamExt};

use crate::ir::{Const, Constant, Function, Instr, Program, Prop, Stmt, TypeRepr};

impl ToTokens for Program {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    let constants = &self.values;
    let functions = &self.functions;

    tokens.append_all(quote! {
      use crate::run::*;

      #( #constants )*
      #( #functions )*

      impl Net {
        pub fn call_native(&mut self, book: &Book, ptr: Ptr, argument: Ptr) -> bool {
          match ptr.val() {
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
      Stmt::Let { name, value } => quote! { let #name = #value; },
      Stmt::Val { name, type_repr } => quote! { let #name: #type_repr; },
      Stmt::Assign { name, value } => match name {
        Prop::Anni => quote! { self.anni = #value; },
        Prop::Oper => quote! { self.oper = #value; },
        Prop::Eras => quote! { self.eras = #value; },
        Prop::Comm => quote! { self.comm = #value; },
        Prop::Var(var) => format_ident!("{}", var).to_token_stream(),
      },
      Stmt::Instr(instr) => quote! { #instr; },
      Stmt::Free(value) => quote! { self.free(#value); },
      Stmt::Return(value) => quote! { return #value; },
      Stmt::SetHeap { idx, port, value } => quote! { self.heap.set(#idx, #port, #value); },
      Stmt::Link { lhs, rhs } => quote! { self.link(#lhs, #rhs) },
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
      Instr::Const(Const::P1) => quote! { P1 },
      Instr::Const(Const::P2) => quote! { P2 },
      Instr::Const(Const::NULL) => quote! { NULL },
      Instr::Const(Const::ROOT) => quote! { ERAS },
      Instr::Const(Const::ERAS) => quote! { ERAS },
      Instr::Const(Const::VR1) => quote! { VR1 },
      Instr::Const(Const::VR2) => quote! { VR2 },
      Instr::Const(Const::RD1) => quote! { RD1 },
      Instr::Const(Const::RD2) => quote! { RD2 },
      Instr::Const(Const::REF) => quote! { REF },
      Instr::Const(Const::ERA) => quote! { ERA },
      Instr::Const(Const::NUM) => quote! { NUM },
      Instr::Const(Const::OP1) => quote! { OP1 },
      Instr::Const(Const::OP2) => quote! { OP2 },
      Instr::Const(Const::MAT) => quote! { MAT },
      Instr::Const(Const::CT0) => quote! { CT0 },
      Instr::Const(Const::CT1) => quote! { CT1 },
      Instr::Const(Const::CT2) => quote! { CT2 },
      Instr::Const(Const::CT3) => quote! { CT3 },
      Instr::Const(Const::CT4) => quote! { CT4 },
      Instr::Const(Const::CT5) => quote! { CT4 },
      Instr::Const(Const::USE) => quote! { USE },
      Instr::Const(Const::ADD) => quote! { ADD },
      Instr::Const(Const::SUB) => quote! { SUB },
      Instr::Const(Const::MUL) => quote! { MUL },
      Instr::Const(Const::DIV) => quote! { DIV },
      Instr::Const(Const::MOD) => quote! { MOD },
      Instr::Const(Const::EQ) => quote! { EQ },
      Instr::Const(Const::NE) => quote! { NE },
      Instr::Const(Const::LT) => quote! { LT },
      Instr::Const(Const::GT) => quote! { GT },
      Instr::Const(Const::AND) => quote! { AND },
      Instr::Const(Const::OR) => quote! { OR },
      Instr::Const(Const::XOR) => quote! { XOR },
      Instr::Const(Const::NOT) => quote! { NOT },
      Instr::Const(Const::RSH) => quote! { RSH },
      Instr::Const(Const::LSH) => quote! { LSH },
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
      Instr::Bin { op, lhs, rhs } => {
        match op.as_str() {
          "==" => quote! { #lhs == #rhs },
          "!=" => quote! { #lhs != #rhs },
          "&&" => quote! { #lhs && #rhs },
          "+" => quote! { #lhs + #rhs },
          "-" => quote! { #lhs - #rhs },
          _ => panic!()
        }
      }
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
