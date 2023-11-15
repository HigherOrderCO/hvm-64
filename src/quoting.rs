use std::str::FromStr;

use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

use crate::ir::{Instr, Program, Stmt};

impl Program {
  pub fn into_token_stream(self) -> TokenStream {
    let constants = self.values.iter().map(|(name, value)| {
      let name = Ident::new(&format!("F_{name}"), Span::call_site());
      let value = TokenStream::from_str(&format!("0x{value:06x}"));

      quote! { pub const #name: u32 = #value; }
    });

    let functions = self.functions.into_iter().map(|function| {
      let name = function.name.clone();
      let name = Ident::new(&format!("F_{name}"), Span::call_site());
      let body = function
        .body
        .into_iter()
        .map(|stmt| stmt.into_token_stream())
        .collect::<Vec<_>>();

      quote! {
        pub fn #name(&mut self, book: &Book, ptr: Ptr, argument: Ptr) -> bool {
          #( #body ; )*
        }
      }
    });

    quote! {
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
    }
  }
}

impl Stmt {
  pub fn into_token_stream(self) -> TokenStream {
    match self {
      Stmt::Let { name, value } => quote! {},
      Stmt::Val { name, type_repr } => quote! {},
      Stmt::Assign { name, value } => quote! {},
      Stmt::Instr(_) => quote! {},
      Stmt::Free(_) => quote! {},
      Stmt::Return(_) => quote! {},
      Stmt::SetHeap { idx, port, value } => quote! {},
      Stmt::Link { lhs, rhs } => quote! {},
    }
  }
}

impl Instr {
  pub fn into_token_stream(self) -> TokenStream {
    match self {
      Instr::True => quote! {},
      Instr::False => quote! {},
      Instr::Int(_) => quote! {},
      Instr::Hex(_) => quote! {},
      Instr::Con(_) => quote! {},
      Instr::Prop(_) => quote! {},
      Instr::Call { name, args } => quote! {},
      Instr::If {
        cond,
        then,
        otherwise,
      } => quote! {},
      Instr::Not { ins } => quote! {},
      Instr::Bin { op, lhs, rhs } => quote! {},
      Instr::Val { ins } => quote! {},
      Instr::Tag { ins } => quote! {},
      Instr::IsNum { ins } => quote! {},
      Instr::IsSkp { ins } => quote! {},
      Instr::NewPtr { tag, value } => quote! {},
      Instr::Op { lhs, rhs } => quote! {},
      Instr::Alloc { size } => quote! {},
      Instr::GetHeap { idx, port } => quote! {},
    }
  }
}
