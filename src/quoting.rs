use std::str::FromStr;

use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens, TokenStreamExt, format_ident};

use crate::ir::{Constant, Function, Instr, Program, Stmt};

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
    let name = Ident::new(&format!("F_{}", self.name), Span::call_site());
    let body = &self.body;

    tokens.append_all(quote! {
      pub fn #name(&mut self, book: &Book, ptr: Ptr, argument: Ptr) -> bool {
        #( #body ; )*
      }
    })
  }
}

impl ToTokens for Stmt {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    tokens.append_all(quote! {})
  }
}

impl ToTokens for Instr {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    tokens.append_all(quote! {})
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
