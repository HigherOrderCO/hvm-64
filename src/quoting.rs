use quote::quote;

use crate::ir::Program;

impl Program {
    pub fn into_token_stream(self) -> proc_macro2::TokenStream  {
        quote! {}
    }
}