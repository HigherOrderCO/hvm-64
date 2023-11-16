//! JIT compilation of Cranelift IR.

use std::collections::HashMap;

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, Linkage, Module};

use crate::ir::{Const, Instr};

struct Lowering {
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

impl Default for Lowering {
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

impl Lowering {
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

/// A collection of state used for translating from toy-language AST nodes
/// into Cranelift IR.
struct FunctionLowering<'a> {
  int: types::Type,
  builder: FunctionBuilder<'a>,
  variables: HashMap<String, Variable>,
  module: &'a mut JITModule,
}

impl FunctionLowering<'_> {
  fn lower_instr(&mut self, instr: Instr) -> Value {
    match instr {
      Instr::True => self.builder.ins().iconst(self.int, 0),
      Instr::False => self.builder.ins().iconst(self.int, 0),
      Instr::Int(v) => self.builder.ins().iconst(self.int, v as i64),
      Instr::Const(cons) => match cons {
        Const::F(v) => todo!(),
        Const::P1 => self.builder.ins().iconst(self.int, crate::run::P1 as i64),
        Const::P2 => self.builder.ins().iconst(self.int, crate::run::P2 as i64),
        Const::NULL => self
          .builder
          .ins()
          .iconst(self.int, crate::run::NULL.0 as i64),
        Const::ROOT => self
          .builder
          .ins()
          .iconst(self.int, crate::run::ERAS.0 as i64),
        Const::ERAS => self
          .builder
          .ins()
          .iconst(self.int, crate::run::ERAS.0 as i64),
        Const::VR1 => self.builder.ins().iconst(self.int, crate::run::VR1 as i64),
        Const::VR2 => self.builder.ins().iconst(self.int, crate::run::VR2 as i64),
        Const::RD1 => self.builder.ins().iconst(self.int, crate::run::RD1 as i64),
        Const::RD2 => self.builder.ins().iconst(self.int, crate::run::RD2 as i64),
        Const::REF => self.builder.ins().iconst(self.int, crate::run::REF as i64),
        Const::ERA => self.builder.ins().iconst(self.int, crate::run::ERA as i64),
        Const::NUM => self.builder.ins().iconst(self.int, crate::run::NUM as i64),
        Const::OP1 => self.builder.ins().iconst(self.int, crate::run::OP1 as i64),
        Const::OP2 => self.builder.ins().iconst(self.int, crate::run::OP2 as i64),
        Const::MAT => self.builder.ins().iconst(self.int, crate::run::MAT as i64),
        Const::CT0 => self.builder.ins().iconst(self.int, crate::run::CT0 as i64),
        Const::CT1 => self.builder.ins().iconst(self.int, crate::run::CT1 as i64),
        Const::CT2 => self.builder.ins().iconst(self.int, crate::run::CT2 as i64),
        Const::CT3 => self.builder.ins().iconst(self.int, crate::run::CT3 as i64),
        Const::CT4 => self.builder.ins().iconst(self.int, crate::run::CT4 as i64),
        Const::CT5 => self.builder.ins().iconst(self.int, crate::run::CT5 as i64),
        Const::USE => self.builder.ins().iconst(self.int, crate::run::USE as i64),
        Const::ADD => self.builder.ins().iconst(self.int, crate::run::ADD as i64),
        Const::SUB => self.builder.ins().iconst(self.int, crate::run::SUB as i64),
        Const::MUL => self.builder.ins().iconst(self.int, crate::run::MUL as i64),
        Const::DIV => self.builder.ins().iconst(self.int, crate::run::DIV as i64),
        Const::MOD => self.builder.ins().iconst(self.int, crate::run::MOD as i64),
        Const::EQ => self.builder.ins().iconst(self.int, crate::run::EQ as i64),
        Const::NE => self.builder.ins().iconst(self.int, crate::run::NE as i64),
        Const::LT => self.builder.ins().iconst(self.int, crate::run::LT as i64),
        Const::GT => self.builder.ins().iconst(self.int, crate::run::GT as i64),
        Const::AND => self.builder.ins().iconst(self.int, crate::run::AND as i64),
        Const::OR => self.builder.ins().iconst(self.int, crate::run::OR as i64),
        Const::XOR => self.builder.ins().iconst(self.int, crate::run::XOR as i64),
        Const::NOT => self.builder.ins().iconst(self.int, crate::run::NOT as i64),
        Const::RSH => self.builder.ins().iconst(self.int, crate::run::RSH as i64),
        Const::LSH => self.builder.ins().iconst(self.int, crate::run::LSH as i64),
      },
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
