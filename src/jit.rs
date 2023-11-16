//! JIT compilation of Cranelift IR.

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, Linkage, Module};

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
