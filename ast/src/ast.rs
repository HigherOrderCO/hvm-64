//! The textual language of HVMC.
//!
//! This file defines an AST for interaction nets, and functions to convert this
//! AST to/from the textual syntax.
//!
//! The grammar is documented in the repo README, as well as within the parser
//! methods, for convenience.
//!
//! The AST is based on the [interaction calculus].
//!
//! [interaction calculus]: https://en.wikipedia.org/wiki/Interaction_nets#Interaction_calculus
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

include!("../../prelude.rs");

#[cfg(feature = "parser")]
mod parser;

use alloc::collections::BTreeMap;

use crate::prelude::*;
use hvmc_util::{array_vec, create_var, deref, maybe_grow, ops::TypedOp as Op, var_to_num};

use arrayvec::ArrayVec;
use ordered_float::OrderedFloat;

pub type Lab = u16;

/// The top level AST node, representing a collection of named nets.
///
/// This is a newtype wrapper around a `BTreeMap<String, Net>`, and is
/// dereferencable to such.
#[derive(Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct Book {
  pub nets: BTreeMap<String, Net>,
}

deref!(Book => self.nets: BTreeMap<String, Net>);

/// An AST node representing an interaction net with one free port.
///
/// The tree connected to the free port is stored in `root`. The active pairs in
/// the net -- trees connected by their roots -- are stored in `redexes`.
///
/// (The wiring connecting the leaves of all the trees is represented within the
/// trees via pairs of [`Tree::Var`] nodes with the same name.)
#[derive(Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct Net {
  pub root: Tree,
  pub redexes: Vec<(Tree, Tree)>,
}

/// An AST node representing an interaction net tree.
///
/// Trees in interaction nets are inductively defined as either wires, or an
/// agent with all of its auxiliary ports (if any) connected to trees.
///
/// Here, the wires at the leaves of the tree are represented with
/// [`Tree::Var`], where the variable name is shared between both sides of the
/// wire.
#[derive(Hash, PartialEq, Eq, Debug, Default)]
pub enum Tree {
  #[default]
  /// A nilary eraser node.
  Era,
  /// A native 60-bit integer.
  Int { val: i64 },
  /// A native 32-bit float.
  F32 { val: OrderedFloat<f32> },
  /// A nilary node, referencing a named net.
  Ref { nam: String },
  /// A n-ary interaction combinator.
  Ctr {
    /// The label of the combinator. (Combinators with the same label
    /// annihilate, and combinators with different labels commute.)
    lab: Lab,
    /// The auxiliary ports of this node.
    ///
    /// - 0 ports: this behaves identically to an eraser node.
    /// - 1 port: this behaves identically to a wire.
    /// - 2 ports: this is a standard binary combinator node.
    /// - 3+ ports: equivalent to right-chained binary nodes; `(a b c)` is
    ///   equivalent to `(a (b c))`.
    ///
    /// The length of this vector must be less than [`MAX_ARITY`].
    ports: Vec<Tree>,
  },
  /// A binary node representing an operation on native integers.
  ///
  /// The principal port connects to the left operand.
  Op {
    /// The operation associated with this node.
    op: Op,
    /// An auxiliary port; connects to the right operand.
    rhs: Box<Tree>,
    /// An auxiliary port; connects to the output.
    out: Box<Tree>,
  },
  /// A binary node representing a match on native integers.
  ///
  /// The principal port connects to the integer to be matched on.
  Mat {
    /// An auxiliary port; connects to the zero branch.
    zero: Box<Tree>,
    /// An auxiliary port; connects to the a CTR with label 0 containing the
    /// predecessor and the output of the succ branch.
    succ: Box<Tree>,
    /// An auxiliary port; connects to the output.
    out: Box<Tree>,
  },
  /// An Scott-encoded ADT node.
  ///
  /// This is always equivalent to:
  /// ```text
  /// {$lab
  ///   * * * ...                // one era node per `variant_index`
  ///   {$lab $f0 $f1 $f2 ... R} // each field, in order, followed by a var node
  ///   * * * ...                // one era node per `variant_count - variant_index - 1`
  ///   R                        // a var node
  /// }
  /// ```
  ///
  /// For example:
  /// ```text
  /// data Option = None | (Some value):
  ///   None:
  ///     (0:2) = Adt { lab: 0, variant_index: 0, variant_count: 2, fields: [] }
  ///     (R * R) = Ctr { lab: 0, ports: [Var { nam: "R" }, Era, Var { nam: "R" }]}
  ///   (Some 123):
  ///     (1:2 #123) = Adt { lab: 0, variant_index: 0, variant_count: 2, fields: [Int { val: 123 }] }
  ///     (* (#123 R) R) = Ctr { lab: 0, ports: [Era, Ctr { lab: 0, ports: [Int { val: 123 }, Var { nam: "R" }] }, Var { nam: "R" }]}
  /// ```
  Adt {
    lab: Lab,
    /// The index of the variant of this ADT node.
    ///
    /// Must be less than `variant_count`.
    variant_index: usize,
    /// The number of variants in the data type.
    ///
    /// Must be greater than `0` and less than `MAX_ADT_VARIANTS`.
    variant_count: usize,
    /// The fields of this ADT node.
    ///
    /// Must have a length less than `MAX_ADT_FIELDS`.
    fields: Vec<Tree>,
  },
  /// One side of a wire; the other side will have the same name.
  Var { nam: String },
}

pub const MAX_ARITY: usize = 8;
pub const MAX_ADT_VARIANTS: usize = MAX_ARITY - 1;
pub const MAX_ADT_FIELDS: usize = MAX_ARITY - 1;

impl Net {
  pub fn trees(&self) -> impl Iterator<Item = &Tree> {
    iter::once(&self.root).chain(self.redexes.iter().flat_map(|(x, y)| [x, y]))
  }

  pub fn trees_mut(&mut self) -> impl Iterator<Item = &mut Tree> {
    iter::once(&mut self.root).chain(self.redexes.iter_mut().flat_map(|(x, y)| [x, y]))
  }

  /// Transforms the net `x & ...` into `y & x ~ (arg y) & ...`
  ///
  /// The result is equivalent a λ-calculus application. Thus,
  /// if the net is a λ-calculus term, then this function will
  /// apply an argument to it.
  pub fn apply_tree(&mut self, arg: Tree) {
    let mut fresh = 0usize;
    self.ensure_no_conflicts(&mut fresh);
    arg.ensure_no_conflicts(&mut fresh);

    let fresh_str = create_var(fresh + 1);

    let fun = mem::take(&mut self.root);
    let app = Tree::Ctr { lab: 0, ports: vec![arg, Tree::Var { nam: fresh_str.clone() }] };
    self.root = Tree::Var { nam: fresh_str };
    self.redexes.push((fun, app));
  }

  pub(crate) fn ensure_no_conflicts(&self, fresh: &mut usize) {
    self.root.ensure_no_conflicts(fresh);
    for (a, b) in &self.redexes {
      a.ensure_no_conflicts(fresh);
      b.ensure_no_conflicts(fresh);
    }
  }
}

impl Tree {
  #[inline(always)]
  pub fn children(&self) -> impl ExactSizeIterator + DoubleEndedIterator<Item = &Tree> {
    ArrayVec::<_, MAX_ARITY>::into_iter(match self {
      Tree::Era | Tree::Int { .. } | Tree::F32 { .. } | Tree::Ref { .. } | Tree::Var { .. } => {
        array_vec::from_array([])
      }
      Tree::Ctr { ports, .. } => array_vec::from_iter(ports),
      Tree::Op { rhs, out, .. } => array_vec::from_array([rhs, out]),
      Tree::Mat { zero, succ, out } => array_vec::from_array([zero, succ, out]),
      Tree::Adt { fields, .. } => array_vec::from_iter(fields),
    })
  }

  #[inline(always)]
  pub fn children_mut(&mut self) -> impl ExactSizeIterator + DoubleEndedIterator<Item = &mut Tree> {
    ArrayVec::<_, MAX_ARITY>::into_iter(match self {
      Tree::Era | Tree::Int { .. } | Tree::F32 { .. } | Tree::Ref { .. } | Tree::Var { .. } => {
        array_vec::from_array([])
      }
      Tree::Ctr { ports, .. } => array_vec::from_iter(ports),
      Tree::Op { rhs, out, .. } => array_vec::from_array([rhs, out]),
      Tree::Mat { zero, succ, out } => array_vec::from_array([zero, succ, out]),
      Tree::Adt { fields, .. } => array_vec::from_iter(fields),
    })
  }

  pub fn lab(&self) -> Option<Lab> {
    match self {
      Tree::Ctr { lab, ports } if ports.len() >= 2 => Some(*lab),
      Tree::Adt { lab, .. } => Some(*lab),
      _ => None,
    }
  }

  pub fn legacy_mat(mut arms: Tree, out: Tree) -> Option<Tree> {
    let Tree::Ctr { lab: 0, ports } = &mut arms else { None? };
    let ports = mem::take(ports);
    let Ok([zero, succ]) = <[_; 2]>::try_from(ports) else { None? };
    let zero = Box::new(zero);
    let succ = Box::new(succ);
    Some(Tree::Mat { zero, succ, out: Box::new(out) })
  }

  /// Increases `fresh` until `create_var(*fresh)` does not conflict
  /// with a [`Tree::Var`]  in `tree`
  ///
  /// This function can be called multiple times with many trees to
  /// ensure that `fresh` does not conflict with any of them.
  pub(crate) fn ensure_no_conflicts(&self, fresh: &mut usize) {
    if let Tree::Var { nam } = self {
      if let Some(var_num) = var_to_num(nam) {
        *fresh = (*fresh).max(var_num);
      }
    }
    self.children().for_each(|child| child.ensure_no_conflicts(fresh));
  }
}

// Manually implemented to avoid stack overflows.
impl Clone for Tree {
  fn clone(&self) -> Tree {
    maybe_grow(|| match self {
      Tree::Era => Tree::Era,
      Tree::Int { val } => Tree::Int { val: *val },
      Tree::F32 { val } => Tree::F32 { val: *val },
      Tree::Ref { nam } => Tree::Ref { nam: nam.clone() },
      Tree::Ctr { lab, ports } => Tree::Ctr { lab: *lab, ports: ports.clone() },
      Tree::Op { op, rhs, out } => Tree::Op { op: *op, rhs: rhs.clone(), out: out.clone() },
      Tree::Mat { zero, succ, out } => Tree::Mat { zero: zero.clone(), succ: succ.clone(), out: out.clone() },
      Tree::Adt { lab, variant_index, variant_count, fields } => {
        Tree::Adt { lab: *lab, variant_index: *variant_index, variant_count: *variant_count, fields: fields.clone() }
      }
      Tree::Var { nam } => Tree::Var { nam: nam.clone() },
    })
  }
}

// Drops non-recursively to avoid stack overflows.
impl Drop for Tree {
  fn drop(&mut self) {
    loop {
      let mut i = self.children_mut().filter(|x| x.children().len() != 0);
      let Some(x) = i.next() else { break };
      if { i }.next().is_none() {
        // There's only one child; move it up to be the new root.
        *self = mem::take(x);
        continue;
      }
      // Rotate the tree right:
      // ```text
      //     a            b
      //    / \          / \
      //   b   e   ->   c   a
      //  / \              / \
      // c   d            d   e
      // ```
      let d = mem::take(x.children_mut().next_back().unwrap());
      let b = mem::replace(x, d);
      let a = mem::replace(self, b);
      mem::forget(mem::replace(self.children_mut().next_back().unwrap(), a));
    }
  }
}

impl fmt::Display for Book {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    for (i, (name, net)) in self.iter().enumerate() {
      if i != 0 {
        f.write_str("\n\n")?;
      }

      write!(f, "@{name} = {net}")?;
    }
    Ok(())
  }
}

impl fmt::Display for Net {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", &self.root)?;

    for (a, b) in &self.redexes {
      write!(f, "\n  & {a} ~ {b}")?;
    }

    Ok(())
  }
}

impl fmt::Display for Tree {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    maybe_grow(move || match self {
      Tree::Era => write!(f, "*"),
      Tree::Ctr { lab, ports } => {
        match lab {
          0 => write!(f, "("),
          1 => write!(f, "["),
          _ => write!(f, "{{{lab}"),
        }?;
        let mut space = *lab > 1;
        for port in ports {
          if space {
            write!(f, " ")?;
          }
          write!(f, "{port}")?;
          space = true;
        }
        match lab {
          0 => write!(f, ")"),
          1 => write!(f, "]"),
          _ => write!(f, "}}"),
        }?;
        Ok(())
      }
      Tree::Adt { lab, variant_index, variant_count, fields } => {
        match lab {
          0 => write!(f, "("),
          1 => write!(f, "["),
          _ => write!(f, "{{{lab}"),
        }?;
        write!(f, ":{}:{}", variant_index, variant_count)?;
        for field in fields {
          write!(f, " {field}")?;
        }
        match lab {
          0 => write!(f, ")"),
          1 => write!(f, "]"),
          _ => write!(f, "}}"),
        }?;
        Ok(())
      }
      Tree::Var { nam } => write!(f, "{nam}"),
      Tree::Ref { nam } => write!(f, "@{nam}"),
      Tree::Int { val } => write!(f, "#{val}"),
      Tree::F32 { val } => write!(f, "#{:?}", val.0),
      Tree::Op { op, rhs, out } => write!(f, "<{op} {rhs} {out}>"),
      Tree::Mat { zero, succ, out } => write!(f, "?<{zero} {succ} {out}>"),
    })
  }
}

// #[test]
// fn test_tree_drop() {
//   use alloc::vec;

//   drop(Tree::from_str("((* (* *)) (* *))"));

//   let mut long_tree = Tree::Era;
//   let mut cursor = &mut long_tree;
//   for _ in 0 .. 100_000 {
//     *cursor = Tree::Ctr { lab: 0, ports: vec![Tree::Era, Tree::Era] };
//     let Tree::Ctr { ports, .. } = cursor else { unreachable!() };
//     cursor = &mut ports[0];
//   }
//   drop(long_tree);

//   let mut big_tree = Tree::Era;
//   for _ in 0 .. 16 {
//     big_tree = Tree::Ctr { lab: 0, ports: vec![big_tree.clone(), big_tree] };
//   }
//   drop(big_tree);
// }
