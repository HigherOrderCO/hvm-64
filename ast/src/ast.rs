//! The textual language of hvm-64.
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

#[cfg(feature = "parser")]
mod parser;

use alloc::collections::BTreeMap;

use hvm64_util::{create_var, deref_to, maybe_grow, multi_iterator, ops::TypedOp as Op, prelude::*, var_to_num};

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

deref_to!(Book => self.nets: BTreeMap<String, Net>);

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
  Ref(String),
  /// A n-ary interaction combinator.
  Ctr {
    /// The label of the combinator. (Combinators with the same label
    /// annihilate, and combinators with different labels commute.)
    lab: Lab,
    p1: Box<Tree>,
    p2: Box<Tree>,
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
    /// An auxiliary port; connects to a tree of the following structure:
    /// ```text
    /// (+value_if_zero (-predecessor_of_number +value_if_succ))
    /// ```
    arms: Box<Tree>,
    /// An auxiliary port; connects to the output.
    out: Box<Tree>,
  },
  /// One side of a wire; the other side will have the same name.
  Var(String),
}

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
    let app = Tree::Ctr { lab: 0, p1: Box::new(arg), p2: Box::new(Tree::Var(fresh_str.clone())) };
    self.root = Tree::Var(fresh_str);
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
    multi_iterator! { Iter { Nil, Two } }
    match self {
      Tree::Era | Tree::Int { .. } | Tree::F32 { .. } | Tree::Ref(_) | Tree::Var(_) => Iter::Nil([]),
      Tree::Ctr { p1, p2, .. } => Iter::Two([&**p1, p2]),
      Tree::Op { rhs, out, .. } => Iter::Two([&**rhs, out]),
      Tree::Mat { arms, out } => Iter::Two([&**arms, out]),
    }
  }

  #[inline(always)]
  pub fn children_mut(&mut self) -> impl ExactSizeIterator + DoubleEndedIterator<Item = &mut Tree> {
    multi_iterator! { Iter { Nil, Two } }
    match self {
      Tree::Era | Tree::Int { .. } | Tree::F32 { .. } | Tree::Ref(_) | Tree::Var(_) => Iter::Nil([]),
      Tree::Ctr { p1, p2, .. } => Iter::Two([&mut **p1, p2]),
      Tree::Op { rhs, out, .. } => Iter::Two([&mut **rhs, out]),
      Tree::Mat { arms, out } => Iter::Two([&mut **arms, out]),
    }
  }

  pub fn lab(&self) -> Option<Lab> {
    match self {
      Tree::Ctr { lab, .. } => Some(*lab),
      _ => None,
    }
  }

  /// Increases `fresh` until `create_var(*fresh)` does not conflict
  /// with a [`Tree::Var`]  in `tree`
  ///
  /// This function can be called multiple times with many trees to
  /// ensure that `fresh` does not conflict with any of them.
  pub(crate) fn ensure_no_conflicts(&self, fresh: &mut usize) {
    if let Tree::Var(name) = self {
      if let Some(var_num) = var_to_num(name) {
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
      Tree::Ref(name) => Tree::Ref(name.clone()),
      Tree::Ctr { lab, p1, p2 } => Tree::Ctr { lab: *lab, p1: p1.clone(), p2: p2.clone() },
      Tree::Op { op, rhs, out } => Tree::Op { op: *op, rhs: rhs.clone(), out: out.clone() },
      Tree::Mat { arms, out } => Tree::Mat { arms: arms.clone(), out: out.clone() },
      Tree::Var(name) => Tree::Var(name.clone()),
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
      Tree::Ctr { lab, p1, p2 } => match lab {
        0 => write!(f, "({p1} {p2})"),
        1 => write!(f, "[{p1} {p2}]"),
        _ => write!(f, "{{{lab} {p1} {p2}}}"),
      },
      Tree::Var(name) => write!(f, "{name}"),
      Tree::Ref(name) => write!(f, "@{name}"),
      Tree::Int { val } => write!(f, "#{val}"),
      Tree::F32 { val } => write!(f, "#{:?}", val.0),
      Tree::Op { op, rhs, out } => write!(f, "<{op} {rhs} {out}>"),
      Tree::Mat { arms, out } => write!(f, "?<{arms} {out}>"),
    })
  }
}

#[test]
#[cfg(feature = "parser")]
fn test_tree_drop() {
  use core::str::FromStr;

  drop(Tree::from_str("((* (* *)) (* *))"));

  let mut long_tree = Tree::Era;
  let mut cursor = &mut long_tree;
  for _ in 0 .. 100_000 {
    *cursor = Tree::Ctr { lab: 0, p1: Box::new(Tree::Era), p2: Box::new(Tree::Era) };
    let Tree::Ctr { p1, .. } = cursor else { unreachable!() };
    cursor = p1;
  }
  drop(long_tree);

  let mut big_tree = Tree::Era;
  for _ in 0 .. 16 {
    big_tree = Tree::Ctr { lab: 0, p1: Box::new(big_tree.clone()), p2: Box::new(big_tree) };
  }
  drop(big_tree);
}
