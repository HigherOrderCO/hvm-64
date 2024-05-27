//! Carries out simple eta-reduction, to reduce the amount of rewrites at
//! runtime.
//!
//! ### Eta-equivalence
//!
//! In interaction combinators, there are some nets that are equivalent and
//! have no observable difference
//!
//! ![Image of eta-equivalence](https://i.postimg.cc/XYVxdMFW/image.png)
//!
//! This module implements the eta-equivalence rule at the top-left of the image
//! above
//!
//! ```txt
//!     /|-, ,-|\     eta_reduce
//! ---| |  X  | |-- ~~~~~~~~~~~~> -------------
//!     \|-' '-|/
//! ```
//!
//! In hvm-64's AST representation, this reduction looks like this
//!
//! ```txt
//! {lab x y} ... {lab x y} ~~~~~~~~> x ..... x
//! ```
//!
//! Essentially, both occurrences of the same constructor are replaced by a
//! variable.
//!
//! ### The algorithm
//!
//! The code uses a two-pass O(n) algorithm, where `n` is the amount of nodes
//! in the AST
//!
//! In the first pass, a node-list is built out of an ordered traversal of the
//! AST. Crucially, the node list stores variable offsets instead of the
//! variable's names Since the AST's order is consistent, the ordering of nodes
//! in the node list can be reproduced with a traversal.
//!
//! This means that each occurrence of a variable is encoded with the offset in
//! the node-list to the _other_ occurrence of the variable.
//!
//! For example, if we start with the net: `[(x y) (x y)]`
//!
//! The resulting node list will look like this:
//!
//! `[Ctr(1), Ctr(0), Var(3), Var(3), Ctr(0), Var(-3), Var(-3)]`
//!
//! The second pass uses the node list to find repeated constructors. If a
//! constructor's children are both variables with the same offset, then we
//! lookup that offset relative to the constructor. If it is equal to the first
//! constructor, it means both of them are equal and they can be replaced with a
//! variable.
//!
//! The pass also reduces subnets such as `(* *) -> *`

use core::ops::RangeFrom;

use crate::prelude::*;
use hvm64_ast::{Net, Tree};

use ordered_float::OrderedFloat;

pub trait EtaReduce {
  fn eta_reduce(&mut self);
}

impl EtaReduce for Net {
  /// Carries out simple eta-reduction
  fn eta_reduce(&mut self) {
    let mut phase1 = Phase1::default();
    for tree in self.trees() {
      phase1.walk_tree(tree);
    }
    let mut phase2 = Phase2 { nodes: phase1.nodes, index: 0 .. };
    for tree in self.trees_mut() {
      phase2.reduce_tree(tree);
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeType {
  Ctr(u16),
  Var(isize),
  Int(i64),
  F32(OrderedFloat<f32>),
  Era,
  Other,
  Hole,
}

#[derive(Default)]
struct Phase1<'a> {
  vars: Map<&'a str, usize>,
  nodes: Vec<NodeType>,
}

impl<'a> Phase1<'a> {
  fn walk_tree(&mut self, tree: &'a Tree) {
    match tree {
      Tree::Ctr { lab, lft, rgt } => {
        self.nodes.push(NodeType::Ctr(*lab));
        self.walk_tree(lft);
        self.walk_tree(rgt);
      }
      Tree::Var { nam } => {
        if let Some(i) = self.vars.get(&**nam) {
          let j = self.nodes.len() as isize;
          self.nodes.push(NodeType::Var(*i as isize - j));
          self.nodes[*i] = NodeType::Var(j - *i as isize);
        } else {
          self.vars.insert(nam, self.nodes.len());
          self.nodes.push(NodeType::Hole);
        }
      }
      Tree::Era => self.nodes.push(NodeType::Era),
      Tree::Int { val } => self.nodes.push(NodeType::Int(*val)),
      Tree::F32 { val } => self.nodes.push(NodeType::F32(*val)),
      _ => {
        self.nodes.push(NodeType::Other);
        for i in tree.children() {
          self.walk_tree(i);
        }
      }
    }
  }
}

struct Phase2 {
  nodes: Vec<NodeType>,
  index: RangeFrom<usize>,
}

impl Phase2 {
  fn reduce_tree(&mut self, tree: &mut Tree) -> NodeType {
    let index = self.index.next().unwrap();
    let ty = self.nodes[index];
    if let Tree::Ctr { lft, rgt, .. } = tree {
      let a = self.reduce_tree(lft);
      let b = self.reduce_tree(rgt);
      if a == b {
        let reducible = match a {
          NodeType::Var(delta) => self.nodes[index.wrapping_add_signed(delta)] == ty,
          NodeType::Era | NodeType::Int(_) | NodeType::F32(_) => true,
          _ => false,
        };
        if reducible {
          *tree = mem::take(lft);
          return a;
        }
      }
    } else {
      for i in tree.children_mut() {
        self.reduce_tree(i);
      }
    }
    ty
  }
}
