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

use arrayvec::ArrayVec;
use ordered_float::OrderedFloat;

use crate::{
  ops::TypedOp as Op,
  run::Lab,
  util::{array_vec, deref, maybe_grow},
};
use std::{collections::BTreeMap, fmt, mem, str::FromStr};

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
    std::iter::once(&self.root).chain(self.redexes.iter().map(|(x, y)| [x, y]).flatten())
  }
  pub fn trees_mut(&mut self) -> impl Iterator<Item = &mut Tree> {
    std::iter::once(&mut self.root).chain(self.redexes.iter_mut().map(|(x, y)| [x, y]).flatten())
  }
}

impl Tree {
  #[inline(always)]
  pub fn children(&self) -> impl Iterator<Item = &Tree> + ExactSizeIterator + DoubleEndedIterator {
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
  pub fn children_mut(&mut self) -> impl Iterator<Item = &mut Tree> + ExactSizeIterator + DoubleEndedIterator {
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

  pub(crate) fn lab(&self) -> Option<Lab> {
    match self {
      Tree::Ctr { lab, ports } if ports.len() >= 2 => Some(*lab),
      Tree::Adt { lab, .. } => Some(*lab),
      _ => None,
    }
  }

  pub fn legacy_mat(mut arms: Tree, out: Tree) -> Option<Tree> {
    let Tree::Ctr { lab: 0, ports } = &mut arms else { None? };
    let ports = std::mem::take(ports);
    let Ok([zero, succ]) = <[_; 2]>::try_from(ports) else { None? };
    let zero = Box::new(zero);
    let succ = Box::new(succ);
    Some(Tree::Mat { zero, succ, out: Box::new(out) })
  }
}

/// The state of the HVMC parser.
struct Parser<'i> {
  /// The remaining characters in the input. An empty string indicates EOF.
  input: &'i str,
}

impl<'i> Parser<'i> {
  /// Book = ("@" Name "=" Net)*
  fn parse_book(&mut self) -> Result<Book, String> {
    maybe_grow(move || {
      let mut book = BTreeMap::new();
      while self.consume("@").is_ok() {
        let name = self.parse_name()?;
        self.consume("=")?;
        let net = self.parse_net()?;
        book.insert(name, net);
      }
      Ok(Book { nets: book })
    })
  }

  /// Net = Tree ("&" Tree "~" Tree)*
  fn parse_net(&mut self) -> Result<Net, String> {
    let mut redexes = Vec::new();
    let root = self.parse_tree()?;
    while self.consume("&").is_ok() {
      let tree1 = self.parse_tree()?;
      self.consume("~")?;
      let tree2 = self.parse_tree()?;
      redexes.push((tree1, tree2));
    }
    Ok(Net { root, redexes })
  }

  fn parse_tree(&mut self) -> Result<Tree, String> {
    maybe_grow(move || {
      self.skip_trivia();
      match self.peek_char() {
        // Era = "*"
        Some('*') => {
          self.advance_char();
          Ok(Tree::Era)
        }
        // Ctr = "(" Tree Tree ")" | "[" Tree Tree "]" | "{" Int Tree Tree "}"
        Some(char @ ('(' | '[' | '{')) => {
          self.advance_char();
          let lab = match char {
            '(' => 0,
            '[' => 1,
            '{' => self.parse_int()? as Lab,
            _ => unreachable!(),
          };
          let close = match char {
            '(' => ')',
            '[' => ']',
            '{' => '}',
            _ => unreachable!(),
          };
          self.skip_trivia();
          if self.peek_char().is_some_and(|x| x == ':') {
            self.advance_char();
            let variant_index = self.parse_int()?;
            self.consume(":")?;
            let variant_count = self.parse_int()?;
            let mut fields = Vec::new();
            self.skip_trivia();
            while self.peek_char() != Some(close) {
              fields.push(self.parse_tree()?);
              self.skip_trivia();
            }
            self.advance_char();
            if variant_count == 0 {
              Err("variant count cannot be zero".to_owned())?;
            }
            if variant_count > (MAX_ADT_VARIANTS as u64) {
              Err("adt has too many variants".to_owned())?;
            }
            if variant_index >= variant_count {
              Err("variant index out of range".to_owned())?;
            }
            let variant_index = variant_index as usize;
            let variant_count = variant_count as usize;
            if fields.len() > MAX_ADT_FIELDS {
              Err("adt has too many fields".to_owned())?;
            }
            Ok(Tree::Adt { lab, variant_index, variant_count, fields })
          } else {
            let mut ports = Vec::new();
            self.skip_trivia();
            while self.peek_char() != Some(close) {
              ports.push(self.parse_tree()?);
              self.skip_trivia();
            }
            self.advance_char();
            if ports.len() > MAX_ARITY {
              Err("ctr has too many ports".to_owned())?;
            }
            Ok(Tree::Ctr { lab, ports })
          }
        }
        // Ref = "@" Name
        Some('@') => {
          self.advance_char();
          self.skip_trivia();
          let nam = self.parse_name()?;
          Ok(Tree::Ref { nam })
        }
        // Int = "#" [-] Int
        // F32 = "#" [-] ( Int "." Int | "NaN" | "inf" )
        Some('#') => {
          self.advance_char();
          let is_neg = self.consume("-").is_ok();
          let num = self.take_while(|c| c.is_alphanumeric() || c == '.');

          if num.contains(".") || num.contains("NaN") || num.contains("inf") {
            let mut val: f32 = num.parse().map_err(|err| format!("{err:?}"))?;
            if is_neg {
              val = -val;
            }
            Ok(Tree::F32 { val: val.into() })
          } else {
            let mut val: i64 = parse_int(num)? as i64;
            if is_neg {
              val = -val;
            }
            Ok(Tree::Int { val })
          }
        }
        // Op = "<" Op Tree Tree ">"
        Some('<') => {
          self.advance_char();
          let op = self.parse_op()?;
          let rhs = Box::new(self.parse_tree()?);
          let out = Box::new(self.parse_tree()?);
          self.consume(">")?;
          Ok(Tree::Op { op, rhs, out })
        }
        // Mat = "?<" Tree Tree ">"
        Some('?') => {
          self.advance_char();
          self.consume("<")?;
          let zero = self.parse_tree()?;
          let succ = self.parse_tree()?;
          self.skip_trivia();
          if self.peek_char() == Some('>') {
            self.advance_char();
            Tree::legacy_mat(zero, succ).ok_or_else(|| "invalid legacy match".to_owned())
          } else {
            let zero = Box::new(zero);
            let succ = Box::new(succ);
            let out = Box::new(self.parse_tree()?);
            self.consume(">")?;
            Ok(Tree::Mat { zero, succ, out })
          }
        }
        // Var = Name
        _ => Ok(Tree::Var { nam: self.parse_name()? }),
      }
    })
  }

  /// Name = /[a-zA-Z0-9_.$]+/
  fn parse_name(&mut self) -> Result<String, String> {
    let name = self.take_while(|c| c.is_alphanumeric() || c == '_' || c == '.' || c == '$');
    if name.is_empty() {
      return Err(format!("Expected a name character, found {:?}", self.peek_char()));
    }
    Ok(name.to_owned())
  }

  /// Int = /[0-9]+/ | /0x[0-9a-fA-F]+/ | /0b[01]+/
  fn parse_int(&mut self) -> Result<u64, String> {
    self.skip_trivia();
    parse_int(self.take_while(|c| c.is_alphanumeric()))
  }

  /// See `ops.rs` for the available operators.
  fn parse_op(&mut self) -> Result<Op, String> {
    let op = self.take_while(|c| c.is_alphanumeric() || ".+-=*/%<>|&^!?$".contains(c));
    op.parse().map_err(|_| format!("Unknown operator: {op:?}"))
  }

  /// Inspects the next character in the input without consuming it.
  fn peek_char(&self) -> Option<char> {
    self.input.chars().next()
  }

  /// Consumes the next character in the input.
  fn advance_char(&mut self) -> Option<char> {
    let char = self.input.chars().next()?;
    self.input = &self.input[char.len_utf8() ..];
    Some(char)
  }

  /// Skips whitespace & comments in the input.
  fn skip_trivia(&mut self) {
    while let Some(c) = self.peek_char() {
      if c.is_ascii_whitespace() {
        self.advance_char();
        continue;
      }
      if c == '/' && self.input.starts_with("//") {
        while self.peek_char() != Some('\n') {
          self.advance_char();
        }
        continue;
      }
      break;
    }
  }

  /// Consumes an instance of the given string, erroring if it is not found.
  fn consume(&mut self, text: &str) -> Result<(), String> {
    self.skip_trivia();
    let Some(rest) = self.input.strip_prefix(text) else {
      return Err(format!("Expected {:?}, found {:?}", text, self.input.split_ascii_whitespace().next().unwrap_or("")));
    };
    self.input = rest;
    Ok(())
  }

  /// Consumes all of the contiguous next characters in the input matching a
  /// given predicate.
  fn take_while(&mut self, mut f: impl FnMut(char) -> bool) -> &'i str {
    let len = self.input.chars().take_while(|&c| f(c)).map(char::len_utf8).sum();
    let (name, rest) = self.input.split_at(len);
    self.input = rest;
    name
  }
}

/// Parses an unsigned integer with an optional radix prefix.
fn parse_int(input: &str) -> Result<u64, String> {
  if let Some(rest) = input.strip_prefix("0x") {
    u64::from_str_radix(rest, 16).map_err(|err| format!("{err:?}"))
  } else if let Some(rest) = input.strip_prefix("0b") {
    u64::from_str_radix(rest, 2).map_err(|err| format!("{err:?}"))
  } else {
    u64::from_str_radix(input, 10).map_err(|err| format!("{err:?}"))
  }
}

/// Parses the input with the callback, ensuring that the whole input is
/// consumed.
fn parse_eof<'i, T>(input: &'i str, parse_fn: impl Fn(&mut Parser<'i>) -> Result<T, String>) -> Result<T, String> {
  let mut parser = Parser { input };
  let out = parse_fn(&mut parser)?;
  if !parser.input.is_empty() {
    return Err("Unable to parse the whole input. Is this not an hvmc file?".to_owned());
  }
  Ok(out)
}

impl FromStr for Book {
  type Err = String;
  fn from_str(str: &str) -> Result<Self, Self::Err> {
    parse_eof(str, Parser::parse_book)
  }
}

impl FromStr for Net {
  type Err = String;
  fn from_str(str: &str) -> Result<Self, Self::Err> {
    parse_eof(str, Parser::parse_net)
  }
}

impl FromStr for Tree {
  type Err = String;
  fn from_str(str: &str) -> Result<Self, Self::Err> {
    parse_eof(str, Parser::parse_tree)
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

// Manually implemented to avoid stack overflows.
impl Clone for Tree {
  fn clone(&self) -> Tree {
    maybe_grow(|| match self {
      Tree::Era => Tree::Era,
      Tree::Int { val } => Tree::Int { val: val.clone() },
      Tree::F32 { val } => Tree::F32 { val: val.clone() },
      Tree::Ref { nam } => Tree::Ref { nam: nam.clone() },
      Tree::Ctr { lab, ports } => Tree::Ctr { lab: lab.clone(), ports: ports.clone() },
      Tree::Op { op, rhs, out } => Tree::Op { op: op.clone(), rhs: rhs.clone(), out: out.clone() },
      Tree::Mat { zero, succ, out } => Tree::Mat { zero: zero.clone(), succ: succ.clone(), out: out.clone() },
      Tree::Adt { lab, variant_index, variant_count, fields } => Tree::Adt {
        lab: lab.clone(),
        variant_index: variant_index.clone(),
        variant_count: variant_count.clone(),
        fields: fields.clone(),
      },
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

#[test]
fn test_tree_drop() {
  drop(Tree::from_str("((* (* *)) (* *))"));

  let mut long_tree = Tree::Era;
  let mut cursor = &mut long_tree;
  for _ in 0 .. 100_000 {
    *cursor = Tree::Ctr { lab: 0, ports: vec![Tree::Era, Tree::Era] };
    let Tree::Ctr { ports, .. } = cursor else { unreachable!() };
    cursor = &mut ports[0];
  }
  drop(long_tree);

  let mut big_tree = Tree::Era;
  for _ in 0 .. 16 {
    big_tree = Tree::Ctr { lab: 0, ports: vec![big_tree.clone(), big_tree] };
  }
  drop(big_tree);
}
