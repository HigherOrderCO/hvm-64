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

use crate::{
  ops::Op,
  run::Lab,
  util::{deref, maybe_grow},
};
use std::{collections::BTreeMap, fmt, str::FromStr};

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
#[derive(Clone, Hash, PartialEq, Eq, Debug, Default)]
pub enum Tree {
  #[default]
  /// A nilary eraser node.
  Era,
  /// A native 60-bit integer.
  Num { val: u64 },
  /// A nilary node, referencing a named net.
  Ref { nam: String },
  /// A binary interaction combinator.
  Ctr {
    /// The label of the combinator. (Combinators with the same label
    /// annihilate, and combinators with different labels commute.)
    lab: Lab,
    lft: Box<Tree>,
    rgt: Box<Tree>,
  },
  /// A binary node representing an operation on native integers.
  ///
  /// The principal port connects to the left operand.
  Op2 {
    /// The operation associated with this node.
    opr: Op,
    /// An auxiliary port; connects to the right operand.
    lft: Box<Tree>,
    /// An auxiliary port; connects to the output.
    rgt: Box<Tree>,
  },
  /// A unary node representing a partially-applied operation on native
  /// integers.
  ///
  /// The left operand is already applied. The principal port connects to the
  /// right operand.
  Op1 {
    /// The operation associated with this node.
    opr: Op,
    /// The left operand.
    lft: u64,
    /// An auxiliary port; connects to the output.
    rgt: Box<Tree>,
  },
  /// A binary node representing a match on native integers.
  ///
  /// The principal port connects to the integer to be matched on.
  Mat {
    /// An auxiliary port; connects to the branches of this match.
    ///
    /// This should be connected to something of the form:
    /// ```text
    /// (+value_if_zero (-predecessor_of_number +value_if_non_zero))
    /// ```
    sel: Box<Tree>,
    /// An auxiliary port; connects to the output.
    ret: Box<Tree>,
  },
  /// One side of a wire; the other side will have the same name.
  Var { nam: String },
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
          let lft = Box::new(self.parse_tree()?);
          let rgt = Box::new(self.parse_tree()?);
          self.consume(match char {
            '(' => ")",
            '[' => "]",
            '{' => "}",
            _ => unreachable!(),
          })?;
          Ok(Tree::Ctr { lab, lft, rgt })
        }
        // Ref = "@" Name
        Some('@') => {
          self.advance_char();
          self.skip_trivia();
          let nam = self.parse_name()?;
          Ok(Tree::Ref { nam })
        }
        // Num = "#" Int
        Some('#') => {
          self.advance_char();
          Ok(Tree::Num { val: self.parse_int()? })
        }
        // Op = "<" Op Tree Tree ">" | "<" Int Op Tree ">"
        Some('<') => {
          self.advance_char();
          if self.peek_char().is_some_and(|c| c.is_digit(10)) {
            let lft = self.parse_int()?;
            let opr = self.parse_op()?;
            let rgt = Box::new(self.parse_tree()?);
            self.consume(">")?;
            Ok(Tree::Op1 { opr, lft, rgt })
          } else {
            let opr = self.parse_op()?;
            let lft = Box::new(self.parse_tree()?);
            let rgt = Box::new(self.parse_tree()?);
            self.consume(">")?;
            Ok(Tree::Op2 { opr, lft, rgt })
          }
        }
        // Mat = "?<" Tree Tree ">"
        Some('?') => {
          self.advance_char();
          self.consume("<")?;
          let sel = Box::new(self.parse_tree()?);
          let ret = Box::new(self.parse_tree()?);
          self.consume(">")?;
          Ok(Tree::Mat { sel, ret })
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
    let radix = if let Some(rest) = self.input.strip_prefix("0x") {
      self.input = rest;
      16
    } else if let Some(rest) = self.input.strip_prefix("0b") {
      self.input = rest;
      2
    } else {
      10
    };
    let mut num: u64 = 0;
    if !self.peek_char().map_or(false, |c| c.is_digit(radix)) {
      return Err(format!("Expected a digit, found {:?}", self.peek_char()));
    }
    while let Some(digit) = self.peek_char().and_then(|c| c.to_digit(radix)) {
      self.advance_char();
      num = num * (radix as u64) + (digit as u64);
    }
    Ok(num)
  }

  /// See `ops.rs` for the available operators.
  fn parse_op(&mut self) -> Result<Op, String> {
    let op = self.take_while(|c| "+-=*/%<>|&^!?".contains(c));
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
    for (name, net) in self.iter() {
      writeln!(f, "@{name} = {net}")?;
    }
    Ok(())
  }
}

impl fmt::Display for Net {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", &self.root)?;
    for (a, b) in &self.redexes {
      write!(f, "\n& {a} ~ {b}")?;
    }
    Ok(())
  }
}

impl fmt::Display for Tree {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    maybe_grow(move || match self {
      Tree::Era => write!(f, "*"),
      Tree::Ctr { lab, lft, rgt } => match lab {
        0 => write!(f, "({lft} {rgt})"),
        1 => write!(f, "[{lft} {rgt}]"),
        _ => write!(f, "{{{lab} {lft} {rgt}}}"),
      },
      Tree::Var { nam } => write!(f, "{nam}"),
      Tree::Ref { nam } => write!(f, "@{nam}"),
      Tree::Num { val } => write!(f, "#{val}"),
      Tree::Op2 { opr, lft, rgt } => write!(f, "<{opr} {lft} {rgt}>"),
      Tree::Op1 { opr, lft, rgt } => write!(f, "<{lft}{opr} {rgt}>"),
      Tree::Mat { sel, ret } => write!(f, "?<{sel} {ret}>"),
    })
  }
}
