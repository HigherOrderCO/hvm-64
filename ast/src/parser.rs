use hvm64_util::prelude::*;

use alloc::collections::BTreeMap;
use core::str::FromStr;

use crate::{Book, Net, Tree};
use hvm64_num::{Num, NumTag};
use hvm64_util::maybe_grow;

use TSPL::{new_parser, Parser};

new_parser!(Hvm64Parser);

impl<'i> Hvm64Parser<'i> {
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

  pub fn parse_tree(&mut self) -> Result<Tree, String> {
    maybe_grow(move || {
      self.skip_trivia();
      match self.peek_one() {
        // Era = "*"
        Some('*') => {
          self.advance_one();
          Ok(Tree::Era)
        }
        // Ctr = ("#" Num)? ("(" Tree Tree ")" | "{" Tree Tree "}")
        mut char @ Some('(' | '{' | '#') => {
          self.advance_one();
          let tag = if char == Some('#') {
            let tag = self.parse_u64().unwrap() as u16;
            self.skip_trivia();
            char = self.advance_one();
            tag
          } else {
            0
          };
          let (close, kind) = match char {
            Some('(') => (")", 0),
            Some('{') => ("}", 1),
            _ => self.expected("`(` or `{`")?,
          };
          let lab = tag * 2 + kind;
          self.skip_trivia();
          let p1 = Box::new(self.parse_tree()?);
          let p2 = Box::new(self.parse_tree()?);
          self.consume(close)?;
          Ok(Tree::Ctr { lab, p1, p2 })
        }
        // Ref = "@" Name
        Some('@') => {
          self.advance_one();
          self.skip_trivia();
          let name = self.parse_name()?;
          Ok(Tree::Ref(name))
        }
        // Op = "$(" Tree Tree ")"
        Some('$') => {
          self.consume("$(");
          let rhs = Box::new(self.parse_tree()?);
          let out = Box::new(self.parse_tree()?);
          self.consume(">")?;
          Ok(Tree::Op { rhs, out })
        }
        // Switch = "?(" Tree Tree ")"
        Some('?') => {
          self.consume("?(")?;
          let arms = Box::new(self.parse_tree()?);
          let out = Box::new(self.parse_tree()?);
          self.consume(")")?;
          Ok(Tree::Switch { arms, out })
        }
        // Var = Name
        _ => Ok(Tree::Var(self.parse_name()?)),
      }
    })
  }

  /// Name = /[a-zA-Z0-9_.$]+/
  fn parse_name(&mut self) -> Result<String, String> {
    let name = self.take_while(|c| c.is_alphanumeric() || c == '_' || c == '.' || c == '$');
    if name.is_empty() {
      return self.expected("name");
    }
    Ok(name.to_owned())
  }

  fn parse_num(&mut self) -> Result<Num, String> {
    self.skip_trivia();

    if self.peek_one() != Some('[') {
      return self.parse_num_lit();
    }

    self.consume("[");

    let Some(op) = NumTag::from_str_prefix(&self.input[self.index ..]) else { self.expected("operator")? };
    self.advance_many(op.as_str().len());

    self.skip_trivia();

    if op.is_ty() || self.peek_one() == Some(']') {
      self.consume("]");
      return Ok(Num::new_sym(op));
    }

    let lhs = self.parse_num_lit()?;

    self.consume("]");

    Ok(Num::new(op, lhs.payload()))
  }

  fn parse_num_lit(&mut self) -> Result<Num, String> {
    let num = self.take_while(|x| x.is_alphanumeric() || x == '+' || x == '-' || x == '.');
    if num.contains('.') || num.contains("inf") || num.contains("NaN") {
      let val: f32 = num.parse().map_err(|err| format!("{err:?}"))?;
      Ok(Num::new_f24(val))
    } else if num.starts_with('+') || num.starts_with('-') {
      let val = Self::parse_int(&num[1 ..])? as i32;
      Ok(Num::new_i24(if num.starts_with('-') { -val } else { val }))
    } else {
      let val = Self::parse_int(num)? as u32;
      Ok(Num::new_u24(val))
    }
  }

  fn parse_int(input: &str) -> Result<u64, String> {
    if let Some(rest) = input.strip_prefix("0x") {
      u64::from_str_radix(rest, 16).map_err(|err| format!("{err:?}"))
    } else if let Some(rest) = input.strip_prefix("0b") {
      u64::from_str_radix(rest, 2).map_err(|err| format!("{err:?}"))
    } else {
      input.parse::<u64>().map_err(|err| format!("{err:?}"))
    }
  }
}

/// Parses an unsigned integer with an optional radix prefix.
fn parse_int(input: &str) -> Result<u64, String> {
  if let Some(rest) = input.strip_prefix("0x") {
    u64::from_str_radix(rest, 16).map_err(|err| format!("{err:?}"))
  } else if let Some(rest) = input.strip_prefix("0b") {
    u64::from_str_radix(rest, 2).map_err(|err| format!("{err:?}"))
  } else {
    input.parse::<u64>().map_err(|err| format!("{err:?}"))
  }
}

/// Parses the input with the callback, ensuring that the whole input is
/// consumed.
fn parse_eof<'i, T>(input: &'i str, parse_fn: impl Fn(&mut Hvm64Parser<'i>) -> Result<T, String>) -> Result<T, String> {
  let mut parser = Hvm64Parser::new(input);
  let out = parse_fn(&mut parser)?;
  if parser.index != parser.input.len() {
    return Err("Unable to parse the whole input. Is this not an hvm64 file?".to_owned());
  }
  Ok(out)
}

impl FromStr for Book {
  type Err = String;
  fn from_str(str: &str) -> Result<Self, Self::Err> {
    parse_eof(str, Hvm64Parser::parse_book)
  }
}

impl FromStr for Net {
  type Err = String;
  fn from_str(str: &str) -> Result<Self, Self::Err> {
    parse_eof(str, Hvm64Parser::parse_net)
  }
}

impl FromStr for Tree {
  type Err = String;
  fn from_str(str: &str) -> Result<Self, Self::Err> {
    parse_eof(str, Hvm64Parser::parse_tree)
  }
}
