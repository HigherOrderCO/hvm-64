use crate::prelude::*;

use alloc::collections::BTreeMap;
use core::str::FromStr;

use crate::{Book, Lab, Net, Tree, MAX_ADT_FIELDS, MAX_ADT_VARIANTS, MAX_ARITY};
use hvm64_util::{maybe_grow, ops::TypedOp as Op};

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

  fn parse_tree(&mut self) -> Result<Tree, String> {
    maybe_grow(move || {
      self.skip_trivia();
      match self.peek_one() {
        // Era = "*"
        Some('*') => {
          self.advance_one();
          Ok(Tree::Era)
        }
        // Ctr = "(" Tree Tree ")" | "[" Tree Tree "]" | "{" Int Tree Tree "}"
        Some(char @ ('(' | '[' | '{')) => {
          self.advance_one();
          let lab = match char {
            '(' => 0,
            '[' => 1,
            '{' => self.parse_u64()? as Lab,
            _ => unreachable!(),
          };
          let close = match char {
            '(' => ')',
            '[' => ']',
            '{' => '}',
            _ => unreachable!(),
          };
          self.skip_trivia();
          if self.peek_one().is_some_and(|x| x == ':') {
            self.advance_one();
            let variant_index = self.parse_u64()?;
            self.consume(":")?;
            let variant_count = self.parse_u64()?;
            let mut fields = Vec::new();
            self.skip_trivia();
            while self.peek_one() != Some(close) {
              fields.push(self.parse_tree()?);
              self.skip_trivia();
            }
            self.advance_one();
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
            while self.peek_one() != Some(close) {
              ports.push(self.parse_tree()?);
              self.skip_trivia();
            }
            self.advance_one();
            if ports.len() > MAX_ARITY {
              Err("ctr has too many ports".to_owned())?;
            }
            Ok(Tree::Ctr { lab, ports })
          }
        }
        // Ref = "@" Name
        Some('@') => {
          self.advance_one();
          self.skip_trivia();
          let nam = self.parse_name()?;
          Ok(Tree::Ref { nam })
        }
        // Int = "#" [-] Int
        // F32 = "#" [-] ( Int "." Int | "NaN" | "inf" )
        Some('#') => {
          self.advance_one();
          let is_neg = self.consume("-").is_ok();
          let num = self.take_while(|c| c.is_alphanumeric() || c == '.');

          if num.contains('.') || num.contains("NaN") || num.contains("inf") {
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
          self.advance_one();
          let op = self.parse_op()?;
          let rhs = Box::new(self.parse_tree()?);
          let out = Box::new(self.parse_tree()?);
          self.consume(">")?;
          Ok(Tree::Op { op, rhs, out })
        }
        // Mat = "?<" Tree Tree ">"
        Some('?') => {
          self.advance_one();
          self.consume("<")?;
          let zero = self.parse_tree()?;
          let succ = self.parse_tree()?;
          self.skip_trivia();
          if self.peek_one() == Some('>') {
            self.advance_one();
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
      return self.expected("name");
    }
    Ok(name.to_owned())
  }

  /// See `ops.rs` for the available operators.
  fn parse_op(&mut self) -> Result<Op, String> {
    let op = self.take_while(|c| c.is_alphanumeric() || ".+-=*/%<>|&^!?$".contains(c));
    op.parse().map_err(|_| format!("Unknown operator: {op:?}"))
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
