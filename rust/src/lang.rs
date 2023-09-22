// An interaction combinator language
// ----------------------------------
//
// This file implements a textual syntax to interact with the runtime. It includes a pure AST for
// nets, as well as functions for parsing, stringifying, and converting pure ASTs to runtime nets.
// On the runtime, a net is represented by a list of active trees, plus a root tree. The textual
// syntax reflects this representation. It is specified below:
//
// <net>    ::= <root> <rdex>
//   <root> ::= "$" <tree>
//   <rdex> ::= "&" <tree> "~" <tree> <net>
// <tree>   ::= <era> | <nod> | <var> | <dec>
//   <era>  ::= "*"
//   <nod>  ::= "(" <decimal> " " <tree> " " <tree> ")"
//   <var>  ::= <string>
//   <u32>  ::= <decimal>
//   <i32>  ::= <sign> <decimal>
//   <ref>  ::= "@" <string>
//   <op2>  ::= "{" <op2_lit> " " <tree> " " <tree> "}"
// <sign>   ::= "+" | "-"
//
// For example, below is the church nat 2, encoded as an interaction net:
//
// $ (0 (1 (0 b a) (0 a R)) (0 b R))
//
// The '$' symbol denotes the net's root. A node is denoted as `(LABEL CHILD_1 CHILD_2)`.
// The label 0 is used for CON nodes, while labels >1 are used for DUP nodes. A node has two
// children, representing Port1->Port0 and Port2->Port0 wires. Variables are denoted by
// alphanumeric names, and used to represent auxiliary wires (Port1->Port2 and Port2->Port1).
// Active wires (Port0->Port0) are represented with by '& left_tree ~ right_tree'. For example:
//
// & (0 x x)
// ~ (0 y y)
//
// The net above represents two identity CON nodes connected by their main ports. This net has no
// root, so it will just reduce to nothingness. Numbers are represented by numeric literals.
// References (to closed nets) are denoted by '@name', where the name must have at most 5 letters,
// from the following alphabet: ".0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_".

use crate::core::*;
use std::collections::HashMap;
use std::iter::Peekable;
use std::str::Chars;

// AST
// ---

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum OP {
  ADD,
  SUB,
  MUL,
  DIV,
  MOD,
  EQ ,
  NEQ,
  LT ,
  GT ,
  LTE,
  GTE,
  AND,
  OR ,
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum LTree {
  Era,
  Nod { 
    tag: Tag,
    lft: Box<LTree>,
    rgt: Box<LTree>,
  },
  Var {
    nam: String,
  },
  Ref {
    nam: Val
  },
  U32 {
    val: u32
  },
  I32 {
    val: i32
  },
  OpX {
    opx: OP,
    lft: Box<LTree>,
    rgt: Box<LTree>,
  },
}

type LRdex = Vec<(LTree,LTree)>;

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct LNet {
  pub root: LTree,
  pub rdex: LRdex,
}

// Parser
// ------

pub fn skip_spaces(chars: &mut Peekable<Chars>) {
  while let Some(c) = chars.peek() {
    if *c != ' ' && *c != '\n' {
      break;
    }
    chars.next();
  }
}

pub fn consume(chars: &mut Peekable<Chars>, text: &str) {
  skip_spaces(chars);
  for c in text.chars() {
    assert_eq!(chars.next().unwrap(), c);
  }
}

pub fn parse_decimal(chars: &mut Peekable<Chars>) -> u64 {
  let mut num : Val = 0;
  skip_spaces(chars);
  while let Some(c) = chars.peek() {
    if !c.is_digit(10) {
      break;
    }
    num = num * 10 + c.to_digit(10).unwrap() as Val;
    chars.next();
  }
  num
}

pub fn parse_string(chars: &mut Peekable<Chars>) -> String {
  let mut str = String::new();
  skip_spaces(chars);
  while let Some(c) = chars.peek() {
    if !c.is_alphanumeric() && *c != '_' && *c != '.' {
      break;
    }
    str.push(*c);
    chars.next();
  }
  str
}

pub fn parse_opx_lit(chars: &mut Peekable<Chars>) -> String {
  let mut opx = String::new();
  skip_spaces(chars);
  while let Some(c) = chars.peek() {
    if !"+-=*/%<>|&^!?".contains(*c) {
      break;
    }
    opx.push(*c);
    chars.next();
  }
  opx
}

pub fn parse_ltree(chars: &mut Peekable<Chars>) -> LTree {
  skip_spaces(chars);
  match chars.peek() {
    Some('*') => {
      chars.next();
      LTree::Era
    },
    Some('(') => {
      chars.next();
      let tag = CON + (parse_decimal(chars) as Tag);
      let lft = Box::new(parse_ltree(chars));
      let rgt = Box::new(parse_ltree(chars));
      consume(chars, ")");
      LTree::Nod { tag, lft, rgt }
    },
    Some('@') => {
      chars.next();
      skip_spaces(chars);
      let name = parse_string(chars);
      LTree::Ref { nam: name_to_val(&name) }
    },
    Some(c) if c.is_digit(10) => {
      LTree::U32 { val: parse_decimal(chars) as u32 }
    },
    Some(s) if *s == '+' || *s == '-' => {
      let s = *s;
      chars.next();
      LTree::I32 { val: parse_decimal(chars) as i32 * (if s == '-' { -1 } else { 1 }) }
    },
    Some('{') => {
      chars.next();
      let opx = string_to_opx(&parse_opx_lit(chars));
      let lft = Box::new(parse_ltree(chars));
      let rgt = Box::new(parse_ltree(chars));
      consume(chars, "}");
      LTree::OpX { opx, lft, rgt }
    },
    _ => {
      LTree::Var { nam: parse_string(chars) }
    },
  }
}

pub fn parse_lnet(chars: &mut Peekable<Chars>) -> LNet {
  let mut rdex = Vec::new();
  let mut root = LTree::Era;
  while let Some(c) = { skip_spaces(chars); chars.peek() } {
    if *c == '$' {
      chars.next();
      root = parse_ltree(chars);
    } else if *c == '&' {
      chars.next();
      let tree1 = parse_ltree(chars);
      consume(chars, "~");
      let tree2 = parse_ltree(chars);
      rdex.push((tree1, tree2));
    } else {
      break;
    }
  }
  LNet { root, rdex }
}

pub fn do_parse_ltree(code: &str) -> LTree {
  parse_ltree(&mut code.chars().peekable())
}

pub fn do_parse_lnet(code: &str) -> LNet {
  parse_lnet(&mut code.chars().peekable())
}

// Stringifier
// -----------

pub fn show_ltree(tree: &LTree) -> String {
  match tree {
    LTree::Era => {
      "*".to_string()
    },
    LTree::Nod { tag, lft, rgt } => {
      format!("({} {} {})", tag - CON, show_ltree(&*lft), show_ltree(&*rgt))
    },
    LTree::Var { nam } => {
      nam.clone()
    },
    LTree::Ref { nam } => {
      format!("@{}", val_to_name(*nam))
    },
    LTree::U32 { val } => {
      format!("{}", (*val as u32).to_string())
    },
    LTree::I32 { val } => {
      format!("{}{}", if *val >= 0 { "+" } else { "" }, (*val as i32).to_string())
    },
    LTree::OpX { opx, lft, rgt } => {
      format!("{{{} {} {}}}", opx_to_string(opx), show_ltree(&*lft), show_ltree(&*rgt))
    },
  }
}

pub fn show_lnet(lnet: &LNet) -> String {
  let mut result = String::new();
  result.push_str(&format!("$ {}\n", show_ltree(&lnet.root)));
  for (a, b) in &lnet.rdex {
    result.push_str(&format!("& {}\n~ {}\n", show_ltree(a), show_ltree(b)));
  }
  return result;
}

pub fn show_net(net: &Net) -> String {
  show_lnet(&readback_lnet(net))
}

// Conversion
// ----------

pub fn num_to_str(num: usize) -> String {
  let mut num = num + 1;
  let mut str = String::new();
  while num > 0 {
    let c = ((num % 26) as u8 + b'a') as char;
    str.push(c);
    num /= 26;
  }
  return str.chars().rev().collect();
}

pub fn tag_to_port(tag: Tag) -> Port {
  match tag {
    VR1 => P1,
    VR2 => P2,
    _   => unreachable!(),
  }
}

pub fn port_to_tag(port: Port) -> Tag {
  match port {
    P1 => VR1,
    P2 => VR2,
    _  => unreachable!(),
  }
}

pub fn lnet_to_net(lnet: &LNet, size: Option<usize>) -> Net {
  let mut vars = HashMap::new();
  let mut net = Net::new(size.unwrap_or(1 << 16));
  net.root = alloc_ltree(&mut net, &lnet.root, &mut vars, Parent::Root);
  for (tree1, tree2) in &lnet.rdex {
    let ptr1 = alloc_ltree(&mut net, tree1, &mut vars, Parent::Rdex);
    let ptr2 = alloc_ltree(&mut net, tree2, &mut vars, Parent::Rdex);
    net.rdex.push((ptr1, ptr2));
  }
  if size.is_none() {
    net.node = net.node[0 .. net.used].to_vec();
  }
  return net;
}

pub fn name_to_letters(name: &str) -> Vec<u8> {
  let mut letters = Vec::new();
  for c in name.chars() {
    letters.push(match c {
      '0'..='9' => c as u8 - '0' as u8 + 0,
      'A'..='Z' => c as u8 - 'A' as u8 + 10,
      'a'..='z' => c as u8 - 'a' as u8 + 36,
      '_'       => 62,
      '.'       => 63,
      _         => panic!("Invalid character in name"),
    });
  }
  return letters;
}

pub fn letters_to_name(letters: Vec<u8>) -> String {
  let mut name = String::new();
  for letter in letters {
    name.push(match letter {
       0..=9  => (letter -  0 + '0' as u8) as char,
      10..=35 => (letter - 10 + 'A' as u8) as char,
      36..=61 => (letter - 36 + 'a' as u8) as char,
           62 => '_',
           63 => '.',
      _       => panic!("Invalid letter in name"),
    });
  }
  return name;
}

pub fn val_to_letters(num: Val) -> Vec<u8> {
  let mut letters = Vec::new();
  let mut num = num;
  while num > 0 {
    letters.push((num % 64) as u8);
    num /= 64;
  }
  letters.reverse();
  return letters;
}

pub fn letters_to_val(letters: Vec<u8>) -> Val {
  let mut num = 0;
  for letter in letters {
    num = num * 64 + letter as Val;
  }
  return num;
}

pub fn name_to_val(name: &str) -> Val {
  letters_to_val(name_to_letters(name))
}

pub fn val_to_name(num: Val) -> String {
  letters_to_name(val_to_letters(num))
}

pub fn string_to_opx(opx: &str) -> OP {
  match opx {
    "+"  => OP::ADD,
    "-"  => OP::SUB,
    "*"  => OP::MUL,
    "/"  => OP::DIV,
    "%"  => OP::MOD,
    "==" => OP::EQ,
    "!=" => OP::NEQ,
    "<"  => OP::LT,
    ">"  => OP::GT,
    "<=" => OP::LTE,
    ">=" => OP::GTE,
    "&&" => OP::AND,
    "||" => OP::OR,
    _    => panic!("Invalid operator"),
  }
}

pub fn opx_to_string(opx: &OP) -> String {
  match opx {
    OP::ADD => "+".to_string(),
    OP::SUB => "-".to_string(),
    OP::MUL => "*".to_string(),
    OP::DIV => "/".to_string(),
    OP::MOD => "%".to_string(),
    OP::EQ  => "==".to_string(),
    OP::NEQ => "!=".to_string(),
    OP::LT  => "<".to_string(),
    OP::GT  => ">".to_string(),
    OP::LTE => "<=".to_string(),
    OP::GTE => ">=".to_string(),
    OP::AND => "&&".to_string(),
    OP::OR  => "||".to_string(),
  }
}

pub fn opx_to_tag(opx: &OP) -> Tag {
  match opx {
    OP::ADD => OPX_ADD,
    OP::SUB => OPX_SUB,
    OP::MUL => OPX_MUL,
    OP::DIV => OPX_DIV,
    OP::MOD => OPX_MOD,
    OP::EQ  => OPX_EQ ,
    OP::NEQ => OPX_NEQ,
    OP::LT  => OPX_LT ,
    OP::GT  => OPX_GT ,
    OP::LTE => OPX_LTE,
    OP::GTE => OPX_GTE,
    OP::AND => OPX_AND,
    OP::OR  => OPX_OR ,
  }
}

pub fn tag_to_opx(tag: Tag) -> OP {
  match tag {
    OPX_ADD => OP::ADD,
    OPX_SUB => OP::SUB,
    OPX_MUL => OP::MUL,
    OPX_DIV => OP::DIV,
    OPX_MOD => OP::MOD,
    OPX_EQ  => OP::EQ ,
    OPX_NEQ => OP::NEQ,
    OPX_LT  => OP::LT ,
    OPX_GT  => OP::GT ,
    OPX_LTE => OP::LTE,
    OPX_GTE => OP::GTE,
    OPX_AND => OP::AND,
    OPX_OR  => OP::OR ,
    _       => panic!("Invalid operator"),
  }
}

// Injection and Readback
// ----------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Parent {
  Rdex,
  Root,
  Node { val: Val, port: Port },
}

pub fn alloc_ltree(net: &mut Net, tree: &LTree, vars: &mut HashMap<String, Parent>, parent: Parent) -> Ptr {
  match tree {
    LTree::Era => {
      Ptr::new(ERA, 0)
    },
    LTree::Nod { tag, lft, rgt } => {
      let val = net.alloc(1);
      let p1 = alloc_ltree(net, &*lft, vars, Parent::Node { val, port: P1 });
      net.set(val, P1, p1);
      let p2 = alloc_ltree(net, &*rgt, vars, Parent::Node { val, port: P2 });
      net.set(val, P2, p2);
      Ptr::new(*tag, val)
    },
    LTree::Var { nam } => {
      if let Parent::Rdex = parent {
        panic!("By definition, can't have variable on active pairs.");
      };
      match vars.get(nam) {
        Some(Parent::Rdex) => {
          unreachable!();
        },
        Some(Parent::Root) => {
          match parent {
            Parent::Rdex => {
              unreachable!();
            }
            Parent::Node { val, port } => {
              net.root = Ptr::new(port_to_tag(port), val);
            }
            Parent::Root => {
              net.root = Ptr::new(VRR, 0);
            }
          }
          return Ptr::new(VRR, 0);
        },
        Some(Parent::Node { val: other_val, port: other_port }) => {
          //println!("linked {} | set {} {:?} as {} {:?}", nam, other_val, other_port, val, port);
          match parent {
            Parent::Rdex => {
              unreachable!();
            }
            Parent::Node { val, port } => {
              net.set(*other_val, *other_port, Ptr::new(port_to_tag(port), val))
            }
            Parent::Root => {
              net.set(*other_val, *other_port, Ptr::new(VRR, 0))
            }
          }
          return Ptr::new(port_to_tag(*other_port), *other_val);
        }
        None => {
          //println!("linkin {} | iam {} {:?}", nam, val, port);
          vars.insert(nam.clone(), parent);
          Ptr::new(NIL, 0)
        }
      }
    },
    LTree::Ref { nam } => {
      Ptr::new(REF, *nam)
    },
    LTree::U32 { val } => {
      Ptr::new(U32, *val as Val)
    },
    LTree::I32 { val } => {
      Ptr::new(I32, *val as Val)
    },
    LTree::OpX { opx, lft, rgt } => {
      let tag = opx_to_tag(opx);
      let val = net.alloc(1);
      let p1 = alloc_ltree(net, &*lft, vars, Parent::Node { val, port: P1 });
      net.set(val, P1, p1);
      let p2 = alloc_ltree(net, &*rgt, vars, Parent::Node { val, port: P2 });
      net.set(val, P2, p2);
      Ptr::new(tag, val)
    },
  }
}

pub fn do_alloc_ltree(net: &mut Net, tree: &LTree) -> Ptr {
  alloc_ltree(net, tree, &mut HashMap::new(), Parent::Root)
}

pub fn readback_ltree(net: &Net, ptr: Ptr, parent: Parent, vars: &mut HashMap<Parent,String>, fresh: &mut usize) -> LTree {
  match ptr.tag() {
    NIL => {
      LTree::Var { nam: "?".to_string() }
    },
    ERA => {
      LTree::Era
    },
    REF => {
      LTree::Ref { nam: ptr.val() }
    },
    U32 => {
      LTree::U32 { val: ptr.val() as u32 }
    },
    I32 => {
      LTree::I32 { val: ptr.val() as i32 }
    },
    MIN_OPX..=MAX_OPX => {
      let lft = readback_ltree(net, net.get(ptr.val(), P1), Parent::Node { val: ptr.val(), port: P1 }, vars, fresh);
      let rgt = readback_ltree(net, net.get(ptr.val(), P2), Parent::Node { val: ptr.val(), port: P2 }, vars, fresh);
      LTree::OpX { opx: tag_to_opx(ptr.tag()), lft: Box::new(lft), rgt: Box::new(rgt) }
    },
    MIN_OPY..=MAX_OPY => {
      todo!()
    },
    VRR | VR1 | VR2 => {
      let key = match ptr.tag() {
        VR1 => Parent::Node { val: ptr.val(), port: P1 },
        VR2 => Parent::Node { val: ptr.val(), port: P2 },
        VRR => Parent::Root,
        _   => unreachable!(),
      };
      if let Some(nam) = vars.get(&key) {
        LTree::Var { nam: nam.clone() }
      } else {
        let nam = num_to_str(*fresh);
        *fresh += 1;
        vars.insert(parent, nam.clone());
        LTree::Var { nam }
      }
    },
    _ => {
      let lft = readback_ltree(net, net.get(ptr.val(), P1), Parent::Node { val: ptr.val(), port: P1 }, vars, fresh);
      let rgt = readback_ltree(net, net.get(ptr.val(), P2), Parent::Node { val: ptr.val(), port: P2 }, vars, fresh);
      LTree::Nod { tag: ptr.tag(), lft: Box::new(lft), rgt: Box::new(rgt) }
    },
  }
}

pub fn readback_lnet(net: &Net) -> LNet {
  let mut vars  = HashMap::new();
  let mut fresh = 0;
  let mut rdex  = Vec::new();
  let root = readback_ltree(net, net.root, Parent::Root, &mut vars, &mut fresh);
  for &(a, b) in &net.rdex {
    let tree_a = readback_ltree(net, a, Parent::Rdex, &mut vars, &mut fresh);
    let tree_b = readback_ltree(net, b, Parent::Rdex, &mut vars, &mut fresh);
    rdex.push((tree_a, tree_b));
  }
  LNet { root, rdex }
}

// Utils
// -----

pub fn define(book: &mut Book, name: &str, code: &str) -> Val {
  let id = name_to_val(name);
  book.def(id, lnet_to_net(&do_parse_lnet(code), None));
  return id;
}
