// Syntax:
// <tree> ::= <era> | <nod> | <var> | <num>
// <era>  ::= "*"
// <nod>  ::= "(" <num_lit> " " <tree> " " <tree> ")"
// <var>  ::= <str_lit>
// <num>  ::= <num_lit>
// <expr> ::= <root> <acts>
// <root> ::= "$" <tree>
// <acts> ::= "&" <tree> "~" <tree> <expr>

use crate::core::*;
use std::collections::HashMap;
use std::iter::Peekable;
use std::str::Chars;

// AST
// ---

#[derive(Debug)]
pub enum LTree {
  Era,
  Nod { 
    tag: u8,
    lft: Box<LTree>,
    rgt: Box<LTree>,
  },
  Var {
    nam: String,
  },
  Num {
    val: u32
  },
  Ref {
    nam: u32
  }
}

type LActs = Vec<(LTree,LTree)>;

#[derive(Debug)]
pub struct LNet {
  root: LTree,
  acts: LActs,
}

// Parser
// ------

fn skip_spaces(chars: &mut Peekable<Chars>) {
  while let Some(c) = chars.peek() {
    if *c != ' ' && *c != '\n' {
      break;
    }
    chars.next();
  }
}

fn consume(chars: &mut Peekable<Chars>, text: &str) {
  skip_spaces(chars);
  for c in text.chars() {
    assert_eq!(chars.next().unwrap(), c);
  }
}

pub fn parse_num_lit(chars: &mut Peekable<Chars>) -> u32 {
  let mut num = 0;
  skip_spaces(chars);
  while let Some(c) = chars.peek() {
    if !c.is_digit(10) {
      break;
    }
    num = num * 10 + c.to_digit(10).unwrap();
    chars.next();
  }
  num
}

pub fn parse_str_lit(chars: &mut Peekable<Chars>) -> String {
  let mut str = String::new();
  skip_spaces(chars);
  while let Some(c) = chars.peek() {
    if !c.is_alphanumeric() {
      break;
    }
    str.push(*c);
    chars.next();
  }
  str
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
      let tag = CON + (parse_num_lit(chars) as u8);
      let lft = Box::new(parse_ltree(chars));
      let rgt = Box::new(parse_ltree(chars));
      consume(chars, ")");
      LTree::Nod { tag, lft, rgt }
    },
    Some('@') => {
      chars.next();
      skip_spaces(chars);
      LTree::Ref { nam: parse_num_lit(chars) }
    },
    Some(c) if c.is_digit(10) => {
      LTree::Num { val: parse_num_lit(chars) }
    },
    _ => {
      LTree::Var { nam: parse_str_lit(chars) }
    },
  }
}

pub fn parse_lnet(chars: &mut Peekable<Chars>) -> LNet {
  let mut acts = Vec::new();
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
      acts.push((tree1, tree2));
    } else {
      break;
    }
  }
  LNet { root, acts }
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
    LTree::Num { val } => {
      val.to_string()
    },
    LTree::Ref { nam } => {
      format!("@{}", nam.to_string())
    },
  }
}

pub fn show_lnet(lnet: &LNet) -> String {
  let mut result = String::new();
  result.push_str(&format!("$ {}\n", show_ltree(&lnet.root)));
  for (a, b) in &lnet.acts {
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
    let c = ((num % 26) as u8 + b'a' - 1) as char;
    str.push(c);
    num /= 26;
  }
  return str.chars().rev().collect();
}

pub fn tag_to_port(tag: Tag) -> Port {
  match tag {
    VR1 => Port::P1,
    VR2 => Port::P2,
    _   => unreachable!(),
  }
}

pub fn port_to_tag(port: Port) -> Tag {
  match port {
    Port::P1 => VR1,
    Port::P2 => VR2,
  }
}

pub fn lnet_to_lnet(lnet: &LNet) -> Net {
  let mut vars = HashMap::new();
  let mut net = Net::new(1 << 16);
  net.root = alloc_ltree(&mut net, &lnet.root, &mut vars, Parent::Root);
  for (tree1, tree2) in &lnet.acts {
    let ptr1 = alloc_ltree(&mut net, tree1, &mut vars, Parent::Acts);
    let ptr2 = alloc_ltree(&mut net, tree2, &mut vars, Parent::Acts);
    net.acts.push((ptr1, ptr2));
  }
  net.node = net.node[0 .. net.used].to_vec();
  return net;
}

// Injection and Readback
// ----------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Parent {
  Acts,
  Root,
  Node { loc: Loc, port: Port },
}

pub fn alloc_ltree(net: &mut Net, tree: &LTree, vars: &mut HashMap<String, Parent>, parent: Parent) -> Ptr {
  match tree {
    LTree::Era => {
      Ptr { tag: ERA, loc: 0 }
    },
    LTree::Nod { tag, lft, rgt } => {
      let loc = net.alloc(1);
      let p1 = alloc_ltree(net, &*lft, vars, Parent::Node { loc, port: Port::P1 });
      net.set(loc, Port::P1, p1);
      let p2 = alloc_ltree(net, &*rgt, vars, Parent::Node { loc, port: Port::P2 });
      net.set(loc, Port::P2, p2);
      Ptr { tag: *tag, loc }
    },
    LTree::Var { nam } => {
      if let Parent::Acts = parent {
        panic!("By definition, can't have variable on active pairs.");
      };
      match vars.get(nam) {
        Some(Parent::Acts) => {
          unreachable!();
        },
        Some(Parent::Root) => {
          match parent {
            Parent::Acts => {
              unreachable!();
            }
            Parent::Node { loc, port } => {
              net.root = Ptr { tag: port_to_tag(port), loc };
            }
            Parent::Root => {
              net.root = Ptr { tag: VRT, loc: 0 };
            }
          }
          return Ptr { tag: VRT, loc: 0 };
        },
        Some(Parent::Node { loc: other_loc, port: other_port }) => {
          //println!("linked {} | set {} {:?} as {} {:?}", nam, other_loc, other_port, loc, port);
          match parent {
            Parent::Acts => {
              unreachable!();
            }
            Parent::Node { loc, port } => {
              net.set(*other_loc, *other_port, Ptr { tag: port_to_tag(port), loc })
            }
            Parent::Root => {
              net.set(*other_loc, *other_port, Ptr { tag: VRT, loc: 0 })
            }
          }
          return Ptr { tag: port_to_tag(*other_port), loc: *other_loc };
        }
        None => {
          //println!("linkin {} | iam {} {:?}", nam, loc, port);
          vars.insert(nam.clone(), parent);
          Ptr { tag: NIL, loc: 0 }
        }
      }
    },
    LTree::Num { val } => {
      Ptr { tag: NUM, loc: *val }
    },
    LTree::Ref { nam } => {
      Ptr { tag: REF, loc: *nam }
    },
  }
}

pub fn do_alloc_ltree(net: &mut Net, tree: &LTree) -> Ptr {
  alloc_ltree(net, tree, &mut HashMap::new(), Parent::Root)
}

pub fn readback_ltree(net: &Net, ptr: Ptr, parent: Parent, vars: &mut HashMap<Parent,String>, fresh: &mut usize) -> LTree {
  match ptr.tag {
    NIL => {
      LTree::Era
    },
    ERA => {
      LTree::Era
    },
    NUM => {
      LTree::Num { val: ptr.loc }
    },
    REF => {
      LTree::Ref { nam: ptr.loc }
    },
    VRT | VR1 | VR2 => {
      let key = match ptr.tag {
        VR1 => Parent::Node { loc: ptr.loc, port: Port::P1 },
        VR2 => Parent::Node { loc: ptr.loc, port: Port::P2 },
        VRT => Parent::Root,
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
      let lft = readback_ltree(net, net.get(ptr.loc, Port::P1), Parent::Node { loc: ptr.loc, port: Port::P1 }, vars, fresh);
      let rgt = readback_ltree(net, net.get(ptr.loc, Port::P2), Parent::Node { loc: ptr.loc, port: Port::P2 }, vars, fresh);
      LTree::Nod { tag: ptr.tag, lft: Box::new(lft), rgt: Box::new(rgt) }
    },
  }
}

pub fn readback_lnet(net: &Net) -> LNet {
  let mut vars = HashMap::new();
  let mut fresh = 0;
  let mut acts = Vec::new();
  let root = readback_ltree(net, net.root, Parent::Root, &mut vars, &mut fresh);
  for &(a, b) in &net.acts {
    let tree_a = readback_ltree(net, a, Parent::Acts, &mut vars, &mut fresh);
    let tree_b = readback_ltree(net, b, Parent::Acts, &mut vars, &mut fresh);
    acts.push((tree_a, tree_b));
  }
  LNet { root, acts }
}

pub fn define(book: &mut Book, id: u32, code: &str) -> u32 {
  book.def(id, lnet_to_lnet(&do_parse_lnet(code)));
  return id;
}
