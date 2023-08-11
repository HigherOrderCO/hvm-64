// Syntax:
// <tree> ::= <era> | <nod> | <var> | <num>
// <era>  ::= "*"
// <nod>  ::= "(" <num_lit> " " <tree> " " <tree> ")"
// <var>  ::= <str_lit>
// <num>  ::= <num_lit>
// <expr> ::= <root> <acts>
// <root> ::= "$" <tree>
// <acts> ::= "&" <tree> "~" <tree> <expr>

use std::str::Chars;
use std::iter::Peekable;
use crate::core::*;

// AST
// ---

#[derive(Debug)]
pub enum Tree {
  Era,
  Nod { 
    tag: u8,
    lft: Box<Tree>,
    rgt: Box<Tree>,
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

type Graph = Vec<(Tree,Tree)>;

#[derive(Debug)]
pub struct Expr {
  root: Tree,
  acts: Graph,
}

// Parser
// ------

// Removes all leading spaces from chars
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

pub fn parse_tree(chars: &mut Peekable<Chars>) -> Tree {
  skip_spaces(chars);
  match chars.peek() {
    Some('*') => {
      chars.next();
      Tree::Era
    },
    Some('(') => {
      chars.next();
      let tag = CON + (parse_num_lit(chars) as u8);
      let lft = Box::new(parse_tree(chars));
      let rgt = Box::new(parse_tree(chars));
      consume(chars, ")");
      Tree::Nod { tag, lft, rgt }
    },
    Some('@') => {
      chars.next();
      skip_spaces(chars);
      Tree::Ref { nam: parse_num_lit(chars) }
    },
    Some(c) if c.is_digit(10) => {
      Tree::Num { val: parse_num_lit(chars) }
    },
    _ => {
      Tree::Var { nam: parse_str_lit(chars) }
    },
  }
}

pub fn do_parse(code: &str) -> Tree {
  parse_tree(&mut code.chars().peekable())
}

pub fn parse_expr(chars: &mut Peekable<Chars>) -> Expr {
  let mut acts = Vec::new();
  let mut root = Tree::Era;
  while let Some(c) = { skip_spaces(chars); chars.peek() } {
    if *c == '$' {
      chars.next();
      root = parse_tree(chars);
    } else if *c == '&' {
      chars.next();
      let tree1 = parse_tree(chars);
      consume(chars, "~");
      let tree2 = parse_tree(chars);
      acts.push((tree1, tree2));
    } else {
      break;
    }
  }
  Expr { root, acts }
}

pub fn do_parse_expr(code: &str) -> Expr {
  parse_expr(&mut code.chars().peekable())
}

// Stringifier
// -----------

pub fn show_tree(tree: &Tree) -> String {
  match tree {
    Tree::Era => {
      "*".to_string()
    },
    Tree::Nod { tag, lft, rgt } => {
      format!("({} {} {})", tag - CON, show_tree(&*lft), show_tree(&*rgt))
    },
    Tree::Var { nam } => {
      nam.clone()
    },
    Tree::Num { val } => {
      val.to_string()
    },
    Tree::Ref { nam } => {
      format!("@{}", nam.to_string())
    },
  }
}

// Shows all active pairs, one tree per line. Uses loops (not iter())
pub fn show_net(net: &Net) -> String {
  let mut result = String::new();
  let expr = read_expr(net);
  result.push_str(&format!("$ {}\n", show_tree(&expr.root)));
  for (a, b) in &expr.acts {
    result.push_str(&format!("& {}\n~ {}\n", show_tree(a), show_tree(b)));
  }
  return result;
}

pub fn show_term(term: &Term) -> String {
  let mut net = Net::new(1 << 16);
  net.term.root = term.root.clone();
  net.term.acts = term.acts.clone();
  net.term.node = term.node.clone();
  return show_net(&net);
}

// Conversion
// ----------

use std::collections::HashMap;

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
    _   => panic!("Invalid tag for port conversion"),
  }
}

pub fn port_to_tag(port: Port) -> Tag {
  match port {
    Port::P1 => VR1,
    Port::P2 => VR2,
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Parent {
  Acts,
  Root,
  Node { loc: Loc, port: Port },
}

pub fn alloc_tree(net: &mut Net, tree: &Tree, vars: &mut HashMap<String, Parent>, parent: Parent) -> Ptr {
  match tree {
    Tree::Era => {
      Ptr { tag: ERA, loc: 0 }
    },
    Tree::Nod { tag, lft, rgt } => {
      let loc = net.alloc(1);
      let p1 = alloc_tree(net, &*lft, vars, Parent::Node { loc, port: Port::P1 });
      net.set(loc, Port::P1, p1);
      let p2 = alloc_tree(net, &*rgt, vars, Parent::Node { loc, port: Port::P2 });
      net.set(loc, Port::P2, p2);
      Ptr { tag: *tag, loc }
    },
    Tree::Var { nam } => {
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
              net.term.root = Ptr { tag: port_to_tag(port), loc };
            }
            Parent::Root => {
              net.term.root = Ptr { tag: VRR, loc: 0 };
            }
          }
          return Ptr { tag: VRR, loc: 0 };
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
              net.set(*other_loc, *other_port, Ptr { tag: VRR, loc: 0 })
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
    Tree::Num { val } => {
      Ptr { tag: NUM, loc: *val }
    },
    Tree::Ref { nam } => {
      Ptr { tag: REF, loc: *nam }
    },
  }
}

pub fn do_alloc_tree(net: &mut Net, tree: &Tree) -> Ptr {
  alloc_tree(net, tree, &mut HashMap::new(), Parent::Root)
}

pub fn read_tree(net: &Net, ptr: Ptr, parent: Parent, vars: &mut HashMap<Parent,String>, fresh: &mut usize) -> Tree {
  match ptr.tag {
    NIL => {
      Tree::Era
    },
    ERA => {
      Tree::Era
    },
    NUM => {
      Tree::Num { val: ptr.loc }
    },
    REF => {
      Tree::Ref { nam: ptr.loc }
    },
    VRR | VR1 | VR2 => {
      let key = match ptr.tag {
        VR1 => Parent::Node { loc: ptr.loc, port: Port::P1 },
        VR2 => Parent::Node { loc: ptr.loc, port: Port::P2 },
        VRR => Parent::Root,
        _   => unreachable!(),
      };
      if let Some(nam) = vars.get(&key) {
        Tree::Var { nam: nam.clone() }
      } else {
        let nam = num_to_str(*fresh);
        *fresh += 1;
        vars.insert(parent, nam.clone());
        Tree::Var { nam }
      }
    },
    _ => {
      let lft = read_tree(net, net.get(ptr.loc, Port::P1), Parent::Node { loc: ptr.loc, port: Port::P1 }, vars, fresh);
      let rgt = read_tree(net, net.get(ptr.loc, Port::P2), Parent::Node { loc: ptr.loc, port: Port::P2 }, vars, fresh);
      Tree::Nod { tag: ptr.tag, lft: Box::new(lft), rgt: Box::new(rgt) }
    },
  }
}

pub fn read_expr(net: &Net) -> Expr {
  let mut vars = HashMap::new();
  let mut fresh = 0;
  let mut acts = Vec::new();
  let root = read_tree(net, net.term.root, Parent::Root, &mut vars, &mut fresh);
  for &(a, b) in &net.term.acts {
    let tree_a = read_tree(net, a, Parent::Acts, &mut vars, &mut fresh);
    let tree_b = read_tree(net, b, Parent::Acts, &mut vars, &mut fresh);
    acts.push((tree_a, tree_b));
  }
  Expr { root, acts }
}

// TODO: avoid needing an intermediate net
pub fn expr_to_term(expr: &Expr) -> Term {
  let mut vars = HashMap::new();
  let mut net = Net::new(1 << 16);
  net.term.root = alloc_tree(&mut net, &expr.root, &mut vars, Parent::Root);
  for (tree1, tree2) in &expr.acts {
    let ptr1 = alloc_tree(&mut net, tree1, &mut vars, Parent::Acts);
    let ptr2 = alloc_tree(&mut net, tree2, &mut vars, Parent::Acts);
    net.term.acts.push((ptr1, ptr2));
  }
  return Term {
    root: net.term.root,
    acts: net.term.acts,
    node: net.term.node[0 .. net.used].to_vec(),
  };
}

pub fn define(net: &mut Net, id: u32, code: &str) -> u32 {
  //println!("--- define {}", id);
  //println!("--- EXPR:");
  let expr = do_parse_expr(code);
  //println!("{:?}", expr);
  //println!("--- TERM:");
  let term = expr_to_term(&expr);
  //println!("{:?}", term);
  //println!("{}", show_term(&term));
  net.defs.insert(id, term);
  return id;
}
