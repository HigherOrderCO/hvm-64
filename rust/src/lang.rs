// Syntax:
// <tree> ::= <era> | <nod> | <var> | <num>
// <era>  ::= "*"
// <nod>  ::= "(" <num_lit> " " <tree> " " <tree> ")"
// <var>  ::= <str_lit>
// <num>  ::= <num_lit>

use std::str::Chars;
use std::iter::Peekable;
use crate::core::*;

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
}

pub type Graph = Vec<(Tree,Tree)>;

pub fn parse_num_lit(chars: &mut Peekable<Chars>) -> u32 {
  let mut num = 0;
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
  while let Some(c) = chars.peek() {
    if !c.is_alphanumeric() {
      break;
    }
    str.push(*c);
    chars.next();
  }
  str
}

// Simple parser for the Tree type.

pub fn parse_tree(chars: &mut Peekable<Chars>) -> Tree {
  match chars.peek() {
    Some('*') => {
      chars.next();
      Tree::Era
    },
    Some('(') => {
      chars.next();
      let tag = CON + (parse_num_lit(chars) as u8);
      chars.next();
      let lft = Box::new(parse_tree(chars));
      chars.next();
      let rgt = Box::new(parse_tree(chars));
      chars.next();
      Tree::Nod { tag, lft, rgt }
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

// Conversion
// ----------

use std::collections::HashMap;

// Converts a number to an alphanumeric string.
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

// tag_to_port
pub fn tag_to_port(tag: Tag) -> Port {
  match tag {
    VR1 => Port::P1,
    VR2 => Port::P2,
    _   => panic!("Invalid tag for port conversion"),
  }
}

// port_to_tag
pub fn port_to_tag(port: Port) -> Tag {
  match port {
    Port::P1 => VR1,
    Port::P2 => VR2,
  }
}

pub fn make_tree(net: &mut Net, tree: &Tree, vars: &mut HashMap<String,(Loc,Port)>, loc: Loc, port: Port) -> Ptr {
  match tree {
    Tree::Era => {
      Ptr { tag: ERA, loc: 0 }
    },
    Tree::Nod { tag, lft, rgt } => {
      let loc = net.alloc();
      let p1 = make_tree(net, &*lft, vars, loc, Port::P1);
      net.set(loc, Port::P1, p1);
      let p2 = make_tree(net, &*rgt, vars, loc, Port::P2);
      net.set(loc, Port::P2, p2);
      Ptr { tag: *tag, loc }
    },
    Tree::Var { nam } => {
      if let Some((other_loc, other_port)) = vars.get(nam) {
        //println!("linked {} | set {} {:?} as {} {:?}", nam, other_loc, other_port, loc, port);
        net.set(*other_loc, *other_port, Ptr { tag: port_to_tag(port), loc });
        Ptr { tag: port_to_tag(*other_port), loc: *other_loc }
      } else {
        //println!("linkin {} | iam {} {:?}", nam, loc, port);
        vars.insert(nam.clone(), (loc, port));
        Ptr { tag: ERA, loc: 0 }
      }
    },
    Tree::Num { val } => {
      Ptr { tag: NUM, loc: *val }
    },
  }
}

pub fn do_make_tree(net: &mut Net, tree: &Tree) -> Ptr {
  make_tree(net, tree, &mut HashMap::new(), 0, Port::P1)
}

pub fn read_tree(net: &Net, ptr: Ptr, loc: Loc, port: Port, vars: &mut HashMap<(Loc,Port),String>, fresh: &mut usize) -> Tree {
  match ptr.tag {
    ERA => {
      Tree::Era
    },
    NUM => {
      Tree::Num { val: ptr.loc }
    },
    VR1 | VR2 => {
      if let Some(nam) = vars.get(&(ptr.loc, tag_to_port(ptr.tag))) {
        Tree::Var { nam: nam.clone() }
      } else {
        let nam = num_to_str(*fresh);
        *fresh += 1;
        vars.insert((loc, port), nam.clone());
        Tree::Var { nam }
      }
    },
    _ => {
      let lft = read_tree(net, net.get(ptr.loc, Port::P1), ptr.loc, Port::P1, vars, fresh);
      let rgt = read_tree(net, net.get(ptr.loc, Port::P2), ptr.loc, Port::P2, vars, fresh);
      Tree::Nod { tag: ptr.tag, lft: Box::new(lft), rgt: Box::new(rgt) }
    },
  }
}

// do_read_tree
pub fn do_read_tree(net: &Net, ptr: Ptr) -> Tree {
  read_tree(net, ptr, 0, Port::P1, &mut HashMap::new(), &mut 0)
}

pub fn read_graph(net: &Net) -> Graph {
  let mut vars = HashMap::new();
  let mut fresh = 0;
  let mut graph = Vec::new();
  for &(a, b) in &net.pair {
    let tree_a = read_tree(net, a, 0, Port::P1, &mut vars, &mut fresh);
    let tree_b = read_tree(net, b, 0, Port::P1, &mut vars, &mut fresh);
    graph.push((tree_a, tree_b));
  }
  return graph;
}

// Recursively copies a tree. Creates a fresh name for each variable.
pub fn copy_tree(tree: &Tree, new_name: &mut HashMap<String,String>, fresh: &mut usize) -> Tree {
  match tree {
    Tree::Era => {
      Tree::Era
    },
    Tree::Nod { tag, lft, rgt } => {
      let lft = Box::new(copy_tree(&*lft, new_name, fresh));
      let rgt = Box::new(copy_tree(&*rgt, new_name, fresh));
      Tree::Nod { tag: *tag, lft, rgt }
    },
    Tree::Var { nam } => {
      if let Some(new_nam) = new_name.get(nam) {
        Tree::Var { nam: new_nam.clone() }
      } else {
        let new_nam = num_to_str(*fresh);
        *fresh += 1;
        new_name.insert(nam.clone(), new_nam.clone());
        Tree::Var { nam: new_nam }
      }
    },
    Tree::Num { val } => {
      Tree::Num { val: *val }
    },
  }
}

pub fn do_copy_tree(tree: &Tree, fresh: &mut usize) -> Tree {
  copy_tree(tree, &mut HashMap::new(), fresh)
}

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
  }
}

// Shows all active pairs, one tree per line. Uses loops (not iter())
pub fn show_net(net: &Net) -> String {
  let mut result = String::new();
  for (a, b) in read_graph(net) {
    result.push_str(&format!(",-{}\n", show_tree(&a)));
    result.push_str(&format!("'-{}\n", show_tree(&b)));
  }
  return result;
}

pub fn arg(arg: Tree, ret: Tree) -> Tree {
  Tree::Nod {
    tag: CON,
    lft: Box::new(arg),
    rgt: Box::new(ret),
  }
}

pub fn num(val: u32) -> Tree {
  Tree::Num { val: val }
}
