use crate::core::*;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::iter::Peekable;
use std::str::Chars;

// AST
// ---

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum LTree {
  Era,
  Ctr { lab: u8, lft: Box<LTree>, rgt: Box<LTree> },
  Var { nam: String },
  Ref { nam: Val },
  Num { val: u32 },
  Op2 { lft: Box<LTree>, rgt: Box<LTree> },
  Ite { sel: Box<LTree>, ret: Box<LTree> },
}

type LRdex = Vec<(LTree, LTree)>;

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct LNet {
  pub root: LTree,
  pub rdex: LRdex,
}

type LBook = BTreeMap<String, LNet>;

// Parser
// ------

fn skip(chars: &mut Peekable<Chars>) {
  while let Some(c) = chars.peek() {
    if *c == '/' {
      chars.next();
      while let Some(c) = chars.peek() {
        if *c == '\n' {
          break;
        }
        chars.next();
      }
    } else if *c != ' ' && *c != '\n' {
      break;
    } else {
      chars.next();
    }
  }
}

pub fn consume(chars: &mut Peekable<Chars>, text: &str) -> Result<(), String> {
  skip(chars);
  for c in text.chars() {
    if chars.next() != Some(c) {
      return Err(format!("Expected '{}', found {:?}", text, chars.peek()));
    }
  }
  return Ok(());
}

pub fn parse_decimal(chars: &mut Peekable<Chars>) -> Result<u32, String> {
  let mut num: u32 = 0;
  skip(chars);
  if !chars.peek().map_or(false, |c| c.is_digit(10)) {
    return Err(format!("Expected a decimal number, found {:?}", chars.peek()));
  }
  while let Some(c) = chars.peek() {
    if !c.is_digit(10) {
      break;
    }
    num = num * 10 + c.to_digit(10).unwrap() as u32;
    chars.next();
  }
  Ok(num)
}

pub fn parse_name(chars: &mut Peekable<Chars>) -> Result<String, String> {
  let mut txt = String::new();
  skip(chars);
  if !chars.peek().map_or(false, |c| c.is_alphanumeric() || *c == '_' || *c == '.') {
    return Err(format!("Expected a name character, found {:?}", chars.peek()))
  }
  while let Some(c) = chars.peek() {
    if !c.is_alphanumeric() && *c != '_' && *c != '.' {
      break;
    }
    txt.push(*c);
    chars.next();
  }
  Ok(txt)
}

pub fn parse_opx_lit(chars: &mut Peekable<Chars>) -> Result<String, String> {
  let mut opx = String::new();
  skip(chars);
  while let Some(c) = chars.peek() {
    if !"+-=*/%<>|&^!?".contains(*c) {
      break;
    }
    opx.push(*c);
    chars.next();
  }
  Ok(opx)
}

pub fn parse_ltree(chars: &mut Peekable<Chars>) -> Result<LTree, String> {
  skip(chars);
  match chars.peek() {
    Some('*') => {
      chars.next();
      Ok(LTree::Era)
    }
    Some('(') => {
      chars.next();
      let lab = 0;
      let lft = Box::new(parse_ltree(chars)?);
      let rgt = Box::new(parse_ltree(chars)?);
      consume(chars, ")")?;
      Ok(LTree::Ctr { lab, lft, rgt })
    }
    Some('[') => {
      chars.next();
      let lab = 1;
      let lft = Box::new(parse_ltree(chars)?);
      let rgt = Box::new(parse_ltree(chars)?);
      consume(chars, "]")?;
      Ok(LTree::Ctr { lab, lft, rgt })
    }
    Some('{') => {
      chars.next();
      let lab = parse_decimal(chars)? as u8;
      let lft = Box::new(parse_ltree(chars)?);
      let rgt = Box::new(parse_ltree(chars)?);
      consume(chars, "}")?;
      Ok(LTree::Ctr { lab, lft, rgt })
    }
    Some('@') => {
      chars.next();
      skip(chars);
      let name = parse_name(chars)?;
      Ok(LTree::Ref { nam: name_to_val(&name) })
    }
    Some('#') => {
      chars.next();
      Ok(LTree::Num { val: parse_decimal(chars)? as u32 })
    }
    Some('<') => {
      chars.next();
      let lft = Box::new(parse_ltree(chars)?);
      let rgt = Box::new(parse_ltree(chars)?);
      consume(chars, ">")?;
      Ok(LTree::Op2 { lft, rgt })
    }
    Some('?') => {
      chars.next();
      let sel = Box::new(parse_ltree(chars)?);
      let ret = Box::new(parse_ltree(chars)?);
      Ok(LTree::Ite { sel, ret })
    }
    _ => {
      Ok(LTree::Var { nam: parse_name(chars)? })
    },
  }
}

pub fn parse_lnet(chars: &mut Peekable<Chars>) -> Result<LNet, String> {
  let mut rdex = Vec::new();
  let root = parse_ltree(chars)?;
  while let Some(c) = { skip(chars); chars.peek() } {
    if *c == '&' {
      chars.next();
      let tree1 = parse_ltree(chars)?;
      consume(chars, "~")?;
      let tree2 = parse_ltree(chars)?;
      rdex.push((tree1, tree2));
    } else {
      break;
    }
  }
  Ok(LNet { root, rdex })
}

pub fn parse_lbook(chars: &mut Peekable<Chars>) -> Result<LBook, String> {
  let mut book = BTreeMap::new();
  while let Some(c) = { skip(chars); chars.peek() } {
    if *c == '@' {
      chars.next();
      let name = parse_name(chars)?;
      consume(chars, "=")?;
      let net = parse_lnet(chars)?;
      book.insert(name, net);
    } else {
      break;
    }
  }
  Ok(book)
}

fn do_parse<T>(code: &str, parse_fn: impl Fn(&mut Peekable<Chars>) -> Result<T, String>) -> T {
  match parse_fn(&mut code.chars().peekable()) {
    Ok(result) => result,
    Err(err) => {
      eprintln!("{}", err);
      std::process::exit(1);
    }
  }
}

pub fn do_parse_ltree(code: &str) -> LTree {
  do_parse(code, parse_ltree)
}

pub fn do_parse_lnet(code: &str) -> LNet {
  do_parse(code, parse_lnet)
}

pub fn do_parse_lbook(code: &str) -> LBook {
  do_parse(code, parse_lbook)
}

// Stringifier
// -----------

pub fn show_ltree(tree: &LTree) -> String {
  match tree {
    LTree::Era => {
      "*".to_string()
    }
    LTree::Ctr { lab, lft, rgt } => {
      match lab {
        0 => { format!("({} {})", show_ltree(&*lft), show_ltree(&*rgt)) }
        1 => { format!("[{} {}]", show_ltree(&*lft), show_ltree(&*rgt)) }
        _ => { format!("{{{} {} {}}}", lab, show_ltree(&*lft), show_ltree(&*rgt)) }
      } 
    }
    LTree::Var { nam } => {
      nam.clone()
    }
    LTree::Ref { nam } => {
      format!("@{}", val_to_name(*nam))
    }
    LTree::Num { val } => {
      format!("#{}", (*val as u32).to_string())
    }
    LTree::Op2 { lft, rgt } => {
      format!("<{} {}>", show_ltree(&*lft), show_ltree(&*rgt))
    }
    LTree::Ite { sel, ret } => {
      format!("? {} {}", show_ltree(&*sel), show_ltree(&*ret))
    }
  }
}

pub fn show_lnet(lnet: &LNet) -> String {
  let mut result = String::new();
  result.push_str(&format!("{}", show_ltree(&lnet.root)));
  for (a, b) in &lnet.rdex {
    result.push_str(&format!("\n& {} ~ {}", show_ltree(a), show_ltree(b)));
  }
  return result;
}

pub fn show_net(net: &Net) -> String {
  show_lnet(&readback_lnet(net))
}

pub fn show_lbook(lbook: &LBook) -> String {
  let mut result = String::new();
  for (name, lnet) in lbook {
    result.push_str(&format!("{} = {}", name, show_lnet(lnet)));
  }
  return result;
}

pub fn show_book(book: &Book) -> String {
  show_lbook(&readback_lbook(book))
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
  let root = alloc_ltree(&mut net, &lnet.root, &mut vars, PARENT_ROOT);
  net.heap.set_root(root);
  for (tree1, tree2) in &lnet.rdex {
    let ptr1 = alloc_ltree(&mut net, tree1, &mut vars, Parent::Rdex);
    let ptr2 = alloc_ltree(&mut net, tree2, &mut vars, Parent::Rdex);
    net.rdex.push((ptr1, ptr2));
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
       0..= 9 => (letter - 0 + '0' as u8) as char,
      10..=35 => (letter - 10 + 'A' as u8) as char,
      36..=61 => (letter - 36 + 'a' as u8) as char,
      62      => '_',
      63      => '.',
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

// Injection and Readback
// ----------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Parent {
  Rdex,
  Node { val: Val, port: Port },
}

const PARENT_ROOT: Parent = Parent::Node { val: 0, port: P2 };

pub fn alloc_ltree(net: &mut Net, tree: &LTree, vars: &mut HashMap<String, Parent>, parent: Parent) -> Ptr {
  match tree {
    LTree::Era => {
      ERAS
    }
    LTree::Ctr { lab, lft, rgt } => {
      let val = net.heap.alloc(1);
      let p1 = alloc_ltree(net, &*lft, vars, Parent::Node { val, port: P1 });
      net.heap.set(val, P1, p1);
      let p2 = alloc_ltree(net, &*rgt, vars, Parent::Node { val, port: P2 });
      net.heap.set(val, P2, p2);
      Ptr::new(*lab + CT0, val)
    }
    LTree::Var { nam } => {
      if let Parent::Rdex = parent {
        panic!("By definition, can't have variable on active pairs.");
      };
      match vars.get(nam) {
        Some(Parent::Rdex) => {
          unreachable!();
        }
        Some(Parent::Node { val: other_val, port: other_port }) => {
          //println!("linked {} | set {} {:?} as {} {:?}", nam, other_val, other_port, val, port);
          match parent {
            Parent::Rdex => { unreachable!(); }
            Parent::Node { val, port } => net.heap.set(*other_val, *other_port, Ptr::new(port_to_tag(port), val)),
          }
          return Ptr::new(port_to_tag(*other_port), *other_val);
        }
        None => {
          //println!("linkin {} | iam {} {:?}", nam, val, port);
          vars.insert(nam.clone(), parent);
          NULL
        }
      }
    }
    LTree::Ref { nam } => {
      Ptr::new(REF, *nam)
    }
    LTree::Num { val } => {
      Ptr::new(NUM, *val as Val)
    }
    LTree::Op2 { lft, rgt } => {
      let val = net.heap.alloc(1);
      let p1 = alloc_ltree(net, &*lft, vars, Parent::Node { val, port: P1 });
      net.heap.set(val, P1, p1);
      let p2 = alloc_ltree(net, &*rgt, vars, Parent::Node { val, port: P2 });
      net.heap.set(val, P2, p2);
      Ptr::new(OP2, val)
    }
    LTree::Ite { sel, ret } => {
      let val = net.heap.alloc(1);
      let p1 = alloc_ltree(net, &*sel, vars, Parent::Node { val, port: P1 });
      net.heap.set(val, P1, p1);
      let p2 = alloc_ltree(net, &*ret, vars, Parent::Node { val, port: P2 });
      net.heap.set(val, P2, p2);
      Ptr::new(ITE, val)
    }
  }
}

pub fn do_alloc_ltree(net: &mut Net, tree: &LTree) -> Ptr {
  alloc_ltree(net, tree, &mut HashMap::new(), PARENT_ROOT)
}

pub fn define(book: &mut Book, name: &str, code: &str) -> Val {
  let id = name_to_val(name);
  book.def(id, lnet_to_net(&do_parse_lnet(code), None).to_def());
  return id;
}

pub fn lbook_to_book(lbook: &LBook) -> Book {
  let mut book = Book::new();
  for (name, lnet) in lbook {
    let id = name_to_val(name);
    book.def(id, lnet_to_net(lnet, None).to_def());
  }
  book
}

pub fn readback_ltree(net: &Net, ptr: Ptr, parent: Parent, vars: &mut HashMap<Parent, String>, fresh: &mut usize) -> LTree {
  match ptr.tag() {
    ERA => {
      LTree::Era
    }
    REF => {
      LTree::Ref { nam: ptr.val() }
    }
    NUM => {
      LTree::Num { val: ptr.val() as u32 }
    }
    OP1 | OP2 => {
      let lft = readback_ltree(net, net.heap.get(ptr.val(), P1), Parent::Node { val: ptr.val(), port: P1 }, vars, fresh);
      let rgt = readback_ltree(net, net.heap.get(ptr.val(), P2), Parent::Node { val: ptr.val(), port: P2 }, vars, fresh);
      LTree::Op2 { lft: Box::new(lft), rgt: Box::new(rgt) }
    }
    ITE => {
      let sel = readback_ltree(net, net.heap.get(ptr.val(), P1), Parent::Node { val: ptr.val(), port: P1 }, vars, fresh);
      let ret = readback_ltree(net, net.heap.get(ptr.val(), P2), Parent::Node { val: ptr.val(), port: P2 }, vars, fresh);
      LTree::Ite { sel: Box::new(sel), ret: Box::new(ret) }
    }
    VR1 | VR2 => {
      let key = match ptr.tag() {
        VR1 => Parent::Node { val: ptr.val(), port: P1 },
        VR2 => Parent::Node { val: ptr.val(), port: P2 },
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
    }
    _ => {
      let p1  = net.heap.get(ptr.val(), P1);
      let p2  = net.heap.get(ptr.val(), P2);
      let lft = readback_ltree(net, p1, Parent::Node { val: ptr.val(), port: P1 }, vars, fresh);
      let rgt = readback_ltree(net, p2, Parent::Node { val: ptr.val(), port: P2 }, vars, fresh);
      LTree::Ctr {
        lab: ptr.tag() - CT0,
        lft: Box::new(lft),
        rgt: Box::new(rgt),
      }
    }
  }
}

pub fn readback_lnet(net: &Net) -> LNet {
  let mut vars = HashMap::new();
  let mut fresh = 0;
  let mut rdex = Vec::new();
  let root = readback_ltree(net, net.heap.get_root(), PARENT_ROOT, &mut vars, &mut fresh);
  for &(a, b) in &net.rdex {
    let tree_a = readback_ltree(net, a, Parent::Rdex, &mut vars, &mut fresh);
    let tree_b = readback_ltree(net, b, Parent::Rdex, &mut vars, &mut fresh);
    rdex.push((tree_a, tree_b));
  }
  LNet { root, rdex }
}

pub fn readback_lbook(book: &Book) -> LBook {
  let mut lbook = BTreeMap::new();
  for id in 0 .. book.defs.len() {
    let def = &book.defs[id];
    if def.node.len() > 0 {
      let name = val_to_name(id as u32);
      let lnet = readback_lnet(&Net::from_def(def.clone()));
      lbook.insert(name, lnet);
    }
  }
  lbook
}
