// An interaction combinator language
// ----------------------------------
// This file implements a textual syntax to interact with the runtime. It includes a pure AST for
// nets, as well as functions for parsing, stringifying, and converting pure ASTs to runtime nets.
// On the runtime, a net is represented by a list of active trees, plus a root tree. The textual
// syntax reflects this representation. The grammar is specified on this repo's README.

use crate::run;
use crate::run::APtr;
use crate::run::Def;
use crate::run::Lab;
use crate::run::Ptr;
use crate::run::Tag;
use std::collections::BTreeMap;
use std::collections::{hash_map::Entry, HashMap};

use std::env::VarError;
use std::iter::Peekable;
use std::str::Chars;

// AST
// ---

#[derive(Clone, Hash, PartialEq, Eq, Debug, Default)]
pub enum Tree {
  #[default]
  Era,
  Ctr {
    lab: run::Lab,
    lft: Box<Tree>,
    rgt: Box<Tree>,
  },
  Var {
    nam: String,
  },
  Ref {
    nam: run::Val,
  },
  Num {
    val: run::Val,
  },
  Op2 {
    opr: run::Lab,
    lft: Box<Tree>,
    rgt: Box<Tree>,
  },
  Mat {
    sel: Box<Tree>,
    ret: Box<Tree>,
  },
}

type Redex = Vec<(Tree, Tree)>;

#[derive(Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct Net {
  pub root: Tree,
  pub rdex: Redex,
}

pub type Book = BTreeMap<String, Net>;

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
    } else if !c.is_ascii_whitespace() {
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

pub fn parse_decimal(chars: &mut Peekable<Chars>) -> Result<u64, String> {
  let mut num: u64 = 0;
  skip(chars);
  if !chars.peek().map_or(false, |c| c.is_digit(10)) {
    return Err(format!(
      "Expected a decimal number, found {:?}",
      chars.peek()
    ));
  }
  while let Some(c) = chars.peek() {
    if !c.is_digit(10) {
      break;
    }
    num = num * 10 + c.to_digit(10).unwrap() as u64;
    chars.next();
  }
  Ok(num)
}

pub fn parse_name(chars: &mut Peekable<Chars>) -> Result<String, String> {
  let mut txt = String::new();
  skip(chars);
  if !chars
    .peek()
    .map_or(false, |c| c.is_alphanumeric() || *c == '_' || *c == '.')
  {
    return Err(format!(
      "Expected a name character, found {:?}",
      chars.peek()
    ));
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

fn parse_opr(chars: &mut Peekable<Chars>) -> Result<run::Lab, String> {
  let opx = parse_opx_lit(chars)?;
  match opx.as_str() {
    "+" => Ok(run::ADD),
    "-" => Ok(run::SUB),
    "*" => Ok(run::MUL),
    "/" => Ok(run::DIV),
    "%" => Ok(run::MOD),
    "==" => Ok(run::EQ),
    "!=" => Ok(run::NE),
    "<" => Ok(run::LT),
    ">" => Ok(run::GT),
    "<=" => Ok(run::LTE),
    ">=" => Ok(run::GTE),
    "&&" => Ok(run::AND),
    "||" => Ok(run::OR),
    "^" => Ok(run::XOR),
    "!" => Ok(run::NOT),
    "<<" => Ok(run::LSH),
    ">>" => Ok(run::RSH),
    _ => Err(format!("Unknown operator: {}", opx)),
  }
}

pub fn parse_tree(chars: &mut Peekable<Chars>) -> Result<Tree, String> {
  skip(chars);
  match chars.peek() {
    Some('*') => {
      chars.next();
      Ok(Tree::Era)
    }
    Some('(') => {
      chars.next();
      let lft = Box::new(parse_tree(chars)?);
      let rgt = Box::new(parse_tree(chars)?);
      consume(chars, ")")?;
      Ok(Tree::Ctr { lab: 0, lft, rgt })
    }
    Some('[') => {
      chars.next();
      let lab = 1;
      let lft = Box::new(parse_tree(chars)?);
      let rgt = Box::new(parse_tree(chars)?);
      consume(chars, "]")?;
      Ok(Tree::Ctr { lab: 1, lft, rgt })
    }
    Some('{') => {
      chars.next();
      let lab = parse_decimal(chars)? as run::Lab;
      let lft = Box::new(parse_tree(chars)?);
      let rgt = Box::new(parse_tree(chars)?);
      consume(chars, "}")?;
      Ok(Tree::Ctr { lab, lft, rgt })
    }
    Some('@') => {
      chars.next();
      skip(chars);
      let name = parse_name(chars)?;
      Ok(Tree::Ref {
        nam: name_to_val(&name),
      })
    }
    Some('#') => {
      chars.next();
      Ok(Tree::Num {
        val: parse_decimal(chars)?,
      })
    }
    Some('<') => {
      chars.next();
      let opr = parse_opr(chars)?;
      let lft = Box::new(parse_tree(chars)?);
      let rgt = Box::new(parse_tree(chars)?);
      consume(chars, ">")?;
      Ok(Tree::Op2 { opr, lft, rgt })
    }
    Some('?') => {
      chars.next();
      consume(chars, "<")?;
      let sel = Box::new(parse_tree(chars)?);
      let ret = Box::new(parse_tree(chars)?);
      consume(chars, ">")?;
      Ok(Tree::Mat { sel, ret })
    }
    _ => Ok(Tree::Var {
      nam: parse_name(chars)?,
    }),
  }
}

pub fn parse_net(chars: &mut Peekable<Chars>) -> Result<Net, String> {
  let mut rdex = Vec::new();
  let root = parse_tree(chars)?;
  while let Some(c) = {
    skip(chars);
    chars.peek()
  } {
    if *c == '&' {
      chars.next();
      let tree1 = parse_tree(chars)?;
      consume(chars, "~")?;
      let tree2 = parse_tree(chars)?;
      rdex.push((tree1, tree2));
    } else {
      break;
    }
  }
  Ok(Net { root, rdex })
}

pub fn parse_book(chars: &mut Peekable<Chars>) -> Result<Book, String> {
  let mut book = BTreeMap::new();
  while let Some(c) = {
    skip(chars);
    chars.peek()
  } {
    if *c == '@' {
      chars.next();
      let name = parse_name(chars)?;
      consume(chars, "=")?;
      let net = parse_net(chars)?;
      book.insert(name, net);
    } else {
      break;
    }
  }
  Ok(book)
}

fn do_parse<T>(code: &str, parse_fn: impl Fn(&mut Peekable<Chars>) -> Result<T, String>) -> T {
  let chars = &mut code.chars().peekable();
  match parse_fn(chars) {
    Ok(result) => {
      if chars.next().is_none() {
        result
      } else {
        eprintln!("Unable to parse the whole input. Is this not an hvmc file?");
        std::process::exit(1);
      }
    }
    Err(err) => {
      eprintln!("{}", err);
      std::process::exit(1);
    }
  }
}

pub fn do_parse_tree(code: &str) -> Tree {
  do_parse(code, parse_tree)
}

pub fn do_parse_net(code: &str) -> Net {
  do_parse(code, parse_net)
}

pub fn do_parse_book(code: &str) -> Book {
  do_parse(code, parse_book)
}

// Stringifier
// -----------

fn show_opr(opr: run::Lab) -> String {
  match opr {
    run::ADD => "+".to_string(),
    run::SUB => "-".to_string(),
    run::MUL => "*".to_string(),
    run::DIV => "/".to_string(),
    run::MOD => "%".to_string(),
    run::EQ => "==".to_string(),
    run::NE => "!=".to_string(),
    run::LT => "<".to_string(),
    run::GT => ">".to_string(),
    run::LTE => "<=".to_string(),
    run::GTE => ">=".to_string(),
    run::AND => "&&".to_string(),
    run::OR => "||".to_string(),
    run::XOR => "^".to_string(),
    run::NOT => "!".to_string(),
    run::LSH => "<<".to_string(),
    run::RSH => ">>".to_string(),
    _ => panic!("Unknown operator label."),
  }
}

pub fn show_tree(tree: &Tree) -> String {
  match tree {
    Tree::Era => "*".to_string(),
    Tree::Ctr { lab, lft, rgt } => match lab {
      0 => format!("({} {})", show_tree(&*lft), show_tree(&*rgt)),
      1 => format!("[{} {}]", show_tree(&*lft), show_tree(&*rgt)),
      _ => format!("{{{} {} {}}}", lab, show_tree(&*lft), show_tree(&*rgt)),
    },
    Tree::Var { nam } => nam.clone(),
    Tree::Ref { nam } => {
      format!("@{}", val_to_name(*nam))
    }
    Tree::Num { val } => {
      format!("#{}", (*val).to_string())
    }
    Tree::Op2 { opr, lft, rgt } => {
      format!(
        "<{} {} {}>",
        show_opr(*opr),
        show_tree(&*lft),
        show_tree(&*rgt)
      )
    }
    Tree::Mat { sel, ret } => {
      format!("?<{} {}>", show_tree(&*sel), show_tree(&*ret))
    }
  }
}

pub fn show_net(net: &Net) -> String {
  let mut result = String::new();
  result.push_str(&format!("{}", show_tree(&net.root)));
  for (a, b) in &net.rdex {
    result.push_str(&format!("\n& {} ~ {}", show_tree(a), show_tree(b)));
  }
  return result;
}

pub fn show_book(book: &Book) -> String {
  let mut result = String::new();
  for (name, net) in book {
    result.push_str(&format!("@{} = {}\n", name, show_net(net)));
  }
  return result;
}

// pub fn show_runtime_tree(rt_net: &run::Net, ptr: run::Ptr) -> String {
//   show_tree(&tree_from_runtime_go(
//     rt_net,
//     ptr,
//     PARENT_ROOT,
//     &mut HashMap::new(),
//     &mut 0,
//   ))
// }

// pub fn show_runtime_net(rt_net: &run::Net) -> String {
//   show_net(&net_from_runtime(rt_net))
// }

// pub fn show_runtime_book(book: &run::Book) -> String {
//   show_book(&book_from_runtime(book))
// }

// Conversion
// ----------

pub fn num_to_str(mut num: usize) -> String {
  let mut txt = String::new();
  num += 1;
  while num > 0 {
    num -= 1;
    let c = ((num % 26) as u8 + b'a') as char;
    txt.push(c);
    num /= 26;
  }
  return txt.chars().rev().collect();
}

pub fn name_to_letters<'a>(name: &'a str) -> impl Iterator<Item = u8> + 'a {
  name.chars().map(|c| match c {
    '0'..='9' => c as u8 - '0' as u8 + 0,
    'A'..='Z' => c as u8 - 'A' as u8 + 10,
    'a'..='z' => c as u8 - 'a' as u8 + 36,
    '_' => 62,
    '.' => 63,
    _ => panic!("Invalid character in name"),
  })
}

pub fn letters_to_name(letters: impl Iterator<Item = u8>) -> String {
  let mut name = String::new();
  for letter in letters {
    name.push(match letter {
      0..=9 => (letter - 0 + '0' as u8) as char,
      10..=35 => (letter - 10 + 'A' as u8) as char,
      36..=61 => (letter - 36 + 'a' as u8) as char,
      62 => '_',
      63 => '.',
      _ => panic!("Invalid letter in name"),
    });
  }
  return name;
}

pub fn val_to_letters(num: run::Val) -> Vec<u8> {
  let mut letters = Vec::new();
  let mut num = num;
  while num > 0 {
    letters.push((num % 64) as u8);
    num /= 64;
  }
  letters.reverse();
  return letters;
}

pub fn letters_to_val(letters: impl Iterator<Item = u8>) -> run::Val {
  let mut num = 0;
  for letter in letters {
    num = num * 64 + letter as run::Val;
  }
  return num;
}

pub fn name_to_val(name: &str) -> run::Val {
  letters_to_val(name_to_letters(name))
}

pub fn val_to_name(num: run::Val) -> String {
  letters_to_name(val_to_letters(num).into_iter())
}

// Injection and Readback
// ----------------------

#[derive(Debug, Clone)]
pub struct Runtime {
  pub defs: HashMap<run::Val, Box<Def>>,
  pub back: HashMap<run::Loc, run::Val>,
}

type Key = run::Val;

impl Runtime {
  pub fn new(book: &Book) -> Runtime {
    let mut defs = calculate_min_safe_labels(book)
      .map(|(nam, lab)| {
        (
          nam,
          Box::new(Def {
            lab,
            comp: None,
            root: run::NULL,
            rdex: vec![],
            node: vec![],
          }),
        )
      })
      .collect::<HashMap<_, _>>();

    for (nam, net) in book.iter() {
      net_to_runtime_def(book, &mut defs, name_to_val(nam), net);
    }

    let back = defs
      .iter()
      .map(|(&nam, def)| (Ptr::new_ref(def).loc(), nam))
      .collect();

    Runtime { defs, back }
  }
  pub fn readback(&self, rt_net: &run::Net) -> Net {
    let mut state = State {
      runtime: self,
      vars: Default::default(),
      next_var: 0,
    };
    let mut net = Net::default();

    net.root = state.read_dir(rt_net.root);

    for &(a, b) in &rt_net.rdex {
      net
        .rdex
        .push((state.read_ptr(a, None), state.read_ptr(b, None)))
    }

    return net;

    struct State<'a> {
      runtime: &'a Runtime,
      vars: HashMap<run::Loc, usize>,
      next_var: usize,
    }

    impl<'a> State<'a> {
      fn read_dir(&mut self, dir: run::Loc) -> Tree {
        let ptr = dir.target().load();
        self.read_ptr(ptr, Some(dir))
      }
      fn read_ptr(&mut self, ptr: run::Ptr, dir: Option<run::Loc>) -> Tree {
        match ptr.tag() {
          Tag::Var => Tree::Var {
            nam: num_to_str(self.vars.remove(&dir.unwrap()).unwrap_or_else(|| {
              let nam = self.next_var;
              self.next_var += 1;
              self.vars.insert(ptr.loc(), nam);
              nam
            })),
          },
          Tag::Red => self.read_dir(ptr.loc()),
          Tag::Ref if ptr == run::ERA => Tree::Era,
          Tag::Ref => Tree::Ref {
            nam: self.runtime.back[&ptr.loc()],
          },
          Tag::Num => Tree::Num { val: ptr.num() },
          Tag::Op2 | Tag::Op1 => Tree::Op2 {
            opr: ptr.lab(),
            lft: Box::new(self.read_dir(ptr.p1().loc())),
            rgt: Box::new(self.read_dir(ptr.p2().loc())),
          },
          Tag::Ctr => Tree::Ctr {
            lab: ptr.lab(),
            lft: Box::new(self.read_dir(ptr.p1().loc())),
            rgt: Box::new(self.read_dir(ptr.p2().loc())),
          },
          Tag::Mat => Tree::Mat {
            sel: Box::new(self.read_dir(ptr.p1().loc())),
            ret: Box::new(self.read_dir(ptr.p2().loc())),
          },
        }
      }
    }
  }
}

fn net_to_runtime_def(
  book: &Book,
  defs: &mut HashMap<run::Val, Box<Def>>,
  nam: run::Val,
  net: &Net,
) {
  let mut state = State {
    book,
    defs,
    scope: Default::default(),
    nodes: Default::default(),
    root: run::NULL,
  };

  enum Place {
    Ptr(Ptr),
    Redex,
    Root,
  }

  state.root = state.visit_tree(&net.root, Some(run::NULL));

  let rdex = net
    .rdex
    .iter()
    .map(|(a, b)| (state.visit_tree(a, None), state.visit_tree(b, None)))
    .collect();

  let root = state.root;
  let node = state.nodes;

  assert!(state.scope.is_empty());

  let def = defs.get_mut(&nam).unwrap();

  def.root = root;
  def.rdex = rdex;
  def.node = node;

  #[derive(Debug)]
  struct State<'a> {
    book: &'a Book,
    defs: &'a HashMap<run::Val, Box<Def>>,
    scope: HashMap<&'a str, run::Ptr>,
    nodes: Vec<run::Node>,
    root: Ptr,
  }

  impl<'a> State<'a> {
    fn visit_tree(&mut self, tree: &'a Tree, place: Option<Ptr>) -> Ptr {
      match tree {
        Tree::Era => run::ERA,
        Tree::Ref { nam } => Ptr::new_ref(&self.defs[nam]),
        Tree::Num { val } => Ptr::new_num(*val),
        Tree::Var { nam } => {
          let place = place.expect("cannot have variables in active pairs");
          match self.scope.entry(nam) {
            Entry::Occupied(e) => {
              let other = e.remove();
              if other == run::NULL {
                self.root = place;
              } else {
                let node = &mut self.nodes[other.loc().index()];
                if other.loc().port() == 0 {
                  node.0 = place;
                } else {
                  node.1 = place;
                }
              }
              other
            }
            Entry::Vacant(e) => {
              e.insert(place);
              run::NULL
            }
          }
        }
        Tree::Ctr { lab, lft, rgt } => self.node(Tag::Ctr, *lab, lft, rgt),
        Tree::Op2 { opr, lft, rgt } => self.node(Tag::Op2, *opr, lft, rgt),
        Tree::Mat { sel, ret } => self.node(Tag::Mat, 0, sel, ret),
      }
    }
    fn node(&mut self, tag: Tag, lab: Lab, lft: &'a Tree, rgt: &'a Tree) -> Ptr {
      let index = self.nodes.len();
      self.nodes.push(Default::default());
      self.nodes[index].0 =
        self.visit_tree(lft, Some(Ptr::new(Tag::Var, 0, run::Loc::local(index, 0))));
      self.nodes[index].1 =
        self.visit_tree(rgt, Some(Ptr::new(Tag::Var, 0, run::Loc::local(index, 1))));
      Ptr::new(tag, lab, run::Loc::local(index, 0))
    }
  }
}

fn calculate_min_safe_labels(book: &Book) -> impl Iterator<Item = (run::Val, Lab)> {
  let mut state = State {
    book,
    labels: HashMap::with_capacity(book.len()),
  };

  for name in book.keys() {
    state.visit_def(name_to_val(name));
  }

  return state.labels.into_iter().map(|(nam, lab)| match lab {
    LabelState::Done(lab) => (nam, lab),
    _ => unreachable!(),
  });

  #[derive(Debug, Clone, Copy)]
  enum LabelState {
    Cycle1,
    Cycle2,
    Done(Lab),
  }

  struct State<'a> {
    book: &'a Book,
    labels: HashMap<run::Val, LabelState>,
  }

  impl<'a> State<'a> {
    fn visit_def(&mut self, key: run::Val) -> Lab {
      let normative = match self.labels.entry(key) {
        Entry::Vacant(e) => {
          e.insert(LabelState::Cycle1);
          true
        }
        Entry::Occupied(mut e) => match *e.get() {
          LabelState::Cycle1 => {
            e.insert(LabelState::Cycle2);
            false
          }
          LabelState::Cycle2 => return 0,
          LabelState::Done(lab) => return lab,
        },
      };
      let mut lab = 0;
      let def = &self.book[&val_to_name(key)];
      self.visit_tree(&def.root, &mut lab);
      for (a, b) in &def.rdex {
        self.visit_tree(a, &mut lab);
        self.visit_tree(b, &mut lab);
      }
      if normative {
        self.labels.insert(key, LabelState::Done(lab));
      }
      lab
    }
    fn visit_tree(&mut self, tree: &'a Tree, out: &mut Lab) {
      match tree {
        Tree::Era | Tree::Var { .. } | Tree::Num { .. } => {}
        Tree::Ctr { lab, lft, rgt } => {
          *out = (*out).max(lab + 1);
          self.visit_tree(lft, out);
          self.visit_tree(rgt, out);
        }
        Tree::Ref { nam } => {
          *out = (*out).max(self.visit_def(*nam));
        }
        Tree::Op2 { lft, rgt, .. } => {
          self.visit_tree(lft, out);
          self.visit_tree(rgt, out);
        }
        Tree::Mat { sel, ret } => {
          self.visit_tree(sel, out);
          self.visit_tree(ret, out);
        }
      }
    }
  }
}

// To runtime

// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
// pub enum Parent {
//   Redex,
//   Node(run::Loc),
// }
// const PARENT_ROOT: Parent = Parent::Node {
//   loc: run::ROOT.loc(),
//   port: tag_to_port(run::ROOT.tag()),
// };

// pub fn tree_to_runtime_go(
//   rt_net: &mut run::Net,
//   tree: &Tree,
//   vars: &mut HashMap<String, Parent>,
//   parent: Parent,
// ) -> run::Ptr {
//   match tree {
//     Tree::Era => run::ERA,
//     Tree::Con { lft, rgt } => {
//       let loc = rt_net.alloc();
//       let p1 = tree_to_runtime_go(rt_net, &*lft, vars, Parent::Node { loc, port: run::P1 });
//       rt_net.heap.set(loc, run::P1, p1);
//       let p2 = tree_to_runtime_go(rt_net, &*rgt, vars, Parent::Node { loc, port: run::P2 });
//       rt_net.heap.set(loc, run::P2, p2);
//       run::Ptr::new(run::LAM, 0, loc)
//     }
//     Tree::Tup { lft, rgt } => {
//       let loc = rt_net.alloc();
//       let p1 = tree_to_runtime_go(rt_net, &*lft, vars, Parent::Node { loc, port: run::P1 });
//       rt_net.heap.set(loc, run::P1, p1);
//       let p2 = tree_to_runtime_go(rt_net, &*rgt, vars, Parent::Node { loc, port: run::P2 });
//       rt_net.heap.set(loc, run::P2, p2);
//       run::Ptr::new(run::TUP, 0, loc)
//     }
//     Tree::Dup { lab, lft, rgt } => {
//       let loc = rt_net.alloc();
//       let p1 = tree_to_runtime_go(rt_net, &*lft, vars, Parent::Node { loc, port: run::P1 });
//       rt_net.heap.set(loc, run::P1, p1);
//       let p2 = tree_to_runtime_go(rt_net, &*rgt, vars, Parent::Node { loc, port: run::P2 });
//       rt_net.heap.set(loc, run::P2, p2);
//       run::Ptr::new(run::DUP, *lab, loc)
//     }
//     Tree::Var { nam } => {
//       if let Parent::Redex = parent {
//         panic!("By definition, can't have variable on active pairs.");
//       };
//       match vars.get(nam) {
//         Some(Parent::Redex) => {
//           unreachable!();
//         }
//         Some(Parent::Node {
//           loc: other_loc,
//           port: other_port,
//         }) => {
//           match parent {
//             Parent::Redex => {
//               unreachable!();
//             }
//             Parent::Node { loc, port } => rt_net.heap.set(
//               *other_loc,
//               *other_port,
//               run::Ptr::new(port_to_tag(port), 0, loc),
//             ),
//           }
//           return run::Ptr::new(port_to_tag(*other_port), 0, *other_loc);
//         }
//         None => {
//           vars.insert(nam.clone(), parent);
//           run::NULL
//         }
//       }
//     }
//     Tree::Ref { nam } => run::Ptr::big(run::REF, *nam),
//     Tree::Num { val } => run::Ptr::big(run::NUM, *val),
//     Tree::Op2 { opr, lft, rgt } => {
//       let loc = rt_net.alloc();
//       let p1 = tree_to_runtime_go(rt_net, &*lft, vars, Parent::Node { loc, port: run::P1 });
//       rt_net.heap.set(loc, run::P1, p1);
//       let p2 = tree_to_runtime_go(rt_net, &*rgt, vars, Parent::Node { loc, port: run::P2 });
//       rt_net.heap.set(loc, run::P2, p2);
//       run::Ptr::new(run::OP2, *opr, loc)
//     }
//     Tree::Mat { sel, ret } => {
//       let loc = rt_net.alloc();
//       let p1 = tree_to_runtime_go(rt_net, &*sel, vars, Parent::Node { loc, port: run::P1 });
//       rt_net.heap.set(loc, run::P1, p1);
//       let p2 = tree_to_runtime_go(rt_net, &*ret, vars, Parent::Node { loc, port: run::P2 });
//       rt_net.heap.set(loc, run::P2, p2);
//       run::Ptr::new(run::MAT, 0, loc)
//     }
//   }
// }

// pub fn tree_to_runtime(rt_net: &mut run::Net, tree: &Tree) -> run::Ptr {
//   tree_to_runtime_go(rt_net, tree, &mut HashMap::new(), PARENT_ROOT)
// }

// pub fn net_to_runtime(rt_net: &mut run::Net, net: &Net) {
//   let mut vars = HashMap::new();
//   let root = tree_to_runtime_go(rt_net, &net.root, &mut vars, PARENT_ROOT);
//   rt_net.heap.set_root(root);
//   for (tree1, tree2) in &net.rdex {
//     let ptr1 = tree_to_runtime_go(rt_net, tree1, &mut vars, Parent::Redex);
//     let ptr2 = tree_to_runtime_go(rt_net, tree2, &mut vars, Parent::Redex);
//     rt_net.rdex.push((ptr1, ptr2));
//   }
// }

// pub fn book_to_runtime(book: &Book) -> run::Book {
//   let mut rt_book = run::Book::new();
//   for (name, net) in book {
//     let fid = name_to_val(name);
//     let data = run::Heap::init(1 << 16);
//     let mut rt = run::Net::new(&data);
//     net_to_runtime(&mut rt, net);
//     rt_book.def(fid, runtime_net_to_runtime_def(&rt));
//   }
//   rt_book
// }

// // Converts to a def.
// pub fn runtime_net_to_runtime_def(net: &run::Net) -> run::Def {
//   // Checks if a ptr blocks the fast dup-ref optimization
//   // FIXME: this is too restrictive; make more flexible
//   fn is_unsafe(ptr: run::Ptr) -> bool {
//     return ptr.is_dup() || ptr.is_ref();
//   }
//   let mut node = vec![];
//   let mut rdex = vec![];
//   let mut safe = true;
//   for i in 0..net.heap.data.len() {
//     let p1 = net.heap.get(node.len() as run::Loc, run::P1);
//     let p2 = net.heap.get(node.len() as run::Loc, run::P2);
//     if p1 != run::NULL || p2 != run::NULL {
//       node.push((p1, p2));
//     } else {
//       break;
//     }
//     // TODO: this is too restrictive and should be
//     if is_unsafe(p1) || is_unsafe(p2) {
//       safe = false;
//     }
//   }
//   for i in 0..net.rdex.len() {
//     let p1 = net.rdex[i].0;
//     let p2 = net.rdex[i].1;
//     if is_unsafe(p1) || is_unsafe(p2) {
//       safe = false;
//     }
//     rdex.push((p1, p2));
//   }
//   return run::Def { safe, rdex, node };
// }

// // Reads back from a def.
// pub fn runtime_def_to_runtime_net<'a>(data: &'a run::Data, def: &run::Def) -> run::Net<'a> {
//   let mut net = run::Net::new(&data);
//   for (i, &(p1, p2)) in def.node.iter().enumerate() {
//     net.heap.set(i as run::Loc, run::P1, p1);
//     net.heap.set(i as run::Loc, run::P2, p2);
//   }
//   net.rdex = def.rdex.clone();
//   net
// }

// pub fn tree_from_runtime_go(
//   rt_net: &run::Net,
//   ptr: run::Ptr,
//   parent: Parent,
//   vars: &mut HashMap<Parent, String>,
//   fresh: &mut usize,
// ) -> Tree {
//   match ptr.tag() {
//     run::ERA => Tree::Era,
//     run::REF => Tree::Ref { nam: ptr.val() },
//     run::NUM => Tree::Num { val: ptr.val() },
//     run::OP1 | run::OP2 => {
//       let opr = ptr.lab();
//       let lft = tree_from_runtime_go(
//         rt_net,
//         rt_net.heap.get(ptr.loc(), run::P1),
//         Parent::Node {
//           loc: ptr.loc(),
//           port: run::P1,
//         },
//         vars,
//         fresh,
//       );
//       let rgt = tree_from_runtime_go(
//         rt_net,
//         rt_net.heap.get(ptr.loc(), run::P2),
//         Parent::Node {
//           loc: ptr.loc(),
//           port: run::P2,
//         },
//         vars,
//         fresh,
//       );
//       Tree::Op2 {
//         opr,
//         lft: Box::new(lft),
//         rgt: Box::new(rgt),
//       }
//     }
//     run::MAT => {
//       let sel = tree_from_runtime_go(
//         rt_net,
//         rt_net.heap.get(ptr.loc(), run::P1),
//         Parent::Node {
//           loc: ptr.loc(),
//           port: run::P1,
//         },
//         vars,
//         fresh,
//       );
//       let ret = tree_from_runtime_go(
//         rt_net,
//         rt_net.heap.get(ptr.loc(), run::P2),
//         Parent::Node {
//           loc: ptr.loc(),
//           port: run::P2,
//         },
//         vars,
//         fresh,
//       );
//       Tree::Mat {
//         sel: Box::new(sel),
//         ret: Box::new(ret),
//       }
//     }
//     run::VR1 | run::VR2 => {
//       let key = match ptr.tag() {
//         run::VR1 => Parent::Node {
//           loc: ptr.loc(),
//           port: run::P1,
//         },
//         run::VR2 => Parent::Node {
//           loc: ptr.loc(),
//           port: run::P2,
//         },
//         _ => unreachable!(),
//       };
//       if let Some(nam) = vars.get(&key) {
//         Tree::Var { nam: nam.clone() }
//       } else {
//         let nam = num_to_str(*fresh);
//         *fresh += 1;
//         vars.insert(parent, nam.clone());
//         Tree::Var { nam }
//       }
//     }
//     run::LAM => {
//       let p1 = rt_net.heap.get(ptr.loc(), run::P1);
//       let p2 = rt_net.heap.get(ptr.loc(), run::P2);
//       let lft = tree_from_runtime_go(
//         rt_net,
//         p1,
//         Parent::Node {
//           loc: ptr.loc(),
//           port: run::P1,
//         },
//         vars,
//         fresh,
//       );
//       let rgt = tree_from_runtime_go(
//         rt_net,
//         p2,
//         Parent::Node {
//           loc: ptr.loc(),
//           port: run::P2,
//         },
//         vars,
//         fresh,
//       );
//       Tree::Con {
//         lft: Box::new(lft),
//         rgt: Box::new(rgt),
//       }
//     }
//     run::TUP => {
//       let p1 = rt_net.heap.get(ptr.loc(), run::P1);
//       let p2 = rt_net.heap.get(ptr.loc(), run::P2);
//       let lft = tree_from_runtime_go(
//         rt_net,
//         p1,
//         Parent::Node {
//           loc: ptr.loc(),
//           port: run::P1,
//         },
//         vars,
//         fresh,
//       );
//       let rgt = tree_from_runtime_go(
//         rt_net,
//         p2,
//         Parent::Node {
//           loc: ptr.loc(),
//           port: run::P2,
//         },
//         vars,
//         fresh,
//       );
//       Tree::Tup {
//         lft: Box::new(lft),
//         rgt: Box::new(rgt),
//       }
//     }
//     run::DUP => {
//       let p1 = rt_net.heap.get(ptr.loc(), run::P1);
//       let p2 = rt_net.heap.get(ptr.loc(), run::P2);
//       let lft = tree_from_runtime_go(
//         rt_net,
//         p1,
//         Parent::Node {
//           loc: ptr.loc(),
//           port: run::P1,
//         },
//         vars,
//         fresh,
//       );
//       let rgt = tree_from_runtime_go(
//         rt_net,
//         p2,
//         Parent::Node {
//           loc: ptr.loc(),
//           port: run::P2,
//         },
//         vars,
//         fresh,
//       );
//       Tree::Dup {
//         lab: ptr.lab(),
//         lft: Box::new(lft),
//         rgt: Box::new(rgt),
//       }
//     }
//     _ => {
//       unreachable!()
//     }
//   }
// }

// pub fn tree_from_runtime(rt_net: &run::Net, ptr: run::Ptr) -> Tree {
//   let mut vars = HashMap::new();
//   let mut fresh = 0;
//   tree_from_runtime_go(rt_net, ptr, PARENT_ROOT, &mut vars, &mut fresh)
// }

// pub fn net_from_runtime(rt_net: &run::Net) -> Net {
//   let mut vars = HashMap::new();
//   let mut fresh = 0;
//   let mut rdex = Vec::new();
//   let root = tree_from_runtime_go(
//     rt_net,
//     rt_net.heap.get_root(),
//     PARENT_ROOT,
//     &mut vars,
//     &mut fresh,
//   );
//   for &(a, b) in &rt_net.rdex {
//     let tree_a = tree_from_runtime_go(rt_net, a, Parent::Redex, &mut vars, &mut fresh);
//     let tree_b = tree_from_runtime_go(rt_net, b, Parent::Redex, &mut vars, &mut fresh);
//     rdex.push((tree_a, tree_b));
//   }
//   Net { root, rdex }
// }

// pub fn book_from_runtime(rt_book: &run::Book) -> Book {
//   let mut book = BTreeMap::new();
//   for (fid, def) in rt_book.defs.iter() {
//     if def.node.len() > 0 {
//       let name = val_to_name(*fid);
//       let data = run::Heap::init(def.node.len());
//       let net = net_from_runtime(&runtime_def_to_runtime_net(&data, &def));
//       book.insert(name, net);
//     }
//   }
//   book
// }
