// An interaction combinator language
// ----------------------------------
// This file implements a textual syntax to interact with the runtime. It includes a pure AST for
// nets, as well as functions for parsing, stringifying, and converting pure ASTs to runtime nets.
// On the runtime, a net is represented by a list of active trees, plus a root tree. The textual
// syntax reflects this representation. The grammar is specified on this repo's README.

use crate::{
  ops::Op,
  run::{self, Def, DefNet, DefType, Half, Lab, Loc, Port, Tag, Wire},
};
use std::collections::{hash_map::Entry, BTreeMap, HashMap};

use std::{iter::Peekable, str::Chars};

// AST
// ---

#[derive(Clone, Hash, PartialEq, Eq, Debug, Default)]
pub enum Tree {
  #[default]
  Era,
  Ctr {
    lab: Lab,
    lft: Box<Tree>,
    rgt: Box<Tree>,
  },
  Var {
    nam: String,
  },
  Ref {
    nam: String,
  },
  Num {
    val: u64,
  },
  Op2 {
    opr: Op,
    lft: Box<Tree>,
    rgt: Box<Tree>,
  },
  Op1 {
    opr: Op,
    lft: u64,
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
  Ok(())
}

pub fn parse_decimal(chars: &mut Peekable<Chars>) -> Result<u64, String> {
  let mut num: u64 = 0;
  skip(chars);
  if !chars.peek().map_or(false, |c| c.is_ascii_digit()) {
    return Err(format!("Expected a decimal number, found {:?}", chars.peek()));
  }
  while let Some(c) = chars.peek() {
    if !c.is_ascii_digit() {
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
  if !chars.peek().map_or(false, |c| c.is_alphanumeric() || *c == '_' || *c == '.') {
    return Err(format!("Expected a name character, found {:?}", chars.peek()));
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

fn parse_opr(chars: &mut Peekable<Chars>) -> Result<Op, String> {
  let opx = parse_opx_lit(chars)?;
  opx.parse().map_err(|_| format!("Unknown operator: {opx}"))
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
      let lft = Box::new(parse_tree(chars)?);
      let rgt = Box::new(parse_tree(chars)?);
      consume(chars, "]")?;
      Ok(Tree::Ctr { lab: 1, lft, rgt })
    }
    Some('{') => {
      chars.next();
      let lab = parse_decimal(chars)? as Lab;
      let lft = Box::new(parse_tree(chars)?);
      let rgt = Box::new(parse_tree(chars)?);
      consume(chars, "}")?;
      Ok(Tree::Ctr { lab, lft, rgt })
    }
    Some('@') => {
      chars.next();
      skip(chars);
      let nam = parse_name(chars)?;
      Ok(Tree::Ref { nam })
    }
    Some('#') => {
      chars.next();
      Ok(Tree::Num { val: parse_decimal(chars)? })
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
    _ => Ok(Tree::Var { nam: parse_name(chars)? }),
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

pub fn show_tree(tree: &Tree) -> String {
  match tree {
    Tree::Era => "*".to_string(),
    Tree::Ctr { lab, lft, rgt } => match lab {
      0 => format!("({} {})", show_tree(lft), show_tree(rgt)),
      1 => format!("[{} {}]", show_tree(lft), show_tree(rgt)),
      _ => format!("{{{} {} {}}}", lab, show_tree(lft), show_tree(rgt)),
    },
    Tree::Var { nam } => nam.clone(),
    Tree::Ref { nam } => {
      format!("@{}", nam)
    }
    Tree::Num { val } => {
      format!("#{}", (*val))
    }
    Tree::Op2 { opr, lft, rgt } => {
      format!("<{} {} {}>", opr, show_tree(lft), show_tree(rgt))
    }
    Tree::Op1 { opr, lft, rgt } => {
      format!("<{}{} {}>", lft, opr, show_tree(rgt))
    }
    Tree::Mat { sel, ret } => {
      format!("?<{} {}>", show_tree(sel), show_tree(ret))
    }
  }
}

pub fn show_net(net: &Net) -> String {
  let mut result = String::new();
  result.push_str(&show_tree(&net.root));
  for (a, b) in &net.rdex {
    result.push_str(&format!("\n& {} ~ {}", show_tree(a), show_tree(b)));
  }
  result
}

pub fn show_book(book: &Book) -> String {
  let mut result = String::new();
  for (name, net) in book {
    result.push_str(&format!("@{} = {}\n", name, show_net(net)));
  }
  result
}

#[derive(Debug, Clone)]
pub enum DefRef {
  Owned(Box<Def>),
  Static(&'static Def),
}

impl std::ops::Deref for DefRef {
  type Target = Def;
  fn deref(&self) -> &Def {
    match self {
      DefRef::Owned(x) => x,
      DefRef::Static(x) => x,
    }
  }
}

#[derive(Debug, Clone, Default)]
pub struct Host {
  pub defs: HashMap<String, DefRef>,
  pub back: HashMap<Loc, String>,
}

impl Host {
  pub fn new(book: &Book) -> Host {
    let mut defs = calculate_min_safe_labels(book)
      .map(|(nam, lab)| (nam.to_owned(), DefRef::Owned(Box::new(Def { lab, inner: DefType::Net(DefNet::default()) }))))
      .collect::<HashMap<_, _>>();

    for (nam, net) in book.iter() {
      let net = net_to_runtime_def(&defs, net);
      match defs.get_mut(nam).unwrap() {
        DefRef::Owned(def) => def.inner = DefType::Net(net),
        DefRef::Static(_) => unreachable!(),
      }
    }

    let back = defs.iter().map(|(nam, def)| (Port::new_ref(def).loc(), nam.clone())).collect();

    Host { defs, back }
  }
  pub fn readback(&self, rt_net: &run::Net) -> Net {
    let mut state = State { runtime: self, vars: Default::default(), next_var: 0 };
    let mut net = Net::default();

    net.root = state.read_dir(rt_net.root.clone());

    for (a, b) in &rt_net.rdex {
      net.rdex.push((state.read_ptr(a.clone(), None), state.read_ptr(b.clone(), None)))
    }

    return net;

    struct State<'a> {
      runtime: &'a Host,
      vars: HashMap<Loc, usize>,
      next_var: usize,
    }

    impl<'a> State<'a> {
      fn read_dir(&mut self, dir: Wire) -> Tree {
        let ptr = dir.load_target();
        self.read_ptr(ptr, Some(dir))
      }
      fn read_ptr(&mut self, ptr: Port, dir: Option<Wire>) -> Tree {
        match ptr.tag() {
          Tag::Var => Tree::Var {
            nam: num_to_str(self.vars.remove(&dir.unwrap().loc()).unwrap_or_else(|| {
              let nam = self.next_var;
              self.next_var += 1;
              self.vars.insert(ptr.loc(), nam);
              nam
            })),
          },
          Tag::Red => self.read_dir(ptr.wire()),
          Tag::Ref if ptr == Port::ERA => Tree::Era,
          Tag::Ref => Tree::Ref { nam: self.runtime.back[&ptr.loc()].clone() },
          Tag::Num => Tree::Num { val: ptr.num() },
          Tag::Op2 | Tag::Op1 => {
            let opr = ptr.op();
            let node = ptr.traverse_node();
            Tree::Op2 { opr, lft: Box::new(self.read_dir(node.p1)), rgt: Box::new(self.read_dir(node.p2)) }
          }
          Tag::Ctr => {
            let node = ptr.traverse_node();
            Tree::Ctr { lab: node.lab, lft: Box::new(self.read_dir(node.p1)), rgt: Box::new(self.read_dir(node.p2)) }
          }
          Tag::Mat => {
            let node = ptr.traverse_node();
            Tree::Mat { sel: Box::new(self.read_dir(node.p1)), ret: Box::new(self.read_dir(node.p2)) }
          }
        }
      }
    }
  }
}

fn net_to_runtime_def(defs: &HashMap<String, DefRef>, net: &Net) -> DefNet {
  let mut state = State { defs, scope: Default::default(), nodes: Default::default(), root: Port::FREE };

  state.root = state.visit_tree(&net.root, Some(Port::FREE));

  let rdex = net.rdex.iter().map(|(a, b)| (state.visit_tree(a, None), state.visit_tree(b, None))).collect();

  assert!(state.scope.is_empty());

  return DefNet { root: state.root, rdex, node: state.nodes };

  #[derive(Debug)]
  struct State<'a> {
    defs: &'a HashMap<String, DefRef>,
    scope: HashMap<&'a str, Port>,
    nodes: Vec<(Port, Port)>,
    root: Port,
  }

  impl<'a> State<'a> {
    fn visit_tree(&mut self, tree: &'a Tree, place: Option<Port>) -> Port {
      match tree {
        Tree::Era => Port::ERA,
        Tree::Ref { nam } => Port::new_ref(&self.defs[nam]),
        Tree::Num { val } => Port::new_num(*val),
        Tree::Var { nam } => {
          let place = place.expect("cannot have variables in active pairs");
          match self.scope.entry(nam) {
            Entry::Occupied(e) => {
              let other = e.remove();
              if other == Port::FREE {
                self.root = place;
              } else {
                let node = &mut self.nodes[other.loc().index()];
                match other.loc().half() {
                  Half::Left => node.0 = place,
                  Half::Right => node.1 = place,
                }
              }
              other
            }
            Entry::Vacant(e) => {
              e.insert(place);
              Port::FREE
            }
          }
        }
        Tree::Ctr { lab, lft, rgt } => self.node(Tag::Ctr, *lab, lft, rgt),
        Tree::Op2 { opr, lft, rgt } => self.node(Tag::Op2, *opr as Lab, lft, rgt),
        Tree::Op1 { opr, lft, rgt } => {
          let index = self.nodes.len();
          self.nodes.push(Default::default());
          self.nodes[index].0 = Port::new_num(*lft);
          self.nodes[index].1 = self.visit_tree(rgt, Some(Port::new(Tag::Var, 0, Loc::new_local(index, Half::Right))));
          Port::new(Tag::Op1, *opr as Lab, Loc::new_local(index, Half::Left))
        }
        Tree::Mat { sel, ret } => self.node(Tag::Mat, 0, sel, ret),
      }
    }
    fn node(&mut self, tag: Tag, lab: Lab, lft: &'a Tree, rgt: &'a Tree) -> Port {
      let index = self.nodes.len();
      self.nodes.push(Default::default());
      self.nodes[index].0 = self.visit_tree(lft, Some(Port::new(Tag::Var, 0, Loc::new_local(index, Half::Left))));
      self.nodes[index].1 = self.visit_tree(rgt, Some(Port::new(Tag::Var, 0, Loc::new_local(index, Half::Right))));
      Port::new(tag, lab, Loc::new_local(index, Half::Left))
    }
  }
}

pub fn calculate_min_safe_labels(book: &Book) -> impl Iterator<Item = (&str, Lab)> {
  let mut state = State { book, labels: HashMap::with_capacity(book.len()) };

  for name in book.keys() {
    state.visit_def(name);
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
    labels: HashMap<&'a str, LabelState>,
  }

  impl<'a> State<'a> {
    fn visit_def(&mut self, key: &'a str) -> Lab {
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
      let def = &self.book[key];
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
          *out = (*out).max(self.visit_def(nam));
        }
        Tree::Op2 { lft, rgt, .. } => {
          self.visit_tree(lft, out);
          self.visit_tree(rgt, out);
        }
        Tree::Op1 { rgt, .. } => {
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

pub fn num_to_str(mut num: usize) -> String {
  let mut txt = Vec::new();
  num += 1;
  while num > 0 {
    num -= 1;
    txt.push((num % 26) as u8 + b'a');
    num /= 26;
  }
  txt.reverse();
  String::from_utf8(txt).unwrap()
}
