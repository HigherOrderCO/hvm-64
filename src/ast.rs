// An interaction combinator language
// ----------------------------------
// This file implements a textual syntax to interact with the runtime. It includes a pure AST for
// nets, as well as functions for parsing, stringifying, and converting pure ASTs to runtime nets.
// On the runtime, a net is represented by a list of active trees, plus a root tree. The textual
// syntax reflects this representation. The grammar is specified on this repo's README.

use crate::{
  ops::Op,
  run::{self, Def, DefNet, DefType, Instruction, Lab, Loc, Port, Tag, TrgId, Wire},
};
use std::collections::{hash_map::Entry, BTreeMap, HashMap};

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

struct Parser<'b> {
  input: &'b str,
}

impl<'i> Parser<'i> {
  fn peek_char(&self) -> Option<char> {
    self.input.chars().next()
  }
  fn advance_char(&mut self) -> Option<char> {
    let char = self.input.chars().next()?;
    self.input = &self.input[char.len_utf8() ..];
    Some(char)
  }
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
  fn consume(&mut self, text: &str) -> Result<(), String> {
    self.skip_trivia();
    let Some(rest) = self.input.strip_prefix(text) else {
      return Err(format!("Expected {:?}, found {:?}", text, self.input.split_ascii_whitespace().next().unwrap_or("")));
    };
    self.input = rest;
    Ok(())
  }
  fn parse_number(&mut self) -> Result<u64, String> {
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
  fn take_while(&mut self, mut f: impl FnMut(char) -> bool) -> &'i str {
    let len = self.input.chars().take_while(|&c| f(c)).map(char::len_utf8).sum();
    let (name, rest) = self.input.split_at(len);
    self.input = rest;
    name
  }
  fn parse_name(&mut self) -> Result<String, String> {
    let name = self.take_while(|c| c.is_alphanumeric() || c == '_' || c == '.');
    if name.len() == 0 {
      return Err(format!("Expected a name character, found {:?}", self.peek_char()));
    }
    Ok(name.to_owned())
  }
  fn parse_op(&mut self) -> Result<Op, String> {
    let op = self.take_while(|c| "+-=*/%<>|&^!?".contains(c));
    op.parse().map_err(|_| panic!("Unknown operator: {op:?}"))
  }
  fn parse_tree(&mut self) -> Result<Tree, String> {
    self.skip_trivia();
    match self.peek_char() {
      Some('*') => {
        self.advance_char();
        Ok(Tree::Era)
      }
      Some(char @ ('(' | '[' | '{')) => {
        self.advance_char();
        let lab = match char {
          '(' => 0,
          '[' => 1,
          '{' => self.parse_number()? as Lab,
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
      Some('@') => {
        self.advance_char();
        self.skip_trivia();
        let nam = self.parse_name()?;
        Ok(Tree::Ref { nam })
      }
      Some('#') => {
        self.advance_char();
        Ok(Tree::Num { val: self.parse_number()? })
      }
      Some('<') => {
        self.advance_char();
        let opr = self.parse_op()?;
        let lft = Box::new(self.parse_tree()?);
        let rgt = Box::new(self.parse_tree()?);
        self.consume(">")?;
        Ok(Tree::Op2 { opr, lft, rgt })
      }
      Some('?') => {
        self.advance_char();
        self.consume("<")?;
        let sel = Box::new(self.parse_tree()?);
        let ret = Box::new(self.parse_tree()?);
        self.consume(">")?;
        Ok(Tree::Mat { sel, ret })
      }
      _ => Ok(Tree::Var { nam: self.parse_name()? }),
    }
  }
  fn parse_net(&mut self) -> Result<Net, String> {
    let mut rdex = Vec::new();
    let root = self.parse_tree()?;
    while self.consume("&").is_ok() {
      let tree1 = self.parse_tree()?;
      self.consume("~")?;
      let tree2 = self.parse_tree()?;
      rdex.push((tree1, tree2));
    }
    Ok(Net { root, rdex })
  }
  fn parse_book(&mut self) -> Result<Book, String> {
    let mut book = BTreeMap::new();
    while self.consume("@").is_ok() {
      let name = self.parse_name()?;
      self.consume("=")?;
      let net = self.parse_net()?;
      book.insert(name, net);
    }
    Ok(book)
  }
}

fn parse<'i, T>(input: &'i str, parse_fn: impl Fn(&mut Parser<'i>) -> Result<T, String>) -> T {
  let mut parser = Parser { input };
  match parse_fn(&mut parser) {
    Ok(result) => {
      if parser.peek_char().is_none() {
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

pub fn parse_tree(code: &str) -> Tree {
  parse(code, Parser::parse_tree)
}

pub fn parse_net(code: &str) -> Net {
  parse(code, Parser::parse_net)
}

pub fn parse_book(code: &str) -> Book {
  parse(code, Parser::parse_book)
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
  pub fn insert(&mut self, name: &str, def: DefRef) {
    self.back.insert(Port::new_ref(&def).loc(), name.to_owned());
    self.defs.insert(name.to_owned(), def);
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
  let mut state =
    State { defs, scope: Default::default(), instr: Default::default(), end: Default::default(), next_index: 1 };

  state.visit_tree(&net.root, TrgId::new(0));

  net.rdex.iter().for_each(|(a, b)| state.visit_redex(a, b));

  assert!(state.scope.is_empty(), "unbound variables: {:?}", state.scope.keys());

  state.instr.extend(state.end.drain(..));

  return DefNet { instr: state.instr };

  #[derive(Debug)]
  struct State<'a> {
    defs: &'a HashMap<String, DefRef>,
    scope: HashMap<&'a str, TrgId>,
    instr: Vec<Instruction>,
    end: Vec<Instruction>,
    next_index: usize,
  }

  impl<'a> State<'a> {
    fn id(&mut self) -> TrgId {
      let i = self.next_index;
      self.next_index += 1;
      TrgId::new(i)
    }
    fn visit_redex(&mut self, a: &'a Tree, b: &'a Tree) {
      let (port, tree) = match (a, b) {
        (Tree::Era, t) | (t, Tree::Era) => (Port::ERA, t),
        (Tree::Ref { nam }, t) | (t, Tree::Ref { nam }) => (Port::new_ref(&self.defs[nam]), t),
        (Tree::Num { val }, t) | (t, Tree::Num { val }) => (Port::new_num(*val), t),
        (t, u) => {
          let av = self.id();
          let aw = self.id();
          let bv = self.id();
          let bw = self.id();
          self.next_index += 4;
          self.instr.push(Instruction::Wires { av, aw, bv, bw });
          self.end.push(Instruction::Link { a: aw, b: bw });
          self.visit_tree(t, av);
          self.visit_tree(u, bv);
          return;
        }
      };
      let trg = self.id();
      self.instr.push(Instruction::Const { port, trg });
      self.visit_tree(tree, trg);
    }
    fn visit_tree(&mut self, tree: &'a Tree, trg: TrgId) {
      match tree {
        Tree::Era => {
          self.instr.push(Instruction::Set { trg, port: Port::ERA });
        }
        Tree::Ref { nam } => {
          self.instr.push(Instruction::Set { trg, port: Port::new_ref(&self.defs[nam]) });
        }
        Tree::Num { val } => {
          self.instr.push(Instruction::Set { trg, port: Port::new_num(*val) });
        }
        Tree::Var { nam } => match self.scope.entry(nam) {
          Entry::Occupied(e) => {
            let other = e.remove();
            self.instr.push(Instruction::Link { a: other, b: trg });
          }
          Entry::Vacant(e) => {
            e.insert(trg);
          }
        },
        Tree::Ctr { lab, lft, rgt } => {
          let l = self.id();
          let r = self.id();
          self.instr.push(Instruction::Ctr { lab: *lab, trg, lft: l, rgt: r });
          self.visit_tree(lft, l);
          self.visit_tree(rgt, r);
        }
        Tree::Op2 { opr, lft, rgt } => {
          let l = self.id();
          let r = self.id();
          self.instr.push(Instruction::Op2 { op: *opr, trg, lft: l, rgt: r });
          self.visit_tree(lft, l);
          self.visit_tree(rgt, r);
        }
        Tree::Op1 { opr, lft, rgt } => {
          let r = self.id();
          self.instr.push(Instruction::Op1 { op: *opr, num: *lft, trg, rgt: r });
          self.visit_tree(rgt, r);
        }
        Tree::Mat { sel, ret } => {
          let l = self.id();
          let r = self.id();
          self.instr.push(Instruction::Mat { trg, lft: l, rgt: r });
          self.visit_tree(sel, l);
          self.visit_tree(ret, r);
        }
      }
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
