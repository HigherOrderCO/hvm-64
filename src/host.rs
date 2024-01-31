use crate::{
  ast::{Book, Net, Tree},
  run::{self, Def, DefNet, DefType, Instruction, LabSet, Loc, Port, Tag, TrgId, Wire},
  util::num_to_str,
};
use std::collections::{hash_map::Entry, HashMap};

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
    let mut host = Host::default();
    host.insert_book(book);
    host
  }
  pub fn insert_book(&mut self, book: &Book) {
    self.defs.reserve(book.len());
    self.back.reserve(book.len());

    for (nam, labs) in calculate_label_sets(book) {
      let def = DefRef::Owned(Box::new(Def { labs, inner: DefType::Net(DefNet::default()) }));
      self.back.insert(Port::new_ref(&def).loc(), nam.to_owned());
      self.defs.insert(nam.to_owned(), def);
    }

    for (nam, net) in book.iter() {
      let net = net_to_runtime_def(&self.defs, net);
      match self.defs.get_mut(nam).unwrap() {
        DefRef::Owned(def) => def.inner = DefType::Net(net),
        DefRef::Static(_) => unreachable!(),
      }
    }
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

  state.instr.append(&mut state.end);

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

pub fn calculate_label_sets(book: &Book) -> impl Iterator<Item = (&str, LabSet)> {
  let mut state = State { book, labels: HashMap::with_capacity(book.len()) };

  for name in book.keys() {
    state.visit_def(name, None);
  }

  return state.labels.into_iter().map(|(nam, lab)| match lab {
    LabelState::Done(lab) => (nam, lab),
    _ => unreachable!(),
  });

  #[derive(Debug, Clone)]
  enum LabelState {
    Cycle1,
    Cycle2,
    Done(LabSet),
  }

  struct State<'a> {
    book: &'a Book,
    labels: HashMap<&'a str, LabelState>,
  }

  impl<'a> State<'a> {
    fn visit_def(&mut self, key: &'a str, out: Option<&mut LabSet>) {
      match self.labels.entry(key) {
        Entry::Vacant(e) => {
          e.insert(LabelState::Cycle1);
          self._visit_def(key, out, true);
        }
        Entry::Occupied(mut e) => match e.get_mut() {
          LabelState::Cycle1 => {
            e.insert(LabelState::Cycle2);
            self._visit_def(key, out, false);
          }
          LabelState::Cycle2 => {}
          LabelState::Done(labs) => {
            if let Some(out) = out {
              out.union(labs)
            }
          }
        },
      }
    }
    fn _visit_def(&mut self, key: &'a str, out: Option<&mut LabSet>, normative: bool) {
      let mut labs = LabSet::default();
      let def = &self.book[key];
      self.visit_tree(&def.root, &mut labs);
      for (a, b) in &def.rdex {
        self.visit_tree(a, &mut labs);
        self.visit_tree(b, &mut labs);
      }
      if let Some(out) = out {
        out.union(&labs);
      }
      if normative {
        self.labels.insert(key, LabelState::Done(labs));
      }
    }
    fn visit_tree(&mut self, tree: &'a Tree, out: &mut LabSet) {
      match tree {
        Tree::Era | Tree::Var { .. } | Tree::Num { .. } => {}
        Tree::Ctr { lab, lft, rgt } => {
          out.add(*lab);
          self.visit_tree(lft, out);
          self.visit_tree(rgt, out);
        }
        Tree::Ref { nam } => {
          self.visit_def(nam, Some(out));
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
