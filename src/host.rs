//! The runtime's host, which acts as a translation layer between the AST and
//! the runtime.

use crate::{
  ast::{Book, Net, Tree},
  run::{self, Addr, Def, Instruction, InterpretedDef, LabSet, Mode, Port, Tag, TrgId, Wire},
  util::create_var,
};
use std::{
  collections::{hash_map::Entry, HashMap},
  ops::{Deref, DerefMut, RangeFrom},
};

/// Stores a bidirectional mapping between names and runtime defs.
#[derive(Default)]
pub struct Host {
  /// the forward mapping, from a name to the runtime def
  pub defs: HashMap<String, DefRef>,
  /// the backward mapping, from the address of a runtime def to the name
  pub back: HashMap<Addr, String>,
}

/// A potentially-owned reference to a [`Def`]. Vitally, the address of the
/// `Def` is stable, even if the `DefRef` moves -- this is why
/// [`std::Borrow::Cow`] cannot be used here.
pub enum DefRef {
  Owned(Box<dyn DerefMut<Target = Def> + Send + Sync>),
  Static(&'static Def),
}

impl Deref for DefRef {
  type Target = Def;
  fn deref(&self) -> &Def {
    match self {
      DefRef::Owned(x) => x,
      DefRef::Static(x) => x,
    }
  }
}

impl Host {
  pub fn new(book: &Book) -> Host {
    let mut host = Host::default();
    host.insert_book(book);
    host
  }

  /// Converts all of the nets from the book into runtime defs, and inserts them
  /// into the host.
  pub fn insert_book(&mut self, book: &Book) {
    self.defs.reserve(book.len());
    self.back.reserve(book.len());

    // Because there may be circular dependencies, inserting the definitions
    // must be done in two phases:

    // First, we insert empty defs into the host. Even though their instructions
    // are not yet set, the address of the def will not change, meaning that
    // `net_to_runtime_def` can safely use `Port::new_def` on them.
    for (name, labs) in calculate_label_sets(book, self) {
      let def = DefRef::Owned(Box::new(Def::new(labs, InterpretedDef::default())));
      self.insert_def(name, def);
    }

    // Now that `defs` is fully populated, we can fill in the instructions of
    // each of the new defs.
    for (nam, net) in book.iter() {
      let instr = ast_net_to_instructions(&self.defs, net);
      match self.defs.get_mut(nam).unwrap() {
        DefRef::Owned(def) => def.downcast_mut::<InterpretedDef>().unwrap().data.instr = instr,
        DefRef::Static(_) => unreachable!(),
      }
    }
  }

  /// Inserts a singular def into the mapping.
  pub fn insert_def(&mut self, name: &str, def: DefRef) {
    self.back.insert(Port::new_ref(&def).addr(), name.to_owned());
    self.defs.insert(name.to_owned(), def);
  }

  /// Creates an ast tree from a wire in a runtime net.
  pub fn readback_tree(&self, wire: &run::Wire) -> Tree {
    ReadbackState { host: self, vars: Default::default(), var_id: 0 .. }.read_wire(wire.clone())
  }

  /// Creates an ast net from a runtime net.
  ///
  /// Note that vicious circles and disconnected subnets will not be in the
  /// resulting ast net, as it is impossible to read these back from the runtime
  /// net representation. In the case of vicious circles, this may result in
  /// unbound variables.
  pub fn readback<M: Mode>(&self, rt_net: &run::Net<M>) -> Net {
    let mut state = ReadbackState { host: self, vars: Default::default(), var_id: 0 .. };
    let mut net = Net::default();

    net.root = state.read_wire(rt_net.root.clone());

    for (a, b) in &rt_net.rdex {
      net.rdex.push((state.read_port(a.clone(), None), state.read_port(b.clone(), None)))
    }

    net
  }
}

/// See [`Host::readback`].
struct ReadbackState<'a> {
  host: &'a Host,
  vars: HashMap<Addr, usize>,
  var_id: RangeFrom<usize>,
}

impl<'a> ReadbackState<'a> {
  /// Reads a tree out from a given `wire`.
  fn read_wire(&mut self, wire: Wire) -> Tree {
    let port = wire.load_target();
    self.read_port(port, Some(wire))
  }
  /// Reads a tree out from a given `port`. If this is a var port, the
  /// `wire` this port was reached from must be supplied to key into the
  /// `vars` map.
  fn read_port(&mut self, port: Port, wire: Option<Wire>) -> Tree {
    match port.tag() {
      Tag::Var => {
        let key = wire.unwrap().addr().min(port.addr());
        Tree::Var {
          nam: create_var(match self.vars.entry(key) {
            Entry::Occupied(e) => e.remove(),
            Entry::Vacant(e) => *e.insert(self.var_id.next().unwrap()),
          }),
        }
      }
      Tag::Red => self.read_wire(port.wire()),
      Tag::Ref if port == Port::ERA => Tree::Era,
      Tag::Ref => Tree::Ref { nam: self.host.back[&port.addr()].clone() },
      Tag::Num => Tree::Num { val: port.num() },
      Tag::Op2 => {
        let opr = port.op();
        let node = port.traverse_node();
        Tree::Op2 { opr, lft: Box::new(self.read_wire(node.p1)), rgt: Box::new(self.read_wire(node.p2)) }
      }
      Tag::Op1 => {
        let opr = port.op();
        let node = port.traverse_op1();
        Tree::Op1 { opr, lft: node.num.num(), rgt: Box::new(self.read_wire(node.p2)) }
      }
      Tag::Ctr => {
        let node = port.traverse_node();
        Tree::Ctr { lab: node.lab, lft: Box::new(self.read_wire(node.p1)), rgt: Box::new(self.read_wire(node.p2)) }
      }
      Tag::Mat => {
        let node = port.traverse_node();
        Tree::Mat { sel: Box::new(self.read_wire(node.p1)), ret: Box::new(self.read_wire(node.p2)) }
      }
    }
  }
}

/// Converts an ast net to a list of instructions to create the net.
///
/// `defs` must be populated with every `Ref` node that may appear in the net.
fn ast_net_to_instructions(defs: &HashMap<String, DefRef>, net: &Net) -> Vec<Instruction> {
  let mut state =
    State { defs, scope: Default::default(), instr: Default::default(), end: Default::default(), next_index: 1 };

  state.visit_tree(&net.root, TrgId::new(0));

  net.rdex.iter().for_each(|(a, b)| state.visit_redex(a, b));

  assert!(state.scope.is_empty(), "unbound variables: {:?}", state.scope.keys());

  state.instr.append(&mut state.end);

  return state.instr;

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
          self.instr.push(Instruction::LinkConst { trg, port: Port::ERA });
        }
        Tree::Ref { nam } => {
          self.instr.push(Instruction::LinkConst { trg, port: Port::new_ref(&self.defs[nam]) });
        }
        Tree::Num { val } => {
          self.instr.push(Instruction::LinkConst { trg, port: Port::new_num(*val) });
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

/// Calculates the labels used in each definition of a book.
///
/// # Background: Naive Algorithms
///
/// The simplest algorithm to calculate labels would be to go to each def,
/// recursively traverse the tree (going into references), and collect all of
/// the labels.
///
/// Now, this algorithm will not terminate for recursive definitions, but fixing
/// this is relatively simple: don't enter a reference twice in one traversal.
///
/// This modified algorithm will work in all cases, but it's not very efficient.
/// Notably, it can take quadratic time; for example, it requires traversing
/// each of these refs 3 times:
/// ```text
/// @foo = (* @bar)
/// @bar = (* @baz)
/// @baz = (* @foo)
/// ```
///
/// This can be resolved with memoization. However, a simple memoization pass
/// will not work here, due to the cycle avoidance algorithm. For example,
/// consider the following program:
/// ```text
/// @foo = (@bar {1 x x})
/// @bar = ({2 x x} @foo)
/// ```
///
/// A simple memoization pass would go as follows:
/// - calculate the labels for `@foo`
///   - encounter `@bar`
///     - this has not yet been processed, so calculate the labels for it
///       - encounter `{2 ...}`; add 2 to the label set
///       - encounter `@foo` -- this has already been visited, so skip it (cycle
///         avoidance)
///     - add all the labels from `@bar` (just 2) to `@foo`
///   - encounter `{1 ...}`; add 1 to the label set
/// - calculate the labels for `@bar` -- this has already been visited, so skip
///   it (memoization)
///
/// The end result of this is `@foo: {1, 2}, @bar: {2}` -- `@bar` is missing
/// `1`.
///
/// Instead, a two-phase memoization approach is needed.
///
/// # Implemented Here: Two-Phase Algorithm
///
/// Each phase follows a similar procedure as the simple memoization algorithm,
/// with a few modifications. (Henceforth, the *head node* of a cycle is the
/// first node in a cycle reached in a traversal.)
/// - in the first phase:
///   - when we process a node involved in a cycle:
///     - if it is not the head node of a cycle it is involved in, instead of
///       saving the computed label set to the memo, we store a placeholder
///       `Cycle` value into the memo
///     - otherwise, after storing the label set in the memo, we enter the
///       second phase, retraversing the node
///   - when we encounter a `Cycle` value in the memo, we treat it akin to an
///     empty label set
/// - in the second phase, we treat `Cycle` values as though they were missing
///   entries in the memo
///
/// In the first phase, to know when we are processing nodes in a cycle, we keep
/// track of the depth of the traversal, and every time we enter a ref, we store
/// `Cycle(depth)` into the memo -- so if we encounter this in the traversal of
/// the children, we know that that node is involved in a cycle. Additionally,
/// when we exit a node, we return the *head depth* of the result that was
/// calculated -- the least depth of a node involved in a cycle with the node
/// just processed. Comparing this with the depth gives us the information
/// needed for the special rules.
///
/// This algorithm runs in linear time (as refs are traversed at most twice),
/// and requires no more space than the naive algorithm.
fn calculate_label_sets<'a>(book: &'a Book, host: &Host) -> impl Iterator<Item = (&'a str, LabSet)> {
  let mut state = State { book, host, labels: HashMap::with_capacity(book.len()) };

  for name in book.keys() {
    state.visit_def(name, Some(0), None);
  }

  return state.labels.into_iter().map(|(nam, lab)| match lab {
    LabelState::Done(lab) => (nam, lab),
    _ => unreachable!(),
  });

  struct State<'a, 'b> {
    book: &'a Book,
    host: &'b Host,
    labels: HashMap<&'a str, LabelState>,
  }

  #[derive(Debug)]
  enum LabelState {
    Done(LabSet),
    /// Encountering this node indicates participation in a cycle with the given
    /// head depth.
    Cycle(usize),
  }

  /// All of these methods share a similar signature:
  /// - `depth` is optional; `None` indicates that this is the second processing
  ///   pass (where the depth is irrelevant, as all cycles have been detected)
  /// - `out`, if supplied, will be unioned with the result of this traversal
  /// - the return value indicates the head depth, as defined above (or an
  ///   arbitrary value `>= depth` if no cycles are involved)
  impl<'a, 'b> State<'a, 'b> {
    fn visit_def(&mut self, key: &'a str, depth: Option<usize>, out: Option<&mut LabSet>) -> usize {
      match self.labels.entry(key) {
        Entry::Vacant(e) => {
          e.insert(LabelState::Cycle(depth.unwrap()));
          self.calc_def(key, depth, out)
        }
        Entry::Occupied(mut e) => match e.get_mut() {
          LabelState::Done(labs) => {
            if let Some(out) = out {
              out.union(labs);
            }
            usize::MAX
          }
          LabelState::Cycle(d) if depth.is_some() => *d,
          LabelState::Cycle(_) => {
            e.insert(LabelState::Done(LabSet::default()));
            self.calc_def(key, depth, out)
          }
        },
      }
    }

    fn calc_def(&mut self, key: &'a str, depth: Option<usize>, out: Option<&mut LabSet>) -> usize {
      let mut labs = LabSet::default();
      let head_depth = self.visit_within_def(key, depth, Some(&mut labs));
      if let Some(out) = out {
        out.union(&labs);
      }
      if depth.is_some_and(|x| x > head_depth) {
        self.labels.insert(key, LabelState::Cycle(head_depth));
      } else {
        self.labels.insert(key, LabelState::Done(labs));
        if depth == Some(head_depth) {
          self.visit_within_def(key, None, None);
        }
      }
      head_depth
    }

    fn visit_within_def(&mut self, key: &str, depth: Option<usize>, mut out: Option<&mut LabSet>) -> usize {
      let def = &self.book[key];
      let mut head_depth = self.visit_tree(&def.root, depth, out.as_deref_mut());
      for (a, b) in &def.rdex {
        head_depth = head_depth.min(self.visit_tree(a, depth, out.as_deref_mut()));
        head_depth = head_depth.min(self.visit_tree(b, depth, out.as_deref_mut()));
      }
      head_depth
    }

    fn visit_tree(&mut self, tree: &'a Tree, depth: Option<usize>, mut out: Option<&mut LabSet>) -> usize {
      match tree {
        Tree::Era | Tree::Var { .. } | Tree::Num { .. } => usize::MAX,
        Tree::Ctr { lab, lft, rgt } => {
          if let Some(out) = out.as_deref_mut() {
            out.add(*lab);
          }
          usize::min(self.visit_tree(lft, depth, out.as_deref_mut()), self.visit_tree(rgt, depth, out.as_deref_mut()))
        }
        Tree::Ref { nam } => {
          if let Some(def) = self.host.defs.get(nam) {
            if let Some(out) = out {
              out.union(&def.labs);
            }
            usize::MAX
          } else {
            self.visit_def(nam, depth.map(|x| x + 1), out)
          }
        }
        Tree::Op1 { rgt, .. } => self.visit_tree(rgt, depth, out),
        Tree::Op2 { lft, rgt, .. } | Tree::Mat { sel: lft, ret: rgt } => {
          usize::min(self.visit_tree(lft, depth, out.as_deref_mut()), self.visit_tree(rgt, depth, out.as_deref_mut()))
        }
      }
    }
  }
}

#[test]
fn test_calculate_labels() {
  use std::collections::BTreeMap;
  assert_eq!(
    calculate_label_sets(
      &"
        @a = {0 @b @c}
        @b = {1 @a *}
        @c = {2 @a *}

        @p = {3 @q {4 x x}}
        @q = {5 @r @t}
        @r = {6 @s *}
        @s = {7 @r *}
        @t = {8 @q {9 @t @u}}
        @u = {10 @u @s}
      "
      .parse()
      .unwrap(),
      &Host::default(),
    )
    .collect::<BTreeMap<_, _>>(),
    [
      ("a", [0, 1, 2].into_iter().collect()),
      ("b", [0, 1, 2].into_iter().collect()),
      ("c", [0, 1, 2].into_iter().collect()),
      //
      ("p", [3, 4, 5, 6, 7, 8, 9, 10].into_iter().collect()),
      ("q", [5, 6, 7, 8, 9, 10].into_iter().collect()),
      ("r", [6, 7].into_iter().collect()),
      ("s", [6, 7].into_iter().collect()),
      ("t", [5, 6, 7, 8, 9, 10].into_iter().collect()),
      ("u", [6, 7, 10].into_iter().collect()),
    ]
    .into_iter()
    .collect()
  );
}
